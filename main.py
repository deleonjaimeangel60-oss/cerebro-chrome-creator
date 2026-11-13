import os, re, json, uuid, base64, shutil, threading, subprocess
from pathlib import Path

import requests
import yt_dlp
import anthropic
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_PUBLISHABLE_KEY"]
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_MIN = int(os.environ.get("MAX_VIDEO_MINUTES", "60"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "400"))
DATA = Path(os.environ.get("DATA_DIR", "/tmp/cerebro"))
(DATA / "clips").mkdir(parents=True, exist_ok=True)
(DATA / "work").mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=[ALLOWED_ORIGIN],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/files", StaticFiles(directory=DATA / "clips"), name="files")

JOBS = {}
LOCK = threading.Semaphore(1)  # un video a la vez para no saturar el servidor
YT_RE = re.compile(r"^https?://(www\.|m\.)?(youtube\.com/(watch\?v=|shorts/|live/)|youtu\.be/)[\w-]{6,}")


class JobIn(BaseModel):
    url: str
    duration: int = 30
    count: int = 6


def get_user(auth):
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(401, "Inicia sesión primero")
    try:
        r = requests.get(f"{SUPABASE_URL}/auth/v1/user",
                         headers={"apikey": SUPABASE_KEY, "Authorization": auth}, timeout=10)
    except Exception as e:
        raise HTTPException(401, f"No se pudo contactar a Supabase: {str(e)[:120]}")
    if r.status_code != 200:
        raise HTTPException(401, f"Sesión inválida (código {r.status_code}). Cierra sesión y vuelve a entrar.")
    return r.json()["id"]


@app.get("/")
def health():
    return {"ok": True}


@app.post("/jobs")
def create_job(body: JobIn, authorization: str = Header(None)):
    user = get_user(authorization)
    if not YT_RE.match(body.url.strip()):
        raise HTTPException(400, "Pega un enlace válido de YouTube")
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = {"id": jid, "user": user, "status": "queued", "step": "En cola…",
                 "progress": 2, "clips": [], "error": None}
    dur = max(15, min(body.duration, 90))
    cnt = max(1, min(body.count, 8))
    threading.Thread(target=run_job, args=(jid, body.url.strip(), dur, cnt), daemon=True).start()
    return {"id": jid}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...), duration: int = Form(30), count: int = Form(6),
                       authorization: str = Header(None)):
    user = get_user(authorization)
    ext = Path((file.filename or "video").lower()).suffix
    if ext not in (".mp4", ".mov", ".mkv", ".webm", ".m4v"):
        raise HTTPException(400, "Formato no admitido. Usa MP4, MOV, MKV o WEBM.")
    jid = uuid.uuid4().hex[:12]
    work = DATA / "work" / jid
    work.mkdir(parents=True, exist_ok=True)
    dest = work / ("video" + ext)
    size = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                f.close()
                shutil.rmtree(work, ignore_errors=True)
                raise HTTPException(413, f"El video pesa más de {MAX_UPLOAD_MB} MB.")
            f.write(chunk)
    JOBS[jid] = {"id": jid, "user": user, "status": "queued", "step": "En cola…",
                 "progress": 2, "clips": [], "error": None}
    dur = max(15, min(duration, 90))
    cnt = max(1, min(count, 8))
    threading.Thread(target=run_job, args=(jid, None, dur, cnt, dest), daemon=True).start()
    return {"id": jid}


@app.get("/jobs/{jid}")
def job_status(jid: str, authorization: str = Header(None)):
    user = get_user(authorization)
    job = JOBS.get(jid)
    if not job or job["user"] != user:
        raise HTTPException(404, "Trabajo no encontrado")
    return {k: v for k, v in job.items() if k != "user"}


def setp(jid, step, progress):
    JOBS[jid]["step"] = step
    JOBS[jid]["progress"] = progress


def run_job(jid, url, dur, cnt, local_video=None):
    work = DATA / "work" / jid
    work.mkdir(parents=True, exist_ok=True)
    try:
        with LOCK:
            JOBS[jid]["status"] = "running"
            if local_video:
                setp(jid, "Revisando el video…", 8)
                video = local_video
                total = probe_duration(video)
                if total > MAX_MIN * 60:
                    raise ValueError(f"El video dura más de {MAX_MIN} minutos. Usa uno más corto.")
            else:
                setp(jid, "Descargando el video…", 8)
                video, total = download(url, work)
            setp(jid, "Leyendo la transcripción…", 40)
            segs = load_transcript(work)
            if not segs:
                setp(jid, "Transcribiendo el audio (puede tardar unos minutos)…", 42)
                segs = whisper_transcript(video, work)
            if not segs:
                raise ValueError("No se pudo obtener la transcripción de este video.")
            setp(jid, "La IA busca los momentos virales…", 55)
            moments = pick_moments(segs, total, dur, cnt)
            if not moments:
                raise ValueError("La IA no encontró momentos claros. Prueba con otro video.")
            out = DATA / "clips" / jid
            out.mkdir(parents=True, exist_ok=True)
            clips = []
            for i, m in enumerate(moments):
                setp(jid, f"Cortando clip {i + 1} de {len(moments)}…", 65 + int(30 * i / len(moments)))
                name = f"clip{i + 1}.mp4"
                cut(video, m["start"], m["end"], out / name)
                clips.append({"url": f"/files/{jid}/{name}", "title": m["title"], "score": m["score"],
                              "reason": m["reason"], "start": m["start"], "end": m["end"]})
            JOBS[jid]["clips"] = clips
            JOBS[jid]["status"] = "done"
            setp(jid, "Listo", 100)
    except Exception as e:
        JOBS[jid]["status"] = "error"
        JOBS[jid]["error"] = friendly(e)
    finally:
        shutil.rmtree(work, ignore_errors=True)


COOKIES_BAD = False


def friendly(e):
    s = str(e)
    if "Sign in to confirm" in s or "not a bot" in s:
        if COOKIES_BAD:
            return ("YouTube bloqueó la descarga, y la variable YT_COOKIES_B64 de Railway no tiene un "
                    "archivo de cookies válido. Vuelve a generarla y pégala de nuevo.")
        return "YouTube bloqueó la descarga desde el servidor. Hay que configurar cookies (pídele ayuda a Claude)."
    if isinstance(e, ValueError):
        return s
    return "Error: " + s[:300]


def probe_duration(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(video)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def pick_lang(info):
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    for code in ("es", "en"):
        if code in manual:
            return code
    orig = [c for c in auto if c.endswith("-orig")]
    if orig:
        return orig[0]
    for code in ("es", "en"):
        if code in auto:
            return code
    return next(iter(manual), None)


def download(url, work):
    base = {"quiet": True, "noplaylist": True, "retries": 5, "extractor_retries": 3}
    b64 = os.environ.get("YT_COOKIES_B64")
    if b64:
        try:
            raw = base64.b64decode("".join(b64.split()), validate=True)
            text = raw.decode("utf-8")
            if "youtube" not in text.lower():
                raise ValueError("no parece un archivo de cookies de YouTube")
            cf = work / "cookies.txt"
            cf.write_text(text, encoding="utf-8")
            base["cookiefile"] = str(cf)
        except Exception:
            global COOKIES_BAD
            COOKIES_BAD = True  # se ignoran y se sigue sin cookies
    with yt_dlp.YoutubeDL({**base, "skip_download": True}) as y:
        info = y.extract_info(url, download=False)
    if info.get("is_live"):
        raise ValueError("Ese enlace es un directo en curso. Usa un video ya terminado.")
    total = info.get("duration") or 0
    if total > MAX_MIN * 60:
        raise ValueError(f"El video dura más de {MAX_MIN} minutos. Usa uno más corto.")
    # 1) el video, sin subtítulos
    opts = {**base, "outtmpl": str(work / "video.%(ext)s"),
            "format": "bv*[height<=720]+ba/b[height<=720]/b",
            "merge_output_format": "mp4"}
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([url])
    files = [f for f in work.glob("video.*") if f.suffix in (".mp4", ".mkv", ".webm")]
    if not files:
        raise ValueError("No se pudo descargar el video.")
    # 2) subtítulos: un solo idioma y sin romper si YouTube dice 429
    lang = pick_lang(info)
    if lang:
        try:
            sopts = {**base, "skip_download": True, "writesubtitles": True,
                     "writeautomaticsub": True, "subtitleslangs": [lang],
                     "subtitlesformat": "json3", "ignoreerrors": True,
                     "outtmpl": str(work / "video.%(ext)s")}
            with yt_dlp.YoutubeDL(sopts) as y:
                y.download([url])
        except Exception:
            pass
    return files[0], total


WHISPER = None


def whisper_transcript(video, work):
    global WHISPER
    from faster_whisper import WhisperModel
    wav = work / "audio.wav"
    r = subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("No se pudo extraer el audio: " + r.stderr[-200:])
    if WHISPER is None:
        WHISPER = WhisperModel(os.environ.get("WHISPER_MODEL", "base"), device="cpu", compute_type="int8")
    segments, _ = WHISPER.transcribe(str(wav), vad_filter=True)
    return [(s.start, s.text.strip()) for s in segments if s.text.strip()]


def load_transcript(work):
    files = sorted(work.glob("video*.json3"))
    if not files:
        return []
    pick = next((f for f in files if ".es" in f.name), files[0])
    data = json.loads(pick.read_text(encoding="utf-8"))
    segs = []
    for ev in data.get("events", []):
        if "segs" not in ev:
            continue
        text = "".join(s.get("utf8", "") for s in ev["segs"]).replace("\n", " ").strip()
        if text:
            segs.append((ev.get("tStartMs", 0) / 1000, text))
    return segs


def mmss(s):
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def pick_moments(segs, total, dur, cnt):
    lines, last = [], -99
    for t, text in segs:  # una línea con marca de tiempo cada ~6 s
        if t - last >= 6 or not lines:
            lines.append(f"[{mmss(t)}] {text}")
            last = t
        else:
            lines[-1] += " " + text
    transcript = "\n".join(lines)[:150000]
    prompt = f"""Eres un editor experto en clips virales para TikTok, Reels y YouTube Shorts.
Te paso la transcripción con marcas de tiempo [mm:ss] de un video de {int(total)} segundos.
Elige los {cnt} mejores momentos para clips de unos {dur} segundos (entre {max(10, dur - 10)} y {dur + 15}).
Reglas: gancho fuerte en los primeros 3 segundos; que se entienda sin contexto; empieza y termina en límites de frase; sin solapamientos.
Responde SOLO con un arreglo JSON, sin texto extra:
[{{"start": segundos, "end": segundos, "score": 0-100, "title": "titular corto en mayúsculas, máx 8 palabras", "reason": "por qué puede ser viral, una frase"}}]

Transcripción:
{transcript}"""
    client = anthropic.Anthropic()
    msg = client.messages.create(model=MODEL, max_tokens=2000,
                                 messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    items = json.loads(m.group(0))
    out = []
    for it in sorted(items, key=lambda x: -float(x.get("score", 0))):
        s, e = float(it["start"]), float(it["end"])
        s, e = max(0, s), min(total or e, e)
        if e - s < 8 or any(not (e <= o["start"] or s >= o["end"]) for o in out):
            continue
        out.append({"start": round(s, 1), "end": round(e, 1), "score": int(it.get("score", 70)),
                    "title": str(it.get("title", "Clip"))[:80], "reason": str(it.get("reason", ""))[:160]})
    return out[:cnt]


def cut(video, start, end, dest):
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", str(video), "-t", str(end - start),
           "-vf", "crop='min(iw,ih*9/16)':ih,scale=720:1280",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg falló: " + r.stderr[-200:])
