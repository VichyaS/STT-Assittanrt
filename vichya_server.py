"""
AI VICHYA - Intelligent Voice & Chat Assistant
Modern Web Interface with Flask Backend
Version 2.0 - 2026
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import time
import base64
import tempfile
import uuid
import traceback
import subprocess
import re
import asyncio
import requests
from datetime import datetime
from openai import OpenAI
from openai import OpenAI

app = Flask(__name__)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])

# Security: limit request size to 25MB (prevents DoS from huge uploads)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

# Simple in-memory rate limiting (prevents API abuse)
from collections import defaultdict
from threading import Lock
import time as _time

_rate_limit_store = defaultdict(list)
_rate_limit_lock = Lock()
RATE_LIMIT_REQUESTS = 30       # max requests
RATE_LIMIT_WINDOW = 60          # per 60 seconds

def _check_rate_limit():
    """Returns True if request is allowed, False if rate-limited."""
    client_ip = request.remote_addr or "unknown"
    now = _time.time()
    with _rate_limit_lock:
        timestamps = _rate_limit_store[client_ip]
        # Remove old entries outside the window
        timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        _rate_limit_store[client_ip] = timestamps
        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            return False
        timestamps.append(now)
        return True

# Ensure console output on Windows can handle Unicode emojis safely
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ==========================================
# 1. Configuration & API Setup
# ==========================================
# Load API key: environment variable (production) → k.py (local dev)
if os.environ.get("OPENROUTER_API_KEY"):
    OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    print("✅ API Key loaded from environment variable")
else:
    try:
        from k import API_KEY
        OPENROUTER_API_KEY = API_KEY
        print("✅ API Key loaded from k.py")
    except ImportError:
        OPENROUTER_API_KEY = "your-api-key-here"
        print("⚠️  API key not found. Set OPENROUTER_API_KEY env var or create k.py")

CHAT_MODEL = "openai/gpt-4o-mini"
STT_MODEL = "openai/whisper-large-v3-turbo"
TTS_MODEL = "google/gemini-3.1-flash-tts-preview"  # Gemini Flash TTS (via OpenRouter, PCM→MP3)
# Grok Voice TTS as alternative (supports MP3 natively, no ffmpeg needed)
TTS_MODEL_GROK = "x-ai/grok-voice-tts-1.0"
POST_CORRECT_MODEL = "google/gemini-2.5-flash"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SAMPLE_RATE = 44100
RECORD_DURATION = 5

# Locate ffmpeg for audio conversion
FFMPEG_PATH = None
# Render/Linux: ffmpeg is usually at /usr/bin/ffmpeg
for candidate in [
    "ffmpeg",
    "ffmpeg.exe",
    "/usr/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
]:
    try:
        subprocess.run([candidate, "-version"], capture_output=True, timeout=5, check=True)
        FFMPEG_PATH = candidate
        print(f"✅ ffmpeg found: {FFMPEG_PATH}")
        break
    except Exception:
        continue

if not FFMPEG_PATH:
    print("⚠️  ffmpeg not found — audio conversion disabled")

# ==========================================
# 2. Speech-to-Text via OpenRouter Whisper
# ==========================================
ALLOWED_AUDIO_FORMATS = {"wav", "webm", "ogg", "mp3", "m4a", "flac"}

def speech_to_text(audio_file_path: str) -> str:
    """Convert audio to text. Receives raw 16kHz WAV from browser → Whisper STT."""
    # Security: validate file path is a real file
    if not audio_file_path or not isinstance(audio_file_path, str):
        return ""
    if not os.path.exists(audio_file_path):
        print(f"🔇 File not found: {audio_file_path}")
        return ""

    file_size = os.path.getsize(audio_file_path)
    print(f"🎤 Transcribing audio: {audio_file_path} ({file_size} bytes)")

    # Skip tiny files
    if file_size < 200:
        print(f"🔇 Source too small ({file_size} bytes) — skipping")
        return ""

    # If it's already WAV, skip conversion
    if audio_file_path.lower().endswith(".wav"):
        wav_path = audio_file_path
        converted = False
    else:
        # Convert to 16kHz mono WAV
        wav_path = audio_file_path
        converted = False
        if FFMPEG_PATH:
            try:
                print("🔄 Converting to 16kHz mono WAV...")
                wav_path = audio_file_path + ".wav"
                result = subprocess.run(
                    [FFMPEG_PATH, "-y", "-i", audio_file_path,
                     "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
                     "-acodec", "pcm_s16le",
                     wav_path],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode != 0:
                    print(f"⚠️  ffmpeg stderr: {result.stderr[:300]}")
                    raise RuntimeError(f"ffmpeg failed with code {result.returncode}")
                converted = True
                print(f"✅ WAV: {os.path.getsize(wav_path)} bytes")
            except Exception as e:
                print(f"⚠️  ffmpeg failed: {e}, sending raw...")
        else:
            print("⚠️  ffmpeg not available, sending raw file...")

    try:
        # ── Quick check: skip tiny files ──
        wav_size = os.path.getsize(wav_path)
        if wav_size < 500:
            print(f"🔇 WAV too small ({wav_size} bytes) — skipping.")
            return ""

        # ── Silence detection disabled — RMS check unreliable on Python 3.14 ──
        # Let Whisper handle silence detection natively.

        # ── Validate audio format (security: restrict to known safe formats) ──
        audio_format = "wav"
        if not converted:
            lower_path = wav_path.lower()
            ext = os.path.splitext(lower_path)[1].lstrip(".")
            if ext in ALLOWED_AUDIO_FORMATS:
                audio_format = ext
            else:
                # If unknown extension, fall back to wav or send raw
                audio_format = "wav" if lower_path.endswith(".wav") else ext

        # Security: enforce max audio size (25MB raw)
        if wav_size > 20 * 1024 * 1024:
            print(f"🔇 Audio too large ({wav_size} bytes) — max 20MB")
            return ""

        print(f"🧾 Sending audio to STT: format={audio_format}, size={wav_size} bytes")
        # ── Whisper transcription ──
        with open(wav_path, "rb") as f:
            b64_audio = base64.b64encode(f.read()).decode("utf-8")

        json_payload = {
            "model": STT_MODEL,
            "input_audio": {"data": b64_audio, "format": audio_format},
            "language": "th",
            "temperature": 0.0,
        }
        print(f"📡 STT request payload format={audio_format}, size={len(b64_audio)} chars")
        resp = requests.post(
            "https://openrouter.ai/api/v1/audio/transcriptions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=json_payload,
            timeout=60,
        )

        if resp.status_code != 200:
            print(f"📡 STT Error {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()

        raw_text = resp.json().get("text", "").strip()
        print(f"📝 Raw STT: [{raw_text}]")

        if not raw_text:
            return ""

        # ── Hallucination filter: skip known silence/phrases ──
        known_hallucinations = [
            "โปรดติดตามตอนต่อไป",
            "สวัสดีครับ ทุกคน", "สวัสดีค่ะ ทุกคน",
            "สวัสดีครับทุกคน", "สวัสดีค่ะทุกคน",
            "โอเค โอเค โอเค",
            "thank you for watching", "thanks for watching",
        ]
        if raw_text in known_hallucinations or len(raw_text) < 3:
            print(f"🗑️  Filtered as silence/noise, returning empty")
            return ""

        # ── Skip LLM post-correction — it was causing more harm than good ──
        # Just return raw STT text directly
        print(f"✅ STT result: [{raw_text}]")
        return raw_text

    except Exception as e:
        print(f"❌ STT Error: {e}")
        print(f"❌ Full traceback:\n{traceback.format_exc()}")
        raise
    finally:
        if converted and os.path.exists(wav_path):
            os.unlink(wav_path)


def llm_correct_stt(text: str) -> str:
    """Use Gemini Pro to fix/correct STT output."""
    if not text or len(text) < 2:
        return ""

    try:
        resp = client.chat.completions.create(
            model=POST_CORRECT_MODEL,
            messages=[{
                "role": "system",
                "content": (
                    "คุณคือผู้ตรวจสอบคำพูดภาษาไทย (Thai ASR post-processor)\n"
                    "กฎสำคัญ:\n"
                    "1. ห้ามตัดทอนหรือทำให้ข้อความสั้นลงเด็ดขาด\n"
                    "2. ห้ามเปลี่ยนความหมายของข้อความ\n"
                    "3. ถ้าข้อความที่ได้เป็น 'โปรดติดตามตอนต่อไป' — "
                    "ให้ตอบว่า '...' (หมายถึงไม่มีเสียงพูดจริง)\n"
                    "4. ถ้ามีเสียงพูดจริง ให้ตรวจสอบว่าเป็นภาษาไทยที่อ่านได้หรือไม่ "
                    "ถ้าอ่านได้แล้วและความหมายตรง — ให้ตอบกลับเหมือนเดิมเป๊ะๆ\n"
                    "5. ถ้าพบคำที่สะกดผิดหรือฟังดูเพี้ยนเล็กน้อย — "
                    "ให้แก้เฉพาะคำนั้นเท่านั้น ห้ามตัดส่วนอื่นทิ้ง\n"
                    "6. ตอบเฉพาะข้อความที่แก้ไขแล้วเท่านั้น ห้ามอธิบายเพิ่ม"
                )
            }, {
                "role": "user",
                "content": f"ตรวจสอบข้อความนี้: {text}"
            }],

            temperature=0.0,
        )
        msg = resp.choices[0].message
        content = msg.content if msg and msg.content else ""
        if not content:
            return text
        result = content.strip()
        if result in ("...", "", "…"):
            return ""
        return result
    except Exception as e:
        print(f"⚠️ LLM correction failed: {e}, returning original")
        return text

# ==========================================
# 3. LLM Chat - Generate AI Response
# ==========================================

def clean_thai_symbols(text: str) -> str:
    """Remove only truly empty/malformed lines — preserve ALL Thai characters including tone marks."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) >= 2:
            has_content = False
            for ch in stripped:
                if '\u0e00' <= ch <= '\u0e7f' or '\u0000' <= ch <= '\u007f':
                    if ch.isalnum() or ch in '.,!?-_ ':
                        has_content = True
                        break
            if not has_content and len(stripped.replace(' ', '')) < 3:
                continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def generate_response(user_text: str, enable_search: bool = False) -> str:
    """Generate AI response using OpenRouter chat API.
    Optionally enables internet search via :online suffix."""
    print("🧠 VICHYA is thinking..." + (" 🌐 with internet search" if enable_search else ""))

    system_prompt = (
        "คุณคือ วิชญะ (VICHYA) ผู้ช่วย AI อัจฉริยะที่ทันสมัย "
        "คุณพูดคุยเหมือนมนุษย์ทั่วไป เป็นกันเอง อบอุ่น มีชีวิตชีวา "
        "ตอบด้วยภาษาไทยธรรมชาติ ไม่เป็นทางการเกินไป ไม่เป็นหุ่นยนต์ "
        "ตอบให้ละเอียด มีเนื้อหา ได้ใจความ มีตัวอย่างหรือขั้นตอนเมื่อเหมาะสม "
        "อย่าตอบสั้นเกินไป — ให้ข้อมูลที่มีประโยชน์และน่าสนใจ "
        "ใช้ภาษาไทยที่สมบูรณ์ ถูกต้อง อ่านง่าย "
        "หลีกเลี่ยงสัญลักษณ์หรือตัวอักษรที่อ่านไม่รู้เรื่อง "
        "ห้ามใช้อีโมจิในคำตอบเด็ดขาด "
        "ถ้าผู้ใช้ส่ง URL หรือลิงก์มา ให้ข้าม ไม่ต้องอ่านหรือตีความ "
        "ตอบให้กระชับภายในขีดจำกัด ไม่ต้องตอบยาวเกินความจำเป็น "
        "ถ้าคำถามต้องการข้อมูลเชิงลึก ให้วิเคราะห์อย่างเป็นขั้นตอน "
        "อย่าเสีย tokens ไปกับการอ่าน URL ที่ผู้ใช้ส่งมา"
        "อย่าอ่านข้อความ *.com หรือ *.org หรือ *.net หรือ *.io หรือ *.xyz"
    )

    model = CHAT_MODEL
    if enable_search:
        model = CHAT_MODEL + ":online"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.7,
    )

    reply = response.choices[0].message.content
    reply = clean_thai_symbols(reply)
    print(f"💬 VICHYA ({len(reply)} chars): {reply[:200]}...")
    return reply

# ==========================================
# ==========================================
# 4. Text-to-Speech via OpenRouter (Gemini 3.1 Flash TTS / Grok Voice / fallbacks)
# ==========================================
import emoji
def remove_emoji(text: str) -> str:
    return emoji.replace_emoji(text, replace='')

def text_to_speech(text: str, filename: str = "vichya_response.mp3") -> str:
    """Convert text to speech MP3 via OpenRouter TTS (Gemini Flash / Grok Voice / fallbacks).
    Pipeline: Gemini 3.1 Flash TTS → Grok Voice TTS → edge-tts → gTTS (last resort)
    Uses Gemini native voice 'Puck' (supports Thai well).
    """
    text = remove_emoji(text)
    if not text.strip():
        return filename

    # ── 1st: OpenRouter Gemini 3.1 Flash TTS (PCM → ffmpeg → MP3) ──
    if FFMPEG_PATH:
        try:
            pcm_path = filename + ".pcm"
            payload = {
                "model": TTS_MODEL,
                "input": text,
                "voice": "Puck",  # Google native voice that works with Thai
                "response_format": "pcm",
            }
            print(f"🔊 Gemini 3.1 Flash TTS ({len(text)} chars)...")
            resp = requests.post(
                "https://openrouter.ai/api/v1/audio/speech",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                json=payload, timeout=60,
            )
            if resp.status_code == 200:
                with open(pcm_path, "wb") as f:
                    f.write(resp.content)
                pcm_size = os.path.getsize(pcm_path)
                if pcm_size > 200:
                    result = subprocess.run(
                        [FFMPEG_PATH, "-y", "-f", "s16le", "-ar", "24000", "-ac", "1",
                         "-i", pcm_path, "-codec:a", "libmp3lame", "-b:a", "64k", filename],
                        capture_output=True, text=True, timeout=30,
                    )
                    os.unlink(pcm_path)
                    if result.returncode == 0:
                        mp3_size = os.path.getsize(filename)
                        print(f"✅ Audio saved (Gemini 3.1 Flash TTS → MP3): {filename} ({mp3_size} bytes)")
                        return filename
                    else:
                        print(f"⚠️  ffmpeg PCM→MP3 failed: {result.stderr[:200]}")
                else:
                    os.unlink(pcm_path)
                    print(f"⚠️  Gemini TTS returned tiny PCM ({pcm_size}b)")
            else:
                print(f"⚠️  Gemini TTS error {resp.status_code}: {resp.text[:150]}")
        except Exception as e:
            print(f"⚠️  Gemini TTS failed: {e}")
    else:
        print("⚠️  ffmpeg not found, skipping Gemini TTS (needs PCM→MP3 conversion)")

    # ── 2nd: Grok Voice TTS (supports MP3 natively, good multilingual) ──
    try:
        print("🔊 Grok Voice TTS...")
        payload = {
            "model": TTS_MODEL_GROK,
            "input": text,
            "voice": "Eve",
            "response_format": "mp3",
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/audio/speech",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json=payload, timeout=60,
        )
        if resp.status_code == 200:
            with open(filename, "wb") as f:
                f.write(resp.content)
            size = os.path.getsize(filename)
            if size > 200:
                print(f"✅ Audio saved (Grok Voice TTS): {filename} ({size} bytes)")
                return filename
            else:
                print(f"⚠️  Grok Voice returned tiny audio ({size}b)")
        else:
            print(f"⚠️  Grok TTS error {resp.status_code}: {resp.text[:150]}")
    except Exception as e:
        print(f"⚠️  Grok TTS failed: {e}")

    # ── 3rd: edge-tts (fast, natural Thai) ──
    try:
        import asyncio as _asyncio
        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)
        try:
            from edge_tts import Communicate
            voice = "th-TH-PremwadeeNeural"
            rate = "+10%"

            async def _speak():
                comm = Communicate(text, voice, rate=rate)
                await comm.save(filename)

            loop.run_until_complete(_speak())
            size = os.path.getsize(filename)
            print(f"✅ Audio saved (edge-tts): {filename} ({size} bytes)")
            return filename
        finally:
            loop.close()
    except ImportError:
        print("⚠️  edge-tts not installed, falling back to gTTS...")
    except Exception as e:
        print(f"⚠️  edge-tts failed: {e}, falling back to gTTS...")

    # ── 3rd: gTTS (last resort) ──
    from gtts import gTTS
    tts = gTTS(text=text, lang="th", slow=False)
    tts.save(filename)
    size = os.path.getsize(filename)
    print(f"✅ Audio saved (gTTS): {filename} ({size} bytes)")
    return filename

# ==========================================
# 5. API Endpoints
# ==========================================
@app.route("/")
def index():
    resp = app.make_response(render_template("index.html", ts=int(time.time())))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route("/favicon.svg")
def favicon():
    return send_from_directory(os.path.join(app.root_path, "templates"), "favicon.svg")

@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "online",
        "name": "VICHYA",
        "version": "2.0",
        "chat_model": CHAT_MODEL,
        "stt_model": STT_MODEL,
        "tts_model": TTS_MODEL,
        "tts_model_alt": TTS_MODEL_GROK,
        "timestamp": datetime.now().isoformat(),
    })

@app.route("/api/chat", methods=["POST"])
def api_chat():
    start = time.time()
    try:
        # Rate limiting check
        if not _check_rate_limit():
            return jsonify({"success": False, "error": "Rate limit exceeded. Try again later."}), 429

        data = request.get_json(force=True)
        message = (data or {}).get("message", "").strip()
        enable_search = (data or {}).get("search", False)

        # Input validation
        if not message or not isinstance(message, str):
            return jsonify({"success": False, "error": "No message provided"}), 400
        if len(message) > 5000:
            return jsonify({"success": False, "error": "Message too long (max 5000 chars)"}), 400
        if not isinstance(enable_search, bool):
            enable_search = False

        reply = generate_response(message, enable_search=enable_search)

        # Generate TTS audio
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.close()
        audio_path = text_to_speech(reply, tmp.name)
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        os.unlink(audio_path)

        elapsed = int((time.time() - start) * 1000)
        return jsonify({
            "success": True,
            "response": reply,
            "audio_base64": audio_b64,
            "response_time_ms": elapsed,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        print(f"❌ Chat error: {traceback.format_exc()}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    start = time.time()
    try:
        # Rate limiting check
        if not _check_rate_limit():
            return jsonify({"success": False, "error": "Rate limit exceeded. Try again later."}), 429

        data = request.get_json(force=True)
        audio_b64 = (data or {}).get("audio_base64", "")
        if not audio_b64 or not isinstance(audio_b64, str):
            return jsonify({"success": False, "error": "No audio data"}), 400

        # Security: validate base64 data size before decoding (prevents bomb attacks)
        if len(audio_b64) > 30 * 1024 * 1024:  # ~22MB raw
            return jsonify({"success": False, "error": "Audio data too large"}), 400

        # Write temp file — browser sends WAV
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        with open(tmp.name, "wb") as f:
            f.write(base64.b64decode(audio_b64))

        size = os.path.getsize(tmp.name)
        print(f"📁 Audio saved: {tmp.name} ({size} bytes) [browser WAV upload]")
        text = speech_to_text(tmp.name)
        os.unlink(tmp.name)

        elapsed = int((time.time() - start) * 1000)
        return jsonify({
            "success": True,
            "text": text,
            "response_time_ms": elapsed,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        print(f"❌ Transcription error: {traceback.format_exc()}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@app.route("/api/full-process", methods=["POST"])
def api_full_process():
    """Record from mic → STT → Chat → TTS (server-side mic required)."""
    start = time.time()
    sid = str(uuid.uuid4())[:8].upper()
    try:
        # Rate limiting check
        if not _check_rate_limit():
            return jsonify({"success": False, "error": "Rate limit exceeded.", "session_id": sid}), 429

        import sounddevice as sd
        from scipy.io import wavfile

        print(f"🎙️ [{sid}] Recording {RECORD_DURATION}s...")
        recording = sd.rec(
            int(RECORD_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        sd.wait()

        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav.close()
        wavfile.write(tmp_wav.name, SAMPLE_RATE, recording)

        with open(tmp_wav.name, "rb") as f:
            input_audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        user_text = speech_to_text(tmp_wav.name)
        os.unlink(tmp_wav.name)

        reply = generate_response(user_text)

        tmp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_mp3.close()
        audio_path = text_to_speech(reply, tmp_mp3.name)
        with open(audio_path, "rb") as f:
            output_audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        os.unlink(audio_path)

        elapsed = int((time.time() - start) * 1000)
        return jsonify({
            "success": True,
            "session_id": sid,
            "user_text": user_text,
            "response": reply,
            "input_audio_base64": input_audio_b64,
            "output_audio_base64": output_audio_b64,
            "response_time_ms": elapsed,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        print(f"❌ Full-process error: {traceback.format_exc()}")
        return jsonify({"success": False, "error": "Internal server error", "session_id": sid}), 500

# ==========================================
# 6. Main Entry Point
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  ⚡  AI VICHYA  v2.0  —  Intelligent Assistant")
    print("=" * 55)
    print(f"  Chat Model : {CHAT_MODEL}")
    print(f"  STT Model  : {STT_MODEL}")
    print(f"  TTS Model  : {TTS_MODEL} (voice: Puck)")
    print(f"  TTS Alt    : {TTS_MODEL_GROK} (voice: Eve)")
    print(f"  URL        : http://localhost:5000")
    print("=" * 55 + "\n")

    os.makedirs("templates", exist_ok=True)
    app.run(host="0.0.0.0", port=5000)