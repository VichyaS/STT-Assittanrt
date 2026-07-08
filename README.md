# 🤖 AI VICHYA — Intelligent Voice & Chat Assistant

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1%2B-lightgrey)](https://flask.palletsprojects.com)
[![OpenRouter](https://img.shields.io/badge/Powered%20by-OpenRouter-FF6B6B)](https://openrouter.ai)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**VICHYA (วิชญะ)** is a modern, production-ready AI voice and chat assistant with Thai language support. It features a beautiful web interface with real-time speech-to-text, AI chat, and text-to-speech capabilities.

---

## ✨ Features

- **🎤 Voice Input** — Record audio directly from the browser (MediaRecorder API)
- **🧠 AI Chat** — Powered by OpenRouter (GPT-4o-mini, Claude, Gemini, and more)
- **🔊 Voice Output** — Multi-layered TTS pipeline: Gemini Flash → Grok Voice → edge-tts → gTTS
- **🌐 Internet Search** — Toggle online search for real-time information
- **🇹🇭 Thai Language** — Full Thai language support for STT, chat, and TTS
- **🔒 Production Security** — Rate limiting, input validation, security headers, CORS control
- **📱 Responsive UI** — Dark-themed modern interface with waveform visualization

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [ffmpeg](https://ffmpeg.org/) (optional, for enhanced TTS quality)
- An [OpenRouter](https://openrouter.ai) API key

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/vichya.git
cd vichya

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### Configuration

Edit `.env` with your settings:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
CORS_ORIGIN=*                    # Set to your domain in production
ENABLE_HSTS=0                    # Set to 1 in production with HTTPS
FLASK_DEBUG=0                    # Set to 1 for development
```

### Run the Server

```bash
# Development
python vichya_server.py

# Production (via gunicorn)
gunicorn vichya_server:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --access-logfile -
```

Open **http://localhost:5000** in your browser.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Browser UI │────▶│  Flask API   │────▶│  OpenRouter │
│  (SPA)      │◀────│  (Backend)   │◀────│  (AI APIs)  │
└─────────────┘     └──────────────┘     └─────────────┘
                          │
                    ┌─────┴─────┐
                    │           │
               ┌────────┐ ┌────────┐
               │  STT   │ │  TTS   │
               │Whisper │ │Pipeline│
               └────────┘ └────────┘
```

### Components

| Component | Technology | Description |
|-----------|-----------|-------------|
| **Web UI** | HTML/CSS/JS (SPA) | Dark-themed chat interface with voice recording |
| **API Server** | Flask 3.1 | RESTful endpoints with rate limiting & security |
| **Chat AI** | OpenRouter API | Configurable model (default: GPT-4o-mini) |
| **STT** | Whisper (OpenRouter) | Speech-to-text with Thai language support |
| **TTS** | Multi-layer fallback | Gemini Flash → Grok Voice → edge-tts → gTTS |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/status` | Server status & model info |
| `POST` | `/api/chat` | Send text message, get AI response + audio |
| `POST` | `/api/transcribe` | Upload audio, get transcription |
| `POST` | `/api/full-process` | Server-side record → STT → Chat → TTS |

---

## 🔒 Security

- **Rate Limiting** — 30 requests per 60 seconds per IP
- **Input Validation** — Max 5000 characters per message
- **File Size Limits** — 10MB max request body, 20MB max audio
- **Security Headers** — X-Content-Type-Options, X-Frame-Options, Referrer-Policy
- **CORS Control** — Configurable origin via `CORS_ORIGIN` env var
- **No Secrets in Code** — API keys via environment variables only
- **Audio Format Whitelist** — Only wav, webm, ogg, mp3, m4a, flac allowed

---

## 🚢 Deployment

### Render / Heroku

This project includes `Procfile` and `runtime.txt` for easy deployment:

```bash
# Deploy to Render
# 1. Connect your GitHub repository
# 2. Set build command: pip install -r requirements.txt
# 3. Set start command: gunicorn vichya_server:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
# 4. Add environment variable: OPENROUTER_API_KEY
```

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "vichya_server:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]
```

---

## 🛠️ Development

### Project Structure

```
vichya/
├── vichya_server.py      # Flask web server (main application)
├── main.py               # CLI voice assistant (local use)
├── templates/
│   ├── index.html        # Web UI (SPA)
│   └── favicon.svg       # App icon
├── .env.example          # Environment template
├── .gitignore            # Git ignore rules
├── Procfile              # Deployment config
├── runtime.txt           # Python version
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

### CLI Voice Assistant

For local use without a browser:

```bash
pip install sounddevice scipy playsound==1.2.2
python main.py
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | ✅ Yes | — | OpenRouter API key |
| `CORS_ORIGIN` | No | `*` | Allowed CORS origin |
| `ENABLE_HSTS` | No | `0` | Enable HSTS headers |
| `FLASK_DEBUG` | No | `0` | Flask debug mode |

### AI Models

| Model | Default | Description |
|-------|---------|-------------|
| Chat | `openai/gpt-4o-mini` | Main conversation AI |
| STT | `openai/whisper-large-v3-turbo` | Speech-to-text |
| TTS (Primary) | `google/gemini-3.1-flash-tts-preview` | Text-to-speech |
| TTS (Alt) | `x-ai/grok-voice-tts-1.0` | Fallback TTS |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [OpenRouter](https://openrouter.ai) for unified AI API access
- [Flask](https://flask.palletsprojects.com) for the web framework
- All open-source TTS libraries (edge-tts, gTTS)
