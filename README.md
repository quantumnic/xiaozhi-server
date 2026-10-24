# xiaozhi-server

Self-hosted backend server for [xiaozhi-esp32](https://github.com/78/xiaozhi-esp32) devices. Use **OpenAI GPT-4o** or **Anthropic Claude** as the brain — no xiaozhi.me account required.

## What Is This?

`xiaozhi-server` replaces the official xiaozhi.me cloud backend with your own server. It speaks the xiaozhi WebSocket protocol natively and routes everything through your own API keys:

- **Speech-to-Text** → OpenAI Whisper
- **LLM** → OpenAI GPT-4o _or_ Anthropic Claude
- **Text-to-Speech** → OpenAI TTS

Your ESP32 connects directly to your server. Full control, full privacy.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/quantumnic/xiaozhi-server.git
cd xiaozhi-server

# 2. Configure
cp .env.example .env
# Edit .env → add your OPENAI_API_KEY (required)

# 3. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[opus]"   # needs libopus (macOS: brew install opus, Debian: apt install libopus0)
# macOS/Homebrew: export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib so opuslib finds libopus

# 4. Run
python server.py
# → Server running on ws://0.0.0.0:8765
```

## ESP32 Configuration

On your xiaozhi-esp32 device, set the WebSocket server URL to:

```
ws://YOUR_SERVER_IP:8765/xiaozhi/v1/
```

If you've configured `AUTH_TOKEN` on the server, set the same token on the ESP32.

### Finding Your Server IP

```bash
# macOS
ipconfig getifaddr en0

# Linux
hostname -I | awk '{print $1}'
```

## LLM Configuration

### OpenAI (Default)

```env
OPENAI_API_KEY=sk-your-key-here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o          # or gpt-4o-mini for lower cost
```

### Anthropic Claude

```env
OPENAI_API_KEY=sk-your-key-here    # Still needed for Whisper STT + TTS
ANTHROPIC_API_KEY=sk-ant-your-key
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514  # or claude-opus-4-6
```

> **Note:** OpenAI API key is always required — Whisper (STT) and TTS run on OpenAI regardless of LLM provider.

## Docker

```bash
# Build
docker build -t xiaozhi-server .

# Run
docker run -d \
  --name xiaozhi-server \
  -p 8765:8765 \
  --env-file .env \
  xiaozhi-server
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | _(required)_ | OpenAI API key (for Whisper STT + TTS + optionally LLM) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (only if using Claude) |
| `LLM_PROVIDER` | `openai` | LLM provider: `openai` or `anthropic` |
| `LLM_MODEL` | auto | Model name (auto-selects `gpt-4o` or `claude-sonnet-4-20250514`) |
| `WHISPER_MODEL` | `whisper-1` | Whisper model for STT |
| `TTS_VOICE` | `alloy` | TTS voice: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |
| `TTS_MODEL` | `tts-1` | TTS model: `tts-1` or `tts-1-hd` |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8765` | Server port |
| `AUTH_TOKEN` | — | Optional Bearer token for ESP32 authentication |
| `SYSTEM_PROMPT` | _German default_ | System prompt for the LLM |
| `MAX_HISTORY` | `10` | Max conversation turns to keep in memory |

## Protocol

The server implements the xiaozhi WebSocket protocol:

```
ESP32 → hello → Server responds with hello + session_id
ESP32 → listen.start → Server starts collecting audio
ESP32 → binary opus frames → Server buffers audio
ESP32 → listen.stop → Server runs STT → LLM → TTS pipeline
Server → stt result → llm result → tts audio frames → ESP32
ESP32 → abort → Server cancels TTS immediately
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check .
```

## License

MIT
