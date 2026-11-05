#!/usr/bin/env sh
# Usage: sh start.sh   (PORT=8765 and HOST=127.0.0.1 by default)
set -eu
cd "$(dirname "$0")"
if [ ! -f .env ]; then
  cp .env.example .env
  echo 'Created .env - add ELEVENLABS_API_KEY (and ANTHROPIC_API_KEY for English).'
fi
command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null || { echo 'Install FFmpeg first.'; exit 1; }
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install -q -r backend/requirements-dev.txt
export PORT="${PORT:-8765}" HOST="${HOST:-127.0.0.1}"
echo "Bulgarian Subtitle API: http://$HOST:$PORT  (docs: /docs, Ctrl+C to stop)"
cd backend && exec ../.venv/bin/python -m app
