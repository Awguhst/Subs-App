# Bulgarian Subtitle API

Upload an audio or video recording with Bulgarian speech; get back **Bulgarian and English subtitles in WebVTT, in one request**.

- **Speech-to-text:** ElevenLabs Scribe v2 with word timestamps, segmented into readable cues (two lines of at most 42 characters, split on pauses, speaker changes and sentences).
- **Translation:** Claude Haiku 4.5 translates every cue. The English track has exactly the same cue timings as the Bulgarian one.
- **Any device:** multipart form upload or raw bytes. The format is detected from the content: MP4, MOV, MKV, WebM, AVI, 3GP, MP3, M4A, AAC, WAV, FLAC, OGG/Opus, AMR, CAF, WMA and more.
- **Fast:** streaming uploads, ~1 s of audio extraction per 10 minutes of media, parallel translation batches, and instant cached results for repeat uploads.
- **Ready for Railway:** a single Dockerfile, `$PORT`, a `/health` check and persistent volume support. Interactive docs are at `/docs`, and an optional test page is at `/`.

```text
upload ──► FFprobe/FFmpeg ──► 16 kHz mono MP3 ──► ElevenLabs Scribe v2 ──► cues ──► bg.vtt
 (streamed to disk, hashed)                                                 └──► Claude Haiku 4.5 ──► en.vtt
```

## Quick start (local)

Requires Python 3.12+ and FFmpeg (`ffmpeg` and `ffprobe` on PATH).

```powershell
Copy-Item .env.example .env        # then set ELEVENLABS_API_KEY and ANTHROPIC_API_KEY
.\start.ps1 -Install               # later: .\start.ps1
```

On macOS/Linux, run `sh start.sh`. Then open **http://127.0.0.1:8765** for the test page, or **http://127.0.0.1:8765/docs** for the API documentation.

## Deploy to Railway

The repository deploys as-is: Railway builds the `Dockerfile` and the app listens on the `PORT` that Railway provides.

**Dashboard**

1. Create a project → **Deploy from GitHub repo** (push this folder to a repository first), or use the CLI below.
2. Under **Variables**, add:
   - `ELEVENLABS_API_KEY`: ElevenLabs secret key with Speech-to-Text access (starts with `sk_`).
   - `ANTHROPIC_API_KEY`: needed for the English track.
   - `API_KEYS`: one or more long random strings, comma-separated, that clients must send. **Set this.** Without it, anyone who finds the URL can spend your provider credit.
3. **Add a volume** mounted at **`/data`**. This keeps the translation spending ledger and the result cache across deploys. Without a volume, both reset on every deploy, and the startup log warns about it.
4. Under **Settings → Deploy**, set the healthcheck path to **`/health`**.
5. Under **Settings → Networking**, select **Generate Domain**. The API is now at `https://<name>.up.railway.app`.

**CLI**

```sh
railway login
railway init                       # create a project (or `railway link` to an existing one)
railway up --detach                # build and deploy this directory
railway variable set ELEVENLABS_API_KEY=sk_...
railway variable set ANTHROPIC_API_KEY=sk-ant-...
railway variable set API_KEYS=$(openssl rand -hex 24)
railway volume add --mount-path /data
railway domain                     # public HTTPS URL
railway logs
```

Then set the `/health` healthcheck path in the dashboard as in step 4. Railway's `railway.json` config-as-code files are deprecated: new services cannot use them, and existing ones stop being read on 2026-12-01. So this project ships no `railway.json`; the Dockerfile and the variables above are all Railway needs.

**Platform limits to know** ([Railway specs](https://docs.railway.com/networking/public-networking/specs-and-limits)):

| Limit | Consequence | What to do |
| --- | --- | --- |
| Upload must finish within 5 minutes | ≈ 375 MB at 10 Mbit/s | For big videos, upload the audio only (e.g. `ffmpeg -i in.mp4 -vn -c:a copy out.m4a`); the result is identical |
| Request closed after 5 minutes without data | Long transcriptions can exceed this | Use `?stream=true`, which sends progress every ≤15 s |
| 15 minutes maximum per request | Very long recordings | The server finishes the work anyway; fetch it with `GET /v1/subtitles/{id}` or upload again (cached, free) |

The image runs as a non-root user. Its entrypoint makes the volume writable, so `RAILWAY_RUN_UID=0` is not needed.

## API

Base URL: your deployment, e.g. `https://<name>.up.railway.app`. Interactive documentation is at `/docs` (Swagger UI with *Try it out*), with a reference at `/redoc` and the raw OpenAPI 3.1 spec at `/openapi.json`.

**Authentication.** When `API_KEYS` is set, every `/v1/*` endpoint needs `X-API-Key: <key>` or `Authorization: Bearer <key>`. `/health`, `/docs` and `/` are public.

### `POST /v1/subtitles`

Send the file as `multipart/form-data` in a field named `file`, or send the raw bytes as the body (any other `Content-Type`; optionally name the file with an `X-Filename` header).

| Query parameter | Default | Meaning |
| --- | --- | --- |
| `translate` | `true` | `false` returns Bulgarian only (no Anthropic cost) |
| `stream` | `false` | `true` returns NDJSON progress events and then the result (see below) |
| `format` | `json` | `vtt` returns one ready-to-use `.vtt` file instead of JSON (see below) |
| `language` | – | `bg` or `en`: which file to return when `format=vtt` |

#### Ready-to-use `.vtt` files: two requests

Make one request per language with the same file. Each response body is the subtitle file itself (`text/vtt`, with a `Content-Disposition` filename), ready to save. The second request is answered from the cache with no extra provider cost, and two requests sent at the same time share one transcription.

```bash
curl -X POST "$BASE_URL/v1/subtitles?format=vtt&language=bg" -H "X-API-Key: $API_KEY" -F "file=@recording.mp4" -o recording.bg.vtt
curl -X POST "$BASE_URL/v1/subtitles?format=vtt&language=en" -H "X-API-Key: $API_KEY" -F "file=@recording.mp4" -o recording.en.vtt
```

In Windows PowerShell, write `curl.exe` (plain `curl` is an alias for `Invoke-WebRequest` there). The files are saved byte-for-byte as UTF-8, so the Cyrillic text stays intact. Responses also carry `X-Subtitle-Id` and `X-Cue-Count` headers. If the English translation fails, the `language=en` request answers `502` with the reason, and the Bulgarian file stays available.

#### JSON: both tracks in one response (default)

```bash
curl -X POST "$BASE_URL/v1/subtitles" -H "X-API-Key: $API_KEY" -F "file=@recording.mp4"
```

```json
{
  "id": "7b0ef3c47bf23a2c4612e83fc4c0c0e5894eeb114cc260ac007e17681e17ca19",
  "filename": "recording.mp4",
  "media_type": "video",
  "duration_seconds": 312.4,
  "cue_count": 118,
  "subtitles": {
    "bg": "WEBVTT\n\n1\n00:00:01.200 --> 00:00:03.400\nЗдравейте на всички!\n\n2\n...",
    "en": "WEBVTT\n\n1\n00:00:01.200 --> 00:00:03.400\nHey everyone!\n\n2\n..."
  },
  "translation": { "status": "complete", "model": "claude-haiku-4-5-20251001", "error": null },
  "cached": false,
  "timings_ms": { "extract": 610, "transcribe": 18400, "segment": 4, "translate": 6900, "total": 25950 }
}
```

- `id` is the SHA-256 of the uploaded bytes. Uploading the same file again returns the stored result immediately, without any provider cost (`cached: true`), for `CACHE_TTL_HOURS`.
- `translation.status` is `complete`, `skipped` (not requested, or no speech) or `failed`. **If only the translation fails, the request still succeeds**: `subtitles.en` is `null` and `translation.error` says why.
- A recording without speech returns empty tracks (`"WEBVTT\n"`) and `cue_count: 0`.

| Status | Meaning |
| --- | --- |
| 400 | Empty or malformed upload |
| 401 | Missing or invalid API key |
| 413 | Larger than `MAX_UPLOAD_MB` or longer than `MAX_DURATION_MINUTES` |
| 415 | Not a supported audio/video file |
| 422 | No decodable audio track |
| 502 / 504 | ElevenLabs failed / timed out |
| 503 | `ELEVENLABS_API_KEY` is not configured |

Errors have the form `{"detail": "<message>"}`.

### Streaming progress (`?stream=true`)

The response is `application/x-ndjson`: one JSON object per line. It starts with `accepted`, sends `progress` events (at least every 15 s, which keeps proxies from closing the connection), and ends with exactly one `result` or `error` event:

```text
{"event": "accepted", "id": "7b0e…", "filename": "recording.mp4", "size_bytes": 48211033}
{"event": "progress", "stage": "extracting", "elapsed_seconds": 0.1}
{"event": "progress", "stage": "transcribing", "duration_seconds": 312.4, "elapsed_seconds": 0.7}
{"event": "progress", "stage": "translating", "completed": 2, "total": 3, "elapsed_seconds": 22.9}
{"event": "result", "result": { …same object as the JSON response… }}
```

The stages are `queued`, `probing`, `extracting`, `transcribing`, `segmenting` and `translating`. An `error` event carries `status` and `detail` like the HTTP errors above. HTTP status codes still apply to problems detected before processing starts (401, 413 on size, 503).

### Other endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/subtitles/{id}` | The finished result for an `id` (200), `202 {"status": "processing", "stage": …}` while running, 404 if unknown/expired |
| `GET /v1/subtitles/{id}/bg.vtt` · `/en.vtt` | One track as a downloadable `.vtt` file |
| `GET /v1/usage` | Translation spending against the lifetime cap |
| `GET /health` | Configuration and readiness (always 200 while the server is up) |

### Client examples

**Python**

```python
import json, requests

BASE, KEY = "https://<name>.up.railway.app", "<api key>"
with open("recording.mp4", "rb") as media:
    response = requests.post(f"{BASE}/v1/subtitles", headers={"X-API-Key": KEY},
                             files={"file": media}, timeout=900)
response.raise_for_status()
result = response.json()
open("recording.bg.vtt", "w", encoding="utf-8").write(result["subtitles"]["bg"])
open("recording.en.vtt", "w", encoding="utf-8").write(result["subtitles"]["en"] or "")

# Long recordings: follow the progress stream
with open("long.mp4", "rb") as media, requests.post(
        f"{BASE}/v1/subtitles", params={"stream": "true"}, headers={"X-API-Key": KEY},
        files={"file": media}, stream=True, timeout=900) as response:
    for line in response.iter_lines():
        event = json.loads(line)
        print(event["event"], event.get("stage", ""))
```

**JavaScript (browser or Node 18+)**

```js
const form = new FormData();
form.append("file", file);                       // a File/Blob, e.g. from <input type="file">
const response = await fetch(`${BASE}/v1/subtitles`, {
  method: "POST", headers: { "X-API-Key": KEY }, body: form,
});
if (!response.ok) throw new Error((await response.json()).detail);
const { subtitles } = await response.json();     // subtitles.bg, subtitles.en
```

**Raw bytes** (shell scripts, iOS Shortcuts, IoT devices):

```bash
curl -X POST "$BASE_URL/v1/subtitles" -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/octet-stream" -H "X-Filename: memo.m4a" --data-binary @memo.m4a
```

CORS is open (`CORS_ORIGINS=*`) by default, so web apps on any origin can call the API. Restrict it with a comma-separated list of origins if you prefer.

## Web test page

`/` serves a single static page ([frontend/index.html](frontend/index.html); no build step). It lets you upload or drag in a file, watch upload and processing progress, play the media with **Bulgarian / English / Off** subtitles, and copy or download each `.vtt`. The page asks for an API key when the server requires one and stores it in the browser. Set `ENABLE_WEB_UI=false` to serve only the API; `/` then redirects to `/docs`.

## Configuration

Set these as environment variables (Railway variables) or in `.env` for local runs. A local `.env` takes precedence over shell variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ELEVENLABS_API_KEY` | – | **Required.** ElevenLabs secret key with Speech-to-Text access |
| `ANTHROPIC_API_KEY` | – | Needed for English subtitles |
| `API_KEYS` | – | Comma-separated client keys; when empty the API is open (a startup warning is logged) |
| `MAX_UPLOAD_MB` | `2048` | Largest accepted upload |
| `MAX_DURATION_MINUTES` | `240` | Longest accepted recording; rejected before any paid call |
| `MAX_CONCURRENT_JOBS` | `2` | Recordings extracted and transcribed at the same time; others wait in `queued` |
| `TRANSLATION_CONCURRENCY` | `6` | Parallel Claude requests across all jobs |
| `TRANSLATION_BUDGET_USD` | `20` | Lifetime translation cap for this installation; can be lowered, never above 20 |
| `CACHE_TTL_HOURS` | `168` | How long results are reused; `0` disables the cache |
| `ELEVENLABS_TIMEOUT_SECONDS` | `1800` | Transcription read timeout |
| `ANTHROPIC_TIMEOUT_SECONDS` | `120` | Per translation request |
| `CORS_ORIGINS` | `*` | Allowed browser origins |
| `ENABLE_WEB_UI` | `true` | Serve the test page at `/` |
| `DATA_DIR` | `./data` (`/data` in Docker; the Railway volume path if one is attached) | Spending ledger, result cache and temporary uploads |
| `PORT` / `HOST` | `8000` / `0.0.0.0` | Listening address (Railway sets `PORT`) |
| `LOG_LEVEL` | `info` | Server log level |
| `FFMPEG_BIN` / `FFPROBE_BIN` | from PATH | Custom FFmpeg executables |

## Performance

- **Uploads** are parsed as a stream and written to disk once while being hashed, never held in memory. Oversized uploads are rejected as soon as they cross the limit. Locally a 500 MB upload is processed at ~175 MB/s, so the network is always the bottleneck.
- **Audio extraction** sends only what speech recognition needs: 16 kHz mono 64 kbit/s MP3, about 29 MB per hour of media whatever the video size, using LAME's fast mode (~0.8 s per 10 minutes of audio on an 8-core PC). The original upload is deleted before the transcription request starts.
- **Transcription** reuses one pooled HTTPS connection to ElevenLabs. For long media this is usually the longest stage.
- **Translation** sends cue text only (never timings) in batches of up to 40 cues, with neighbouring lines as context. Batches run concurrently over a pooled connection, and every request uses the same JSON schema, which Anthropic compiles once and caches. Measured live: three cues in 2.5 s for $0.0009.
- **Caching and de-duplication:** identical uploads are served from the result cache in milliseconds. Identical uploads arriving at the same time share one transcription.
- **Concurrency:** one async process. FFmpeg runs in worker threads, and the provider calls are non-blocking. Scale throughput with `MAX_CONCURRENT_JOBS` and `TRANSLATION_CONCURRENCY` (within your provider rate limits), not with extra workers.

## Costs and the translation spending cap

ElevenLabs bills transcription per audio minute to your account; check its pricing and limits there. Anthropic translation runs on **Claude Haiku 4.5** ($1 / $5 per million input/output tokens), pinned so the cost accounting stays exact.

Every Claude request first **reserves its worst-case cost** in a persistent ledger (`DATA_DIR/translation-budget.json`, or the earlier `DATA_DIR/jobs/.translation-budget.json` when it exists), then settles to the reported usage. A failed batch (rate limit, overload, server error, timeout or missing lines) gets up to three attempts in total, and only the missing lines are requested again. Rate-limited and overloaded attempts are not charged. Attempts whose outcome is unknown (timeouts, server errors, cancellations) keep their full reservation. When the next request would exceed `TRANSLATION_BUDGET_USD`, translation stops and only Bulgarian is returned (`translation.status: "failed"`). The ledger never resets by itself; do not edit or delete it to regain budget. `GET /v1/usage` shows the totals.

**Each installation has its own ledger.** The local Docker studio has used $0.55 of its $20 so far, and a Railway deployment starts its own $20 allowance. If you run both and want to stay within a €28 total, lower `TRANSLATION_BUDGET_USD` on one of them (for example to `8` on Railway). Railway's ledger only survives redeploys when a volume is attached.

## Security

- **Set `API_KEYS` on any public deployment.** Keys are compared in constant time, and requests are authenticated before any upload is read.
- Uploads are identified by content type, never trusted by filename. FFmpeg may only open real audio/video container formats from local files: playlists and indirection formats (HLS, DASH, concat) are refused, so a crafted upload cannot make FFmpeg read server files or URLs.
- The container runs FFmpeg and the API as an unprivileged user. Temporary files are deleted after each request, and leftovers are cleaned at startup.
- Provider keys stay on the server and are redacted from error messages.
- There is no per-client rate limiting. Use separate keys per client and watch `/v1/usage` and your provider dashboards.

## Local Docker

```sh
docker compose up --build -d          # http://127.0.0.1:8765, data in the studio-data volume
```

`compose.yaml` reuses the `studio-data` volume and the `studio` service name of the earlier Subtitle Studio. Running it therefore replaces the old container in place and keeps its translation ledger. The old studio's uploads, prepared videos and subtitle files stay untouched in the volume (under `/data/uploads`, `/data/outputs` and `/data/jobs`) but are no longer used. Set `STUDIO_BIND=0.0.0.0` to reach the API from phones on your network.

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest backend/tests -q
cd backend; ..\.venv\Scripts\python.exe -m app      # PORT/HOST from the environment
```

The 80 tests run real FFmpeg on synthetic media in the formats phones and browsers produce, with ElevenLabs and Claude replaced by offline fakes, so **tests spend no credits**. They cover:

- the HTTP API end to end: multipart and raw uploads, ready-to-use `.vtt` responses (including Cyrillic filenames), streaming events, caching and de-duplication, auth, limits, error mapping, and cleanup of temporary files;
- cue segmentation and WebVTT output;
- translation batching, partial-response recovery, retries and cancellation;
- the spending ledger;
- media detection and the refusal of playlists and non-media.

## Project layout

```text
backend/app/main.py         FastAPI app: routes, auth, OpenAPI docs, NDJSON streaming
backend/app/pipeline.py     Stage orchestration, result cache, de-duplication, progress events
backend/app/uploads.py      Streaming multipart/raw upload to disk with SHA-256
backend/app/media.py        FFprobe inspection and speech-audio extraction (format allow-list)
backend/app/elevenlabs.py   Async ElevenLabs Scribe v2 client
backend/app/subtitles.py    Word timestamps → cues → WebVTT; English line wrapping
backend/app/translation.py  Concurrent Claude Haiku 4.5 translation with cost reservation
backend/app/budget.py       Persistent lifetime spending ledger
backend/app/config.py       Environment settings
backend/tests/              Offline test suite (real FFmpeg, fake providers)
frontend/index.html         Optional test page served at /
Dockerfile, docker-entrypoint.sh   Image for Railway and Docker
compose.yaml                Local Docker deployment
```
