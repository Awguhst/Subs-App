param([switch]$Install, [int]$Port = 8765, [string]$HostAddress = '127.0.0.1')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (!(Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
    Write-Host 'Created .env - add ELEVENLABS_API_KEY (and ANTHROPIC_API_KEY for English).'
}
if (!(Get-Command ffmpeg -ErrorAction SilentlyContinue) -or !(Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    throw 'Install FFmpeg and add ffmpeg and ffprobe to PATH.'
}
if (!(Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Creating the Python virtual environment failed.' }
    $Install = $true
}
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if ($Install) {
    & $python -m pip install -r backend/requirements-dev.txt
    if ($LASTEXITCODE -ne 0) { throw 'Installing Python dependencies failed.' }
}
Write-Host "Bulgarian Subtitle API: http://${HostAddress}:$Port  (docs: /docs, Ctrl+C to stop)"
$env:HOST = $HostAddress
$env:PORT = "$Port"
Push-Location -LiteralPath backend
try { & $python -m app } finally { Pop-Location }
