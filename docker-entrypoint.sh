#!/bin/sh
# Railway and Docker mount volumes owned by root. Make the data directory
# writable for the unprivileged "app" user, then drop root before starting, so
# FFmpeg never processes uploads with root privileges.
set -eu

DATA_DIR="${DATA_DIR:-${RAILWAY_VOLUME_MOUNT_PATH:-/data}}"
# The app resolves a relative DATA_DIR against /app; use the same absolute path here.
case "$DATA_DIR" in
    /*) ;;
    *) DATA_DIR="/app/${DATA_DIR#./}" ;;
esac
export DATA_DIR

if [ "$(id -u)" = "0" ]; then
    mkdir -p "$DATA_DIR"
    if [ "$(stat -c %u "$DATA_DIR")" != "$(id -u app)" ]; then
        chown -R app:app "$DATA_DIR"
    fi
    exec setpriv --reuid=app --regid=app --init-groups -- "$@"
fi
exec "$@"
