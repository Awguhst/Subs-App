"""Run the API: ``python -m app`` (from the backend directory).

HOST and PORT come from the environment, as Railway and other platforms expect.
"""
import logging
import os

import uvicorn


def main():
    level = os.getenv("LOG_LEVEL", "info").lower()
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(
        "app.main:create_app", factory=True,
        host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")),
        # One process: requests are async, and the heavy work runs in FFmpeg
        # subprocesses and remote APIs. Scale with MAX_CONCURRENT_JOBS instead.
        workers=1,
        proxy_headers=True, forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "*"),
        timeout_keep_alive=75,  # longer than typical proxy idle timeouts
        log_level=level,
    )


if __name__ == "__main__":
    main()
