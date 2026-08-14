"""Run the upstream server with HASY latency instrumentation installed.

Equivalent to `uv run run_server.py`, plus per-turn latency timing. Upstream's
run_server.py is imported, not modified.

    uv run python -m hasy.run_instrumented [--verbose] [--hf_mirror]

On exit (Ctrl+C), prints the latency table and writes latency_logs/*.csv.
"""

from __future__ import annotations

import os
import sys

from loguru import logger


def main() -> None:
    # Import upstream entrypoint. Its __main__ guard means importing is safe.
    import run_server

    args = run_server.parse_args()
    console_log_level = "DEBUG" if args.verbose else "INFO"
    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # init_logger() is called inside run(); install our patches first so they
    # are in place before any conversation can start.
    from hasy.latency import install

    install()

    try:
        run_server.run(console_log_level=console_log_level)
    except KeyboardInterrupt:
        logger.info("Interrupted — printing latency summary.")
        # atexit handler in hasy.latency prints the table and writes the CSV.
        sys.exit(0)


if __name__ == "__main__":
    main()
