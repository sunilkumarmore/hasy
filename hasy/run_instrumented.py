"""Run the upstream server with HASY's layers installed.

Equivalent to `uv run run_server.py`, plus per-turn latency timing (Phase 1)
and the entity-resolved memory layer (Phase 3). Upstream's run_server.py is
imported, not modified.

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
    from hasy.latency import install as install_latency

    install_latency()

    # Memory patches AgentFactory, which is only consulted when a client
    # connects — but install before the server starts so the very first
    # conversation gets it too. Independent of the latency patches: they hook
    # the conversation/LLM/TTS path, this hooks agent construction.
    try:
        from hasy.memory.install import install as install_memory

        install_memory()
    except Exception as e:
        logger.warning(f"HASY memory unavailable ({e}); continuing without it.")

    try:
        run_server.run(console_log_level=console_log_level)
    except KeyboardInterrupt:
        logger.info("Interrupted — printing latency summary.")
        # atexit handler in hasy.latency prints the table and writes the CSV.
        sys.exit(0)


if __name__ == "__main__":
    main()
