"""JSON catalog and JSON-lines progress CLI."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from .catalog import list_models


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="audio_vocal_engine")
    commands = parser.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models")
    models.add_argument("--model-root", required=True)
    run = commands.add_parser("run")
    run.add_argument("request")
    run.add_argument("result")
    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)
    stream = sys.stdout

    def event(value):
        print(json.dumps(value, ensure_ascii=False, allow_nan=False), file=stream, flush=True)

    if args.command == "models":
        event({"models": list_models(args.model_root)})
        return 0
    if args.command == "cleanup":
        from .work import cleanup_work
        event({"job_id": args.job_id, "removed": cleanup_work(args.job_id)})
        return 0
    request = {}
    try:
        from .job import run_job
        request = json.loads(Path(args.request).read_text(encoding="utf-8-sig"))
        # Vendor code prints diagnostics; reserve stdout for the JSON protocol.
        with redirect_stdout(sys.stderr):
            result = run_job(request, args.result, event)
        return 1 if result.get("error") else 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        # A malformed request also gets a machine-readable result when the
        # caller supplied a new destination. Never overwrite an existing file.
        failure = {"schema_version": 1, "job_id": request.get("job_id") if isinstance(request, dict) else None,
                   "model": None, "items": [], "error": error}
        try:
            destination = Path(args.result)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as handle:
                json.dump(failure, handle, ensure_ascii=False)
        except OSError:
            pass
        event({"progress": 0.0, "message": error, "stage": "failed", "indeterminate": False})
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
