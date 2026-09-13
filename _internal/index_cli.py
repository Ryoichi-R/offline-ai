"""CLI for explicit Embedding index preparation."""

from __future__ import annotations

import argparse

from index_service import IndexCoordinator


def _print_status(status: dict) -> None:
    fields = (
        ("state", "state"),
        ("job_id", "job_id"),
        ("mode", "mode"),
        ("processed", "processed"),
        ("total", "total"),
        ("generated", "generated"),
        ("reused", "reused"),
        ("failed", "failed"),
        ("checkpointed", "checkpointed"),
        ("elapsed_seconds", "elapsed_seconds"),
        ("rate_per_second", "rate_per_second"),
        ("rate_basis", "rate_basis"),
        ("eta_seconds", "eta_seconds"),
        ("error_code", "error_code"),
    )
    for label, key in fields:
        if key in status and status[key] is not None:
            print(f"{label}: {status[key]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="offline-ai Embedding index manager")
    parser.add_argument("command", choices=("build", "status", "cancel", "resume"))
    parser.add_argument("--job-id", default=None)
    parser.add_argument(
        "--full",
        action="store_true",
        help="ignore the completed cache and rebuild every current source chunk",
    )
    args = parser.parse_args(argv)
    coordinator = IndexCoordinator()
    # 前回プロセスが所有者不在のまま残した building/cancelling を実測へ戻す（F-9）。
    coordinator.reconcile_orphaned_state()

    if args.command == "status":
        _print_status(coordinator.status())
        return 0
    if args.command == "cancel":
        _print_status(coordinator.cancel(args.job_id))
        return 0
    if args.command in {"build", "resume"}:
        if args.full and args.command != "build":
            parser.error("--full is only valid with the build command")
        mode = "full" if args.full else None

        def emit(status: dict) -> None:
            processed = int(status.get("processed", 0))
            total = int(status.get("total", 0))
            percent = (processed / total * 100) if total else 100.0
            suffix = f" / {total} ({percent:.1f}%)"
            eta = status.get("eta_seconds")
            eta_text = f", eta~{eta}s" if eta is not None else ""
            print(
                f"index: {processed}{suffix}, generated={status.get('generated', 0)}, "
                f"reused={status.get('reused', 0)}, failed={status.get('failed', 0)}, "
                f"checkpointed={status.get('checkpointed', 0)}{eta_text}",
                flush=True,
            )

        return coordinator.run_blocking(
            resume=args.command == "resume", mode=mode, emit=emit
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
