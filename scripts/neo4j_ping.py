#!/usr/bin/env python3
"""
Richer Neo4j ping script for Citation Compass.

This CLI gathers a snapshot of basic database telemetry and optionally records
it back into Neo4j for auditing. By default, it prints the collected metrics as
pretty JSON and writes a `PingLog` node unless `--dry-run` is supplied.

Usage example:

    python scripts/neo4j_ping.py \\
        --uri "$NEO4J_URI" \\
        --user "$NEO4J_USER" \\
        --password "$NEO4J_PWD" \\
        --monitor-id primary
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure src/ is importable when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from database.connection import Neo4jConnection, Neo4jError  # type: ignore  # noqa: E402
from database.ping import (  # type: ignore  # noqa: E402
    collect_ping_metrics,
    record_ping_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("neo4j_ping")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Collect and store Neo4j ping metrics.")
    parser.add_argument("--uri", help="Neo4j bolt URI (bolt://host:7687)", default=os.getenv("NEO4J_URI"))
    parser.add_argument("--user", help="Neo4j username", default=os.getenv("NEO4J_USER"))
    parser.add_argument("--password", help="Neo4j password", default=os.getenv("NEO4J_PWD") or os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--monitor-id", default="default", help="Identifier for the PingLog node.")
    parser.add_argument("--machine-id", help="Optional machine identifier override.")
    parser.add_argument("--dry-run", action="store_true", help="Collect metrics without writing to Neo4j.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip initial connection test.")
    parser.add_argument(
        "--output",
        help="Optional JSON file path to append ping metrics (newline-delimited JSON).",
    )
    return parser.parse_args(argv)


def ensure_env(uri: Optional[str], user: Optional[str], password: Optional[str]) -> None:
    """Populate environment variables expected by Neo4jConnection."""
    if uri:
        os.environ["NEO4J_URI"] = uri
    if user:
        os.environ["NEO4J_USER"] = user
    if password:
        os.environ["NEO4J_PWD"] = password


def write_output(metrics_dict: dict, output_path: Optional[str]) -> None:
    """Append the metrics to a JSONL file if requested."""
    if not output_path:
        return

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics_dict) + "\n")
    logger.info("Appended ping metrics to %s", path)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    ensure_env(args.uri, args.user, args.password)

    if not all([os.getenv("NEO4J_URI"), os.getenv("NEO4J_USER"), os.getenv("NEO4J_PWD") or os.getenv("NEO4J_PASSWORD")]):
        logger.error("Neo4j credentials are incomplete. Provide them via args or environment variables.")
        return 1

    try:
        connection = Neo4jConnection(validate_connection=not args.skip_validation)
    except Neo4jError as exc:
        logger.error("Unable to establish Neo4j connection: %s", exc)
        return 2

    try:
        metrics = collect_ping_metrics(connection, machine_id=args.machine_id)
        metrics_dict = metrics.as_dict()
        print(json.dumps(metrics_dict, indent=2))

        if args.dry_run:
            logger.info("Dry run requested; skipping persistence.")
        else:
            summary = record_ping_metrics(connection, metrics, monitor_id=args.monitor_id)
            logger.info(
                "Updated PingLog '%s'; retained %d entries.",
                summary["monitor_id"],
                summary["log_size"],
            )

        write_output(metrics_dict, args.output)

    except Neo4jError as exc:
        logger.error("Ping failed: %s", exc)
        return 3
    except Exception as exc:  # Defensive catch for unexpected errors
        logger.exception("Unexpected error during ping: %s", exc)
        return 4
    finally:
        connection.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
