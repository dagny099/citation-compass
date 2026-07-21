"""
Utilities for recording Neo4j ping metrics.

This module builds on the existing Neo4jConnection wrapper to collect and
persist basic health-check telemetry on demand. The collected metrics include:
- UTC timestamp and machine identifier
- Counts of total nodes/relationships
- Counts of unique node labels and relationship types

Metrics are stored on a single `PingLog` node (per monitor id) whose properties
track the latest ping and maintain a rolling JSON-encoded history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import platform
import uuid
from typing import Any, Dict, Optional

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:  # pragma: no cover - Python <3.9 fallback
    ZoneInfo = None  # type: ignore

from .connection import Neo4jConnection, Neo4jError


def _to_json_safe(value: Any) -> Any:
    """Recursively coerce values into JSON-serialisable Python primitives."""
    if isinstance(value, dict):
        return {key: _to_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]

    # Try numpy scalar conversion without a hard dependency.
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
    except ModuleNotFoundError:
        pass

    # Handle pandas timestamp-like objects if pandas is available.
    try:
        import pandas as pd  # type: ignore

        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return value.isoformat()
    except ModuleNotFoundError:
        pass

    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except Exception:
            return str(value)

    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value

    return value


@dataclass(frozen=True)
class PingMetrics:
    """Structured payload describing a single ping event."""

    timestamp_iso: str
    timestamp_display: str
    machine_id: str
    number_node_types: int
    number_nodes: int
    number_relationship_types: int
    number_relationships: int
    neo4j_version: Optional[str]
    neo4j_edition: Optional[str]
    raw: Dict[str, Any]
    ping_id: str

    def as_dict(self) -> Dict[str, Any]:
        """Convert to a serialisable dictionary."""
        return _to_json_safe({
            "ping_id": self.ping_id,
            "timestamp_iso": self.timestamp_iso,
            "timestamp_display": self.timestamp_display,
            "machine_id": self.machine_id,
            "number_node_types": self.number_node_types,
            "number_nodes": self.number_nodes,
            "number_relationship_types": self.number_relationship_types,
            "number_relationships": self.number_relationships,
            "neo4j_version": self.neo4j_version,
            "neo4j_edition": self.neo4j_edition,
            "raw": self.raw,
        })


def collect_ping_metrics(
    connection: Neo4jConnection,
    machine_id: Optional[str] = None,
) -> PingMetrics:
    """
    Collect Neo4j database telemetry for a ping operation.

    Args:
        connection: Active Neo4j connection.
        machine_id: Optional machine identifier override.

    Returns:
        PingMetrics dataclass with populated telemetry details.

    Raises:
        Neo4jError: If the database info cannot be retrieved.
    """
    db_info = connection.get_database_info()
    sanitized_info = _to_json_safe(db_info)

    timestamp_utc = datetime.now(timezone.utc)
    if ZoneInfo is not None:
        try:
            ct_zone = ZoneInfo("America/Chicago")
        except Exception:  # pragma: no cover - defensive fallback
            ct_zone = timezone(-timedelta(hours=6))
    else:  # pragma: no cover - Python <3.9 fallback
        ct_zone = timezone(-timedelta(hours=6))

    timestamp_ct = timestamp_utc.astimezone(ct_zone)
    timestamp_display = timestamp_ct.strftime("%Y-%m-%d %I:%M:%S %p %Z")

    machine_identifier = machine_id or platform.node()

    node_labels = sanitized_info.get("node_labels") or []
    relationship_types = sanitized_info.get("relationship_types") or []

    return PingMetrics(
        timestamp_iso=timestamp_utc.isoformat(),
        timestamp_display=timestamp_display,
        machine_id=machine_identifier,
        number_node_types=len(node_labels),
        number_nodes=int(sanitized_info.get("total_nodes", 0) or 0),
        number_relationship_types=len(relationship_types),
        number_relationships=int(sanitized_info.get("total_relationships", 0) or 0),
        neo4j_version=sanitized_info.get("version"),
        neo4j_edition=sanitized_info.get("edition"),
        raw=sanitized_info,
        ping_id=str(uuid.uuid4()),
    )


def record_ping_metrics(
    connection: Neo4jConnection,
    metrics: PingMetrics,
    monitor_id: str = "default",
) -> Dict[str, Any]:
    """
    Update a single PingLog node with the latest metrics and append to its log.

    Args:
        connection: Active Neo4j connection.
        metrics: PingMetrics instance to store.
        monitor_id: Identifier for the monitor anchor node.

    Returns:
        Dictionary with monitor identifier and updated log size.

    Raises:
        Neo4jError: If the write operation fails.
    """
    log_entry_payload = {
        "ping_id": metrics.ping_id,
        "timestamp_iso": metrics.timestamp_iso,
        "timestamp_display": metrics.timestamp_display,
        "machine_id": metrics.machine_id,
        "number_node_types": metrics.number_node_types,
        "number_nodes": metrics.number_nodes,
        "number_relationship_types": metrics.number_relationship_types,
        "number_relationships": metrics.number_relationships,
    }
    log_entry = json.dumps(_to_json_safe(log_entry_payload), separators=(",", ":"), sort_keys=True)

    cypher = """
    MERGE (monitor:PingLog {monitor_id: $monitor_id})
    ON CREATE SET monitor.created_at = datetime($timestamp_iso)
    SET monitor.`lastPing-machine-id` = $machine_id,
        monitor.`lastPing-timestamp` = $timestamp_display,
        monitor.`lastPing-number_nodes` = $number_nodes,
        monitor.`lastPing-number_rels` = $number_relationships,
        monitor.neo4j_version = $neo4j_version,
        monitor.neo4j_edition = $neo4j_edition,
        monitor.log = (coalesce(monitor.log, []) + $log_entry)[-365..]
    RETURN monitor.monitor_id AS monitor_id, size(monitor.log) AS log_size
    """

    params = {
        "monitor_id": monitor_id,
        "timestamp_iso": metrics.timestamp_iso,
        "timestamp_display": metrics.timestamp_display,
        "machine_id": metrics.machine_id,
        "number_nodes": metrics.number_nodes,
        "number_relationships": metrics.number_relationships,
        "neo4j_version": metrics.neo4j_version,
        "neo4j_edition": metrics.neo4j_edition,
        "log_entry": [log_entry],
    }

    try:
        result = connection.query(cypher, params)
    except Neo4jError:
        raise
    except Exception as exc:
        raise Neo4jError(f"Failed to record ping metrics: {exc}") from exc

    if result.empty:
        raise Neo4jError("Ping metrics write succeeded but returned no summary.")

    return {
        "monitor_id": result.iloc[0]["monitor_id"],
        "log_size": int(result.iloc[0]["log_size"]) if result.iloc[0]["log_size"] is not None else 0,
        "entry": log_entry_payload,
    }


__all__ = ["PingMetrics", "collect_ping_metrics", "record_ping_metrics"]
