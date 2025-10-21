# Neo4j Daily Ping Playbook

![Citation Compass](assets/images/logo.svg)

Welcome! This guide walks you through reproducing the **Neo4j health ping** we set up for Citation Compass. It is written for anyone maintaining a similar stack—Python services talking to Neo4j, with connection credentials stored in shell profiles—and aims to be both practical and friendly.

---

## Why This Ping Matters

- **Confidence in availability**: Know your database answered with real stats, not just a TCP handshake.
- **Audit trail**: Keep a rolling log of metrics without flooding your graph with tiny nodes.
- **Fast insight**: Grab machine id, counts, and build edition at a glance for debugging.

---

## Architecture at a Glance

| Component | Responsibility |
|-----------|----------------|
| `scripts/neo4j_ping.py` | CLI wrapper that accepts credentials, prints JSON, and writes results |
| `src/database/ping.py` | Collects rich stats, sanitises them for JSON, updates a single `PingLog` node |
| `src/database/connection.py` | Existing connection helper that provides `get_database_info()` |
| Neo4j | Stores the `PingLog` node with `lastPing-*` properties and a rolling JSON array |

The CLI script only prints and writes—no scheduling is built in. That means you can reuse it anywhere: cron, GitHub Actions, or a manual check.

---

## Prerequisites

1. **Python environment** with the project dependencies (`pip install -e ".[all]"`).
2. **Neo4j credentials** available through environment variables (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PWD` or `NEO4J_PASSWORD`).
3. **Access to the repository** so you can run `scripts/neo4j_ping.py`.

Tip: if you store credentials in `.zshrc`, export them there and then source the file before running the script.

```bash
export NEO4J_URI="neo4j+s://your-host.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PWD="super-secret"
source ~/.zshrc  # ensures these values are available
```

---

## Running the Ping

```bash
python scripts/neo4j_ping.py \
  --uri "$NEO4J_URI" \
  --user "$NEO4J_USER" \
  --password "$NEO4J_PWD" \
  --monitor-id primary
```

What you get:

- **Pretty-printed JSON** describing the ping (includes both UTC ISO and human-friendly Central Time timestamps).
- **Log update in Neo4j**: the script merges a single node labeled `PingLog` with the supplied `monitor_id`, updates its latest properties, and appends the JSON string to a bounded `log` array (max 365 entries).
- **Summary line**: `Updated PingLog 'primary'; retained 42 entries.`

### Dry Runs & Local Files

- Use `--dry-run` to skip writing to Neo4j while still printing JSON.
- Use `--output /path/to/pings.jsonl` to append each ping payload to a local JSONL file.

---

## Data Model in Neo4j

Only one node per monitor id:

```cypher
MERGE (monitor:PingLog {monitor_id: $monitor_id})
SET monitor.`lastPing-machine-id` = "...",
    monitor.`lastPing-timestamp` = "2025-10-21 11:18:38 AM CDT",
    monitor.`lastPing-number_nodes` = 12345,
    monitor.`lastPing-number_rels` = 67890,
    monitor.log = (coalesce(monitor.log, []) + $log_entry)[-365..]
```

- `lastPing-*` properties are human readable; the timestamp is formatted in Central Time.
- `log` is a list of JSON strings. Each entry includes both ISO and display timestamps, machine id, and counts. Trimming with `[-365..]` keeps only the most recent 365 pings.
- Because these properties live on one node, you avoid ballooning the graph with hundreds of historical nodes.

You can query the latest values with:

```cypher
MATCH (p:PingLog {monitor_id: 'primary'})
RETURN
  p.`lastPing-timestamp` AS last_checked,
  p.`lastPing-machine-id` AS machine,
  p.`lastPing-number_nodes` AS nodes,
  p.`lastPing-number_rels` AS relationships,
  p.log[-1] AS latest_entry;
```

---

## Scheduling Example (macOS Cron)

1. Create a wrapper script that sources your profile and calls the ping:

```bash
#!/bin/zsh
source ~/.zshrc
cd /Users/you/PROJECTS/citation-compass
/usr/bin/env python scripts/neo4j_ping.py --monitor-id primary >> ~/neo4j_ping.log 2>&1
```

2. Make it executable: `chmod +x ~/bin/neo4j_ping.sh`
3. Add a cron entry (runs daily at 7:00 AM Central):

```
0 7 * * * /Users/you/bin/neo4j_ping.sh
```

Replace with launchd, systemd, or your CI scheduler as needed.

---

## Troubleshooting Tips

- **JSON serialization error**: The helper now coerces NumPy and pandas scalars to plain Python types. If you ever hit a serialization error again, ensure `numpy` or `pandas` are installed and up to date.
- **Credential mismatch**: Run with `--dry-run` and confirm the script prints metrics without attempting writes; this is an easy way to debug connectivity before touching Neo4j.
- **Central Time output looks wrong**: The script uses the `America/Chicago` time zone via `zoneinfo`. On Python <3.9, it falls back to UTC-6 with no daylight adjustment; consider upgrading Python for accurate DST handling.

---

## Extending the Ping

- **Multiple environments**: Call the script with different `--monitor-id` values (e.g., `prod`, `staging`) so each environment maintains its own rolling history.
- **Alerting**: Hook the JSON output into your monitoring system. For example, send a Slack message if the script exits with a non-zero code or if node counts fall unexpectedly.
- **Visualization**: In Neo4j Bloom or a dashboard tool, plot `lastPing-number_nodes` over time by parsing the `log` field.

---

You now have everything needed to clone this setup. Feel free to adapt the node property names, adjust the retention window, or wrap the CLI in whatever scheduling system you prefer. Happy monitoring!
