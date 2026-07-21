# Keeping Your Neo4j Database Alive: Automated Health Monitoring

!!! tip "The Problem This Solves"
    Neo4j AuraDB free tier instances pause after **30 days of inactivity**. If you're running a long-term research project with intermittent database access, you need a way to keep your instance alive without manual daily logins.

This guide shows you how to set up automated daily health checks that:

- ✅ Keep your free Neo4j instance from pausing
- ✅ Collect valuable database telemetry (node counts, response times, build info)
- ✅ Create an audit trail stored in Neo4j itself
- ✅ Work even when your computer is asleep (yes, really!)

---

## What We're Building

A Python script that pings your Neo4j database daily, collecting metrics and storing them as a rolling history. The twist? **We'll make your Mac wake up** just before the scheduled ping so cron jobs actually run.

**What you'll get:**
- Daily confirmation your database is responding
- Historical metrics (paper counts, citation counts, response times)
- A `PingLog` node in Neo4j with the last 365 pings
- Peace of mind for long-running research projects

---

## Prerequisites

- Python environment with Citation Compass installed (`pip install -e ".[all]"`)
- Neo4j AuraDB credentials (URI, username, password)
- macOS with admin access (for the wake-up trick)
- Access to the [Citation Compass repository](https://github.com/dagny099/citation-compass)

---

## The Ping Script

Citation Compass includes a rich health monitoring script at [`scripts/neo4j_ping.py`](https://github.com/dagny099/citation-compass/blob/main/scripts/neo4j_ping.py).

### What It Does

When you run the ping script, it:

1. **Connects to Neo4j** using your credentials
2. **Collects metrics**:
    - Total nodes and relationships
    - Database build edition and version
    - Machine ID (useful if monitoring from multiple locations)
    - Response time (ISO timestamp and human-readable format)
3. **Prints results** as pretty JSON to stdout
4. **Stores history** in a `PingLog` node (keeps last 365 entries)
5. **Optional**: Appends to a local JSONL file for offline analysis

### Try It Manually

```bash
# Basic usage (assuming credentials in environment)
python scripts/neo4j_ping.py \
  --uri "$NEO4J_URI" \
  --user "$NEO4J_USER" \
  --password "$NEO4J_PASSWORD" \
  --monitor-id my-research-db
```

**Example output:**
```json
{
  "timestamp_iso": "2025-10-24T11:30:00Z",
  "timestamp_display": "2025-10-24 06:30:00 AM CDT",
  "machine_id": "MacBook-Pro.local",
  "number_nodes": 12543,
  "number_rels": 68721,
  "build_edition": "enterprise-aura",
  "response_time_ms": 142
}
```

!!! note "What Gets Stored"
    A single `PingLog` node is created/updated with your `monitor-id`. Latest values are stored as `lastPing-*` properties, and full history is kept in a JSON array (last 365 entries).

### Dry Run Mode

Test without writing to Neo4j:

```bash
python scripts/neo4j_ping.py \
  --uri "$NEO4J_URI" \
  --user "$NEO4J_USER" \
  --password "$NEO4J_PASSWORD" \
  --dry-run
```

---

## Automating with Cron (macOS)

### Step 1: Create a Wrapper Script

Cron doesn't source your shell profile, so we need a wrapper that loads environment variables.

Create `~/bin/neo4j_ping_wrapper.sh`:

```bash
#!/bin/zsh
# Load your shell environment (where Neo4j credentials are exported)
source ~/.zshrc

# Navigate to your project
cd /Users/YOUR_USERNAME/PROJECTS/citation-compass || exit 1

# Log to file
LOGFILE="$HOME/neo4j_ping.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') Starting Neo4j ping" >> "$LOGFILE"

# Run the ping script
python scripts/neo4j_ping.py \
  --uri "$NEO4J_URI" \
  --user "$NEO4J_USER" \
  --password "$NEO4J_PASSWORD" \
  --monitor-id my-research-db \
  >> "$LOGFILE" 2>&1

STATUS=$?
echo "$(date '+%Y-%m-%d %H:%M:%S') Ping completed (exit $STATUS)" >> "$LOGFILE"
exit $STATUS
```

**Make it executable:**
```bash
chmod +x ~/bin/neo4j_ping_wrapper.sh
```

### Step 2: Add to Crontab

Schedule daily pings at 11:55 PM:

```bash
# Edit crontab
crontab -e

# Add this line (runs every day at 23:55)
55 23 * * * /Users/YOUR_USERNAME/bin/neo4j_ping_wrapper.sh
```

**But wait—there's a catch!** 🚨

---

## The KEY Insight: Waking Your Mac

**Problem**: macOS sleeps at night. Cron jobs don't run when your computer is asleep.

**Solution**: Tell macOS to wake up just before your scheduled cron job!

### The Magic Command

```bash
sudo pmset repeat wakeorpoweron MTWRFSU 23:55:00
```

This tells your Mac: **"Wake up or power on every day (Monday-Sunday) at 11:55 PM"**—right when the cron job runs!

**Verify it's set:**
```bash
pmset -g sched
```

You should see:
```
Repeating power events:
  wakeorpoweron MTWRFSU at 23:55:00
```

!!! warning "Requires Admin Password"
    The `pmset` command requires `sudo` (admin privileges). Your Mac will briefly wake, run the cron job, then go back to sleep.

---

## What Happens Daily

1. **11:55 PM**: Mac wakes up (thanks to `pmset`)
2. **11:55 PM**: Cron triggers your wrapper script
3. Script sources environment variables from `.zshrc`
4. Script runs `neo4j_ping.py`
5. Ping collects metrics and updates the `PingLog` node in Neo4j
6. Results appended to `~/neo4j_ping.log`
7. Mac goes back to sleep

**Your Neo4j instance stays active!** ✨

---

## Querying Your Ping History

Once you've collected some pings, you can analyze them in Neo4j Browser:

```cypher
// Get latest ping status
MATCH (p:PingLog {monitor_id: 'my-research-db'})
RETURN
  p.`lastPing-timestamp` AS last_checked,
  p.`lastPing-number_nodes` AS nodes,
  p.`lastPing-number_rels` AS relationships,
  p.`lastPing-response_time_ms` AS response_ms;
```

```cypher
// Get full history (last 10 pings)
MATCH (p:PingLog {monitor_id: 'my-research-db'})
UNWIND p.log[-10..] AS entry
RETURN entry;
```

```cypher
// Plot node growth over time (export to CSV for visualization)
MATCH (p:PingLog {monitor_id: 'my-research-db'})
UNWIND p.log AS entry
WITH entry
ORDER BY entry.timestamp_iso
RETURN
  entry.timestamp_display AS date,
  entry.number_nodes AS nodes,
  entry.number_rels AS relationships;
```

---

## Advanced Options

### Multiple Databases

If you're monitoring multiple Neo4j instances (dev, prod, etc.), use different `monitor-id` values:

```bash
# Development database
python scripts/neo4j_ping.py \
  --uri "$NEO4J_URI_DEV" \
  --monitor-id dev-db

# Production database
python scripts/neo4j_ping.py \
  --uri "$NEO4J_URI_PROD" \
  --monitor-id prod-db
```

Each gets its own `PingLog` node with separate history.

### Export to Local File

Keep an offline backup of ping data:

```bash
python scripts/neo4j_ping.py \
  --uri "$NEO4J_URI" \
  --user "$NEO4J_USER" \
  --password "$NEO4J_PASSWORD" \
  --output ~/neo4j_pings.jsonl
```

This appends each ping as a JSON line (JSONL format)—perfect for importing into pandas or other analysis tools.

### Alerting on Failures

The script exits with non-zero status on errors. Wrap it in a notification system:

```bash
# In your wrapper script, add:
if [ $STATUS -ne 0 ]; then
  # Send alert (e.g., email, Slack, etc.)
  echo "Neo4j ping failed!" | mail -s "DB Alert" you@example.com
fi
```

---

## Troubleshooting

??? question "Cron job not running"
    **Check the log:**
    ```bash
    tail -20 ~/neo4j_ping.log
    ```

    **Verify crontab:**
    ```bash
    crontab -l
    ```

    **Check if Mac woke up:**
    ```bash
    pmset -g log | grep Wake
    ```

??? question "Connection errors in log"
    **Test manually:**
    ```bash
    ~/bin/neo4j_ping_wrapper.sh
    ```

    **Check credentials:**
    ```bash
    echo $NEO4J_URI
    echo $NEO4J_USER
    echo $NEO4J_PASSWORD
    ```

    Make sure these are exported in `.zshrc`, not just set.

??? question "Mac isn't waking up"
    **Check power schedule:**
    ```bash
    pmset -g sched
    ```

    **If nothing shows**, re-run:
    ```bash
    sudo pmset repeat wakeorpoweron MTWRFSU 23:55:00
    ```

    **Note**: Laptops must be plugged in or have sufficient battery.

??? question "JSON serialization errors"
    The script handles NumPy/pandas types automatically. If you hit errors, ensure:
    ```bash
    pip install numpy pandas
    ```

---

## Why This Matters

Beyond just keeping your free Neo4j instance alive, this workflow gives you:

1. **Confidence**: Your database is healthy and responding
2. **Insight**: Track your citation network growth over time
3. **History**: Audit trail for debugging ("When did my data import run?")
4. **Foundation**: Infrastructure for future monitoring (alerting, dashboards, etc.)

For research projects spanning months or years, knowing your infrastructure is solid lets you focus on the actual research.

---

## Next in This Series

This is the first in a series of practical Neo4j workflows for academic research. Coming soon:

- **Visualizing Ping Metrics**: Build a dashboard showing database health over time
- **Automated Backups**: Schedule exports of your citation network
- **Query Performance Monitoring**: Track slow queries and optimize your schema

---

**Questions or improvements?** [Open an issue on GitHub](https://github.com/dagny099/citation-compass/issues) or check the [main repository](https://github.com/dagny099/citation-compass) for the latest scripts.

Happy monitoring! 🩺✨
