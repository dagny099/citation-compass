---
layout: post
title: "Keeping Neo4j Alive (and Awake): A Daily Health Check Workflow"
date: 2025-10-24
categories: [neo4j, automation, research-tools]
tags: [neo4j, python, cron, macos, database-monitoring]
excerpt: "How I automated Neo4j database health checks to keep my free AuraDB instance alive—including the macOS trick nobody tells you about."
---

I run two Neo4j AuraDB instances for my research projects. Both are on the free tier, which is generous—but comes with a catch: **after 30 days of inactivity, they pause**. For long-running citation network research where I might not touch the database for weeks, that's a problem.

The solution? An automated daily "ping" that keeps the database alive while collecting useful health metrics. Here's the workflow I built, including **the KEY detail about macOS sleep** that took me way too long to figure out.

<!--more-->

---

## Why This Matters

Beyond just keeping your instance alive, this workflow gives you:

- ✅ **Confidence**: Daily confirmation your database is responding
- ✅ **Telemetry**: Track node/relationship counts, response times, database version
- ✅ **Audit trail**: Historical metrics stored in Neo4j itself (last 365 days)
- ✅ **Early warning**: Catch issues before they become problems

If you're using [Neo4j AuraDB's free tier](https://neo4j.com/cloud/aura-free/) for academic research, side projects, or prototypes, this is essential infrastructure.

---

## The Solution: Automated Health Pings

I built a Python script that collects rich database telemetry and stores it back into Neo4j as a rolling history. The script is part of my [Citation Compass](https://github.com/dagny099/citation-compass) project, but the approach works for any Neo4j use case.

### What It Collects

Every ping captures:

```json
{
  "timestamp_iso": "2025-10-24T06:30:00Z",
  "timestamp_display": "2025-10-24 01:30:00 AM CDT",
  "machine_id": "MacBook-Pro.local",
  "number_nodes": 12543,
  "number_rels": 68721,
  "build_edition": "enterprise-aura",
  "response_time_ms": 142
}
```

This gets stored as:
1. **Console output** (pretty JSON) for monitoring logs
2. **A `PingLog` node in Neo4j** with latest values + rolling history
3. **Optional JSONL file** for offline analysis

### Data Model

The clever part: instead of creating 365 tiny nodes, the script maintains **one `PingLog` node per database** with:

- `lastPing-*` properties for quick access to latest values
- `log` property: a JSON array of the last 365 pings

This keeps your graph clean while preserving history. Here's what it looks like in Cypher:

```cypher
MATCH (p:PingLog {monitor_id: 'my-db'})
RETURN
  p.`lastPing-timestamp`,
  p.`lastPing-number_nodes`,
  p.`lastPing-response_time_ms`;
```

---

## Setting It Up

### 1. The Python Script

The full script is at [`scripts/neo4j_ping.py`](https://github.com/dagny099/citation-compass/blob/main/scripts/neo4j_ping.py) in my repo. Try it manually first:

```bash
python scripts/neo4j_ping.py \
  --uri "neo4j+s://your-instance.databases.neo4j.io" \
  --user "neo4j" \
  --password "your-password" \
  --monitor-id my-research-db
```

You'll see pretty JSON output and a confirmation that the `PingLog` node was updated.

### 2. Automate with Cron

Create a wrapper script (`~/bin/neo4j_ping_wrapper.sh`) that sources your environment:

```bash
#!/bin/zsh
source ~/.zshrc  # Load Neo4j credentials
cd /path/to/citation-compass || exit 1

LOGFILE="$HOME/neo4j_ping.log"
echo "$(date) Starting ping" >> "$LOGFILE"

python scripts/neo4j_ping.py \
  --uri "$NEO4J_URI" \
  --user "$NEO4J_USER" \
  --password "$NEO4J_PASSWORD" \
  --monitor-id my-db \
  >> "$LOGFILE" 2>&1

echo "$(date) Completed" >> "$LOGFILE"
```

Make it executable (`chmod +x ~/bin/neo4j_ping_wrapper.sh`), then add to crontab:

```bash
# Run daily at 11:55 PM
55 23 * * * /Users/you/bin/neo4j_ping_wrapper.sh
```

**But there's a catch...**

---

## The KEY Insight: Waking Your Mac

Here's the thing **nobody tells you about cron on macOS**: 🚨

**Cron jobs don't run when your computer is asleep.**

I scheduled my ping for midnight, assuming it would "just work." It didn't. For weeks, I couldn't figure out why my logs were sporadic—until I realized my MacBook was sleeping at midnight.

### The Fix: pmset

macOS has a built-in tool to schedule wake events. One command solves everything:

```bash
sudo pmset repeat wakeorpoweron MTWRFSU 23:55:00
```

This tells your Mac: **"Wake up (or power on) every day at 11:55 PM"**—right before the cron job runs.

Verify it worked:

```bash
pmset -g sched
```

You should see:
```
Repeating power events:
  wakeorpoweron MTWRFSU at 23:55:00
```

Now your Mac wakes, runs the ping, writes the log, and goes back to sleep. **Your Neo4j instance stays active.** ✨

---

## What You Get

After a few weeks, you can analyze your database health over time:

```cypher
// Plot node growth
MATCH (p:PingLog {monitor_id: 'my-db'})
UNWIND p.log AS entry
WITH entry ORDER BY entry.timestamp_iso
RETURN
  entry.timestamp_display AS date,
  entry.number_nodes AS nodes,
  entry.number_rels AS relationships;
```

Export to CSV, plot in Excel/Python, and you've got a growth chart for your citation network.

I've also found it useful for debugging: "When did that bulk import finish?" → Check the ping logs for the date node counts jumped.

---

## Next in This Series

This is the first in a series where I'll share practical Neo4j workflows for academic research. Coming soon:

- **Visualizing database health**: Build a dashboard from ping metrics
- **Query performance monitoring**: Track slow Cypher queries automatically
- **Automated exports**: Backup your citation network on a schedule

The full code is open source in [Citation Compass](https://github.com/dagny099/citation-compass). If you adapt this for your own projects, I'd love to hear about it—[reach out via GitHub](https://github.com/dagny099/citation-compass/issues) or drop me a note!

---

## TL;DR

**The workflow:**
1. Python script pings Neo4j daily, collects metrics
2. Stores history in a `PingLog` node (last 365 pings)
3. Cron runs it at 11:55 PM
4. **Key trick**: `sudo pmset repeat wakeorpoweron MTWRFSU 23:55:00` makes your Mac wake up before the cron job

**Why it matters:**
- Keeps free-tier Neo4j instances from pausing
- Provides database health telemetry
- Creates an audit trail for research projects

**Resources:**
- [Full script on GitHub](https://github.com/dagny099/citation-compass/blob/main/scripts/neo4j_ping.py)
- [Complete guide in Citation Compass docs](https://docs.barbhs.com/citation-compass/resources/neo4j-health-monitoring/)
- [Neo4j AuraDB free tier](https://neo4j.com/cloud/aura-free/)

Happy monitoring! 🩺📊
