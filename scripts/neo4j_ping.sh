#!/bin/zsh
source ~/.zshrc
cd /Users/bhs/PROJECTS/citation-compass || exit 1

LOGFILE="$HOME/neo4j_ping.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') starting neo4j ping" >> "$LOGFILE"

python scripts/neo4j_ping.py --uri "$NEO4J_URI_dagnyazu" --user "neo4j" --password "$NEO4J_PASSWORD_dagnyazu" --monitor-id "dagnyazu"  >> "$LOGFILE" 2>&1
python scripts/neo4j_ping.py --uri "$NEO4J_URI" --user "neo4j" --password "$NEO4J_PASSWORD" --monitor-id "dagny099"     >> "$LOGFILE" 2>&1
STATUS=$?

echo "$(date '+%Y-%m-%d %H:%M:%S') ping completed (exit $STATUS)" >> "$LOGFILE"
exit $STATUS
