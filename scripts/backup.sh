#!/bin/sh
# Copy the live SQLite database out of the container into backups/ (safe while the app runs).
# Run from the project folder, e.g. daily from cron or launchd. Keeps the newest 30.
set -eu
cd "$(dirname "$0")/.."
mkdir -p backups
name="playground-$(date +%Y%m%d-%H%M%S).db"
docker compose exec -T web python -c "
import sqlite3
src = sqlite3.connect('instance/playground.db'); dst = sqlite3.connect('/tmp/backup.db')
src.backup(dst); dst.close(); src.close()"
docker compose cp web:/tmp/backup.db "backups/$name"
docker compose exec -T web rm -f /tmp/backup.db
ls -1t backups/playground-*.db | tail -n +31 | xargs rm -f
echo "backups/$name"
