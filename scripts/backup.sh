#!/bin/sh
# Copy the live SQLite database out of the container into BACKUP_DIR (safe while the app runs).
# Writes to backups/ unless BACKUP_DIR is set (e.g. a synced cloud folder). Times are this computer's local time.
set -eu
PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"   # launchd starts with a bare PATH
cd "$(dirname "$0")/.."
dir="${BACKUP_DIR:-backups}"
mkdir -p "$dir"
name="playground-$(date +%Y%m%d-%H%M%S).db"
trap 'rm -f "$dir/.$name.part"' EXIT
# /tmp is a tmpfs, which docker cp cannot read, so the copy comes out through stdout.
docker compose exec -T web python -c "
import os, sqlite3, sys
src = sqlite3.connect('instance/playground.db'); dst = sqlite3.connect('/tmp/backup.db')
src.backup(dst); dst.close(); src.close()
sys.stdout.buffer.write(open('/tmp/backup.db', 'rb').read()); os.remove('/tmp/backup.db')" > "$dir/.$name.part"
[ "$(sqlite3 "$dir/.$name.part" 'pragma integrity_check')" = ok ]
mv "$dir/.$name.part" "$dir/$name"
# Keep the newest 30. A launchd job may not list cloud-synced folders (macOS privacy); then this is skipped.
if all=$(ls -1t "$dir"/playground-*.db 2>/dev/null); then
  echo "$all" | tail -n +31 | xargs rm -f
else
  echo "cannot list $dir, old backups not pruned" >&2
fi
echo "$(date '+%F %T') $dir/$name"
