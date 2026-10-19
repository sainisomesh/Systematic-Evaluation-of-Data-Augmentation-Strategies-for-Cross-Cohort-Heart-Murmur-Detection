#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for candidate in python3.12 python3.13 python3.14 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c 'import sys; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, 14))' 2>/dev/null; then
        exec "$candidate" reproduce.py "$@"
    fi
done
echo "Python 3.12 to 3.14 is required." >&2
exit 1
