#!/usr/bin/env bash
set -euo pipefail

exec python3 "$(dirname "${BASH_SOURCE[0]}")/generate_recent_trailer_feed.py" "$@"
