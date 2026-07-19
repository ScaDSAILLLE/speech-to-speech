#!/usr/bin/env bash
# Convenience wrapper around `./scripts/rpi_start.sh --status`.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_ROOT/scripts/rpi_start.sh" --status
