#!/usr/bin/env bash
# Run from anywhere; Python's standard library is the only dependency.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$SCRIPT_DIR/export_paper_metrics.py" "$@"
