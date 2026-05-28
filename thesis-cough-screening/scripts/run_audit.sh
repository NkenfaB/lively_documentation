#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m src.data.audit_metadata "$@"
python -m src.data.unify_metadata
python -m src.data.create_splits
