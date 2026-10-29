#!/usr/bin/env bash
# Create one isolated environment for both retrieval stages.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/.." && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
env_dir="$experiment_dir/.venv"

python_cmd="${PYTHON:-python3.10}"
if ! command -v "$python_cmd" >/dev/null 2>&1; then
    if [[ -n "${PYTHON:-}" ]]; then
        echo "Python executable not found: $PYTHON" >&2
        exit 1
    fi
    python_cmd=python3
fi
"$python_cmd" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit('Python 3.10 or newer is required.')
PY

if [[ -e "$env_dir" && ! -f "$env_dir/pyvenv.cfg" ]]; then
    echo "Not a virtual environment: $env_dir. Move it before running setup." >&2
    exit 1
fi
if [[ ! -x "$env_dir/bin/python" ]]; then
    "$python_cmd" -m venv "$env_dir"
fi

"$env_dir/bin/python" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit('The virtual environment needs Python 3.10 or newer. Move it and rerun setup.')
PY

"$env_dir/bin/python" -m pip install -r "$repo_root/requirements.txt"
"$env_dir/bin/python" -m pip install --no-deps --no-build-isolation -e "$repo_root"
"$env_dir/bin/python" -m pip check

echo "Setup complete. Run bash $script_dir/run_all.sh"
