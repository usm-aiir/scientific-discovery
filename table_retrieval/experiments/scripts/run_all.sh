#!/usr/bin/env bash
# Run both stages in order; stop immediately if either stage fails.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$script_dir/run_stage1.sh"
bash "$script_dir/run_stage2.sh"
