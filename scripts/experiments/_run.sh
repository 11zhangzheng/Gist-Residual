#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 EXPERIMENT_ID [runner arguments...]" >&2
  exit 2
fi

experiment_id="$1"
shift
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m fidmem.experiments.pack_cli \
  --project-root "${project_root}" \
  --experiment "${experiment_id}" \
  "$@"
