#!/usr/bin/env bash
# Run a WorldServe benchmark on the GCP H100 VM.
# Usage:
#   bash scripts/gcp/run.sh helios_baseline
#   bash scripts/gcp/run.sh helios_baseline --num-frames 81 --num-steps 50
#
# The first argument selects the benchmark module name (without .py extension).
# Remaining arguments are forwarded to the script.

set -euo pipefail

if [[ -z "${PROJECT_ID:-}" ]]; then
  echo "ERROR: source scripts/gcp/config.sh first." >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <benchmark-name> [args...]" >&2
  echo "Example: $0 helios_baseline --num-iters 2" >&2
  exit 1
fi

BENCHMARK="$1"
shift
EXTRA_ARGS="$*"

# Find the benchmark file under benchmarks/baseline/ or benchmarks/optimised/.
REMOTE_CMD="
set -euo pipefail
cd ${REMOTE_PROJECT_DIR}
set -a; source .env; set +a       # export HF_TOKEN

# Resolve script path
for d in benchmarks/baseline benchmarks/optimised; do
  if [[ -f \"\${d}/${BENCHMARK}.py\" ]]; then
    SCRIPT=\"\${d}/${BENCHMARK}.py\"
    break
  fi
done
if [[ -z \"\${SCRIPT:-}\" ]]; then
  echo \"ERROR: ${BENCHMARK}.py not found in benchmarks/{baseline,optimised}/\" >&2
  exit 1
fi

echo \"Running \${SCRIPT} on \$(hostname) ...\"
echo \"-----------------------------------------\"
PYTHONPATH=. python \${SCRIPT} ${EXTRA_ARGS}
"

gcloud compute ssh "${VM_NAME}" \
  --zone="${ZONE}" --project="${PROJECT_ID}" \
  --command="${REMOTE_CMD}"
