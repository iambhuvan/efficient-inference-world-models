#!/usr/bin/env bash
# Push project source to the VM via gcloud compute scp.
# Excludes weights, run outputs, caches — those live on the VM only.

set -euo pipefail

if [[ -z "${PROJECT_ID:-}" ]]; then
  echo "ERROR: source scripts/gcp/config.sh first." >&2
  exit 1
fi

# Ensure the destination dir exists on the VM.
gcloud compute ssh "${VM_NAME}" \
  --zone="${ZONE}" --project="${PROJECT_ID}" \
  --command="mkdir -p ${REMOTE_PROJECT_DIR}"

# rsync via gcloud — tunneled through IAP.
echo "Syncing ${LOCAL_PROJECT_DIR} → ${VM_NAME}:${REMOTE_PROJECT_DIR} ..."
gcloud compute scp --recurse \
  --zone="${ZONE}" --project="${PROJECT_ID}" \
  --compress \
  "${LOCAL_PROJECT_DIR}/worldserve" \
  "${LOCAL_PROJECT_DIR}/benchmarks" \
  "${LOCAL_PROJECT_DIR}/scripts" \
  "${LOCAL_PROJECT_DIR}/docs" \
  "${VM_NAME}:${REMOTE_PROJECT_DIR}/"

# Push HF token via env file (NOT scp'd into version control).
gcloud compute ssh "${VM_NAME}" \
  --zone="${ZONE}" --project="${PROJECT_ID}" \
  --command="cat > ${REMOTE_PROJECT_DIR}/.env <<EOF
HF_TOKEN=${HF_TOKEN}
EOF
chmod 600 ${REMOTE_PROJECT_DIR}/.env"

echo
echo "Sync done. Next:"
echo "  bash scripts/gcp/run.sh helios_baseline"
