#!/usr/bin/env bash
# Stop or delete the GCP H100 VM to stop incurring charges.
# Default: stop (preserves disk + IP, fast restart, cheap).
# Pass --delete to remove the VM entirely.

set -euo pipefail

if [[ -z "${PROJECT_ID:-}" ]]; then
  echo "ERROR: source scripts/gcp/config.sh first." >&2
  exit 1
fi

ACTION="${1:-stop}"

case "${ACTION}" in
  stop|--stop)
    echo "Stopping ${VM_NAME} (preserves disk; cheap)..."
    gcloud compute instances stop "${VM_NAME}" \
      --zone="${ZONE}" --project="${PROJECT_ID}"
    echo "VM stopped. Restart with:"
    echo "  gcloud compute instances start ${VM_NAME} --zone=${ZONE}"
    ;;
  delete|--delete)
    echo "WARNING: this will permanently delete ${VM_NAME} AND its disk."
    read -rp "Type the VM name to confirm: " confirm
    if [[ "${confirm}" != "${VM_NAME}" ]]; then
      echo "Aborted."
      exit 1
    fi
    gcloud compute instances delete "${VM_NAME}" \
      --zone="${ZONE}" --project="${PROJECT_ID}" --quiet
    echo "VM deleted."
    ;;
  *)
    echo "Usage: $0 [stop|delete]" >&2
    exit 1
    ;;
esac
