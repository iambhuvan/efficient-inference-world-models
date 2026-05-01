#!/usr/bin/env bash
# Create a GCP A3 VM with one H100 80GB attached.
# Prereq: source ./config.sh first.
# Run once. Idempotent — exits cleanly if VM already exists.

set -euo pipefail

if [[ -z "${PROJECT_ID:-}" ]]; then
  echo "ERROR: source scripts/gcp/config.sh first." >&2
  exit 1
fi

# Set the active project.
gcloud config set project "${PROJECT_ID}"

# Enable required APIs (idempotent — no-op if already enabled).
echo "Enabling required GCP APIs..."
gcloud services enable \
  compute.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT_ID}"

# Check H100 quota — fail loud if zero.
echo "Checking H100 quota in ${REGION}..."
QUOTA=$(gcloud compute regions describe "${REGION}" \
  --format="value(quotas.filter(metric:NVIDIA_H100_80GB_GPUS).limit)" || echo "0")
echo "  H100_80GB quota in ${REGION}: ${QUOTA}"
if [[ "${QUOTA}" == "0" || -z "${QUOTA}" ]]; then
  echo
  echo "WARNING: Your H100 quota in ${REGION} appears to be 0."
  echo "Request quota at:"
  echo "  https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}"
  echo "Filter: 'NVIDIA H100 80GB GPUs', region '${REGION}', set to 1+."
  echo "(Course projects often need a written justification — be specific.)"
  echo
  read -rp "Continue anyway? [y/N] " yn
  [[ "${yn}" =~ ^[Yy]$ ]] || exit 1
fi

# Skip if VM already exists.
if gcloud compute instances describe "${VM_NAME}" \
     --zone="${ZONE}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "VM ${VM_NAME} already exists in ${ZONE}. Skipping create."
  exit 0
fi

SPOT_FLAG=""
if [[ "${USE_SPOT:-true}" == "true" ]]; then
  SPOT_FLAG="--provisioning-model=SPOT --instance-termination-action=STOP"
fi

echo "Creating ${VM_NAME} in ${ZONE} (${MACHINE_TYPE}, 1× H100 80GB)..."
# shellcheck disable=SC2086
gcloud compute instances create "${VM_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator="${ACCELERATOR}" \
  --image-family="${BOOT_IMAGE_FAMILY}" \
  --image-project="${BOOT_IMAGE_PROJECT}" \
  --boot-disk-size="${BOOT_DISK_SIZE_GB}GB" \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True" \
  --scopes="cloud-platform" \
  ${SPOT_FLAG}

echo
echo "VM created. Waiting 30 s for SSH to come online..."
sleep 30

echo "Testing SSH..."
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" \
  --command="nvidia-smi | head -n 20"

echo
echo "VM ready. Next:"
echo "  bash scripts/gcp/bootstrap_vm.sh    # install python deps on the VM"
echo "  bash scripts/gcp/sync_code.sh       # push project to the VM"
