#!/usr/bin/env bash
# Copy this file to config.sh, fill in your values, then `source config.sh`
# config.sh is .gitignore'd so secrets stay out of version control.

# ── GCP project / billing ────────────────────────────────────────────────
export PROJECT_ID="your-gcp-project-id"          # gcloud projects list
export BILLING_ACCOUNT_ID=""                     # optional — only if creating
                                                 #   a new project

# ── Region / zone (must support a3-highgpu-1g — 1× H100 80GB) ────────────
# As of Apr 2026, single-H100 zones include:
#   us-central1-a, us-central1-c
#   us-east5-a
#   europe-west4-b
#   asia-east1-c
# Run `gcloud compute accelerator-types list --filter="name:nvidia-h100-80gb"`
# to see live availability per zone.
export REGION="us-central1"
export ZONE="us-central1-a"

# ── VM identity ──────────────────────────────────────────────────────────
export VM_NAME="worldserve-h100"
export VM_USER="$(whoami)"                       # GCP creates Linux user from
                                                 #   your gcloud account email

# ── Machine type ────────────────────────────────────────────────────────
# a3-highgpu-1g  : 1× H100 80GB, 26 vCPU, 234 GB RAM     ~$11/hr on-demand
# a3-highgpu-8g  : 8× H100 80GB, 208 vCPU, 1872 GB RAM   ~$88/hr on-demand
# Use spot/preemptible for ~70% discount.
export MACHINE_TYPE="a3-highgpu-1g"
export ACCELERATOR="type=nvidia-h100-80gb,count=1"
export USE_SPOT="true"                           # "true" for spot pricing
                                                 # — VM may be preempted

# ── Boot disk (Deep Learning VM image ships CUDA 12.4 + PyTorch 2.5) ────
export BOOT_IMAGE_FAMILY="pytorch-latest-gpu"
export BOOT_IMAGE_PROJECT="deeplearning-platform-release"
export BOOT_DISK_SIZE_GB="500"                   # 14B model = ~28 GB FP16,
                                                 # plus VAE + cache headroom

# ── Hugging Face token (for any gated models you might run later) ────────
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# ── Project paths ───────────────────────────────────────────────────────
export LOCAL_PROJECT_DIR="$HOME/Desktop/MLSYS_FINAL_PROJECT"
export REMOTE_PROJECT_DIR="/home/${VM_USER}/MLSYS_FINAL_PROJECT"
