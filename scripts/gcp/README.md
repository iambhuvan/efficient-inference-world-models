# GCP H100 setup — WorldServe benchmark runner

End-to-end pipeline to run any WorldServe benchmark on a GCP A3 VM with one
NVIDIA H100 80GB. Replaces Modal for environments that need direct GCP H100
access (course-credit accounts, university clusters, billing constraints).

## Quick start

```bash
# 1. Copy the template config and fill in your values
cp scripts/gcp/config.example.sh scripts/gcp/config.sh
# edit PROJECT_ID, ZONE, HF_TOKEN

# 2. Source it (every shell session)
source scripts/gcp/config.sh

# 3. Create the VM (one-time, ~3 min)
bash scripts/gcp/setup_vm.sh

# 4. Install python deps on the VM (one-time, ~5 min)
bash scripts/gcp/bootstrap_vm.sh

# 5. Push the project (re-run any time you change code locally)
bash scripts/gcp/sync_code.sh

# 6. Run a benchmark
bash scripts/gcp/run.sh helios_baseline
bash scripts/gcp/run.sh helios_baseline --num-iters 2

# 7. Stop the VM when done (don't pay overnight)
bash scripts/gcp/teardown_vm.sh stop      # preserves disk, fast restart
bash scripts/gcp/teardown_vm.sh delete    # permanent
```

## Pre-flight checklist

Before `setup_vm.sh` will succeed you need:

- [ ] **GCP project** with billing enabled (gcloud projects list)
- [ ] **gcloud CLI** installed locally and authenticated
      (`gcloud auth login && gcloud auth application-default login`)
- [ ] **H100 80GB quota** in your chosen zone — usually 0 by default,
      request via [console.cloud.google.com/iam-admin/quotas](https://console.cloud.google.com/iam-admin/quotas)
      (filter "NVIDIA H100 80GB GPUs", region us-central1 etc., set to 1+)
- [ ] **HF token** (`huggingface-cli login` or paste into config.sh)

`setup_vm.sh` checks H100 quota for you and warns loudly if it's zero.

## Cost estimates (Apr 2026)

| Variant | Hourly | 1 ablation cell (~30 min) | Full Helios stack (~6 hr) |
|---|---|---|---|
| `a3-highgpu-1g` on-demand | ~$11/hr | ~$5.50 | **~$66** |
| `a3-highgpu-1g` spot | ~$3.30/hr | ~$1.65 | **~$20** |

`USE_SPOT="true"` in config.sh saves ~70 % at the cost of possible mid-run
preemption. For short ablation cells (≤30 min each), spot is the right choice.
For final publication runs, switch to on-demand.

**Run `bash scripts/gcp/teardown_vm.sh stop` immediately after every session.**
A forgotten on-demand VM is $250+/day.

## Files in this directory

| File | Purpose |
|---|---|
| `config.example.sh` | Template — copy to `config.sh`, fill in, source |
| `setup_vm.sh` | Create the A3 VM with 1× H100 (idempotent) |
| `bootstrap_vm.sh` | Install pip deps on the VM (one-time) |
| `sync_code.sh` | rsync project source to the VM (re-run after edits) |
| `run.sh` | Execute a benchmark over SSH |
| `teardown_vm.sh` | Stop or delete the VM |
| `README.md` | This file |

## How `run.sh` resolves benchmark names

`bash scripts/gcp/run.sh <name>` searches `benchmarks/baseline/<name>.py`
first, then `benchmarks/optimised/<name>.py`. Any extra args are forwarded
to the Python script.

## Differences vs the Modal scripts

The existing `*_modal.py` scripts under `benchmarks/` use `@app.function`
decorators and Modal volume / secret bindings — those don't run outside
Modal. We provide a parallel **`*_baseline.py` / `*_<opt>.py`** family that
contains the same logic but without Modal wrappers (just plain Python).

The Modal scripts are kept intact — use whichever stack matches your access.

## Common issues

**"Quota exceeded — H100 80GB"**
Quota request, not a billing issue. Open the IAM Quotas page and submit a
written request with project justification. Course-project requests are
usually approved within 24–48 hours.

**"Image not found: pytorch-latest-gpu"**
Some regions have different image families. Try `pytorch-2-5-cu124-ubuntu-2204`
or browse `gcloud compute images list --project=deeplearning-platform-release`.

**"VM stuck preempted"**
Spot VMs can be preempted any time. `gcloud compute instances start ${VM_NAME}`
brings it back; data on the boot disk persists.

**"flash-attn install failed"**
Build takes 30+ min and sometimes hangs on A3. The `bootstrap_vm.sh` script
treats this as best-effort and continues. If you need FA3 specifically, run
`pip install flash-attn --no-build-isolation` manually with a longer timeout.

## Next: running the Helios baseline

```bash
source scripts/gcp/config.sh
bash scripts/gcp/run.sh helios_baseline
```

This pulls `BestWishYsh/Helios-Base` (~30 GB) into the VM's local SSD on first
run, then executes one warmup + one timed generation. Expected output:

```
Helios-Base baseline result:
{
  "model": "BestWishYsh/Helios-Base",
  "n_params_B": 14.3,
  "latency_ms_mean": 25000-40000,
  "frames_per_sec": 2-5,
  "vram_gb": 35-55,
  ...
}
```

Save this as cell A. Every subsequent optimization cell reports speedup
relative to it.
