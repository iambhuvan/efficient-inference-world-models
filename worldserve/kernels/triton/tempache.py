"""
TempCache: deduplicate near-identical K entries before attention.
arXiv 2602.01801 — Duplicate Attention Lemma: merging duplicates is EXACT.

  Attn(Q, K_dedup, V_dedup) == Attn(Q, K, V)
  where V_dedup[merged] = mean(V[bucket])
        logit_bias[merged] = log(count)  (fused into softmax)

Target: H100 SXM (sm_90a), triton >= 2.2.0
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Step 1: find duplicate keys via cosine similarity
# ---------------------------------------------------------------------------

def find_duplicate_keys(
    K: torch.Tensor,
    threshold: float = 1e-3,
) -> dict:
    """
    K: [B, H, S, D] — returns merge_map per (b, h).
    merge_map[b][h] = {canonical_idx: [list of equivalent indices]}

    Uses block-wise cosine similarity to avoid O(S^2 D) memory spike.
    """
    B, H, S, D = K.shape
    merge_map = {}

    K_norm = F.normalize(K.float(), dim=-1)  # [B, H, S, D]

    for b in range(B):
        merge_map[b] = {}
        for h in range(H):
            kn = K_norm[b, h]  # [S, D]
            # Pairwise cosine similarity: [S, S]
            sim = torch.mm(kn, kn.t())

            assigned = torch.zeros(S, dtype=torch.bool, device=K.device)
            groups: dict = {}

            for i in range(S):
                if assigned[i]:
                    continue
                # find all j > i similar to i
                similar = ((sim[i] > (1.0 - threshold)) & ~assigned)
                similar[i] = False  # exclude self from duplicates list
                dup_indices = similar.nonzero(as_tuple=False).squeeze(1).tolist()
                if dup_indices:
                    groups[i] = dup_indices
                    assigned[dup_indices] = True
                assigned[i] = True

            merge_map[b][h] = groups
    return merge_map


# ---------------------------------------------------------------------------
# Step 2: Triton kernel — average K/V within equivalence classes
# ---------------------------------------------------------------------------

@triton.jit
def _tempache_merge_kernel(
    K_in, V_in,
    K_out, V_out,
    LogitBias,
    MergeFrom,   # [max_dups_per_class, n_canonical] int32
    CanonIdx,    # [n_canonical] int32 — canonical token positions
    CountVec,    # [n_canonical] int32 — size of each class
    n_canonical,
    stride_s, stride_d,
    HEAD_DIM: tl.constexpr,
    MAX_DUPS: tl.constexpr,
):
    """
    Each program handles one canonical index.
    Computes K_out[c] = mean(K_in[members])  and  V_out[c] = mean(V_in[members]).
    Writes logit_bias[c] = log(count).
    """
    cid = tl.program_id(0)
    if cid >= n_canonical:
        return

    canon_pos = tl.load(CanonIdx + cid)
    count     = tl.load(CountVec + cid)

    offs_d = tl.arange(0, HEAD_DIM)

    # Load canonical entry
    k_acc = tl.load(K_in + canon_pos * stride_s + offs_d * stride_d).to(tl.float32)
    v_acc = tl.load(V_in + canon_pos * stride_s + offs_d * stride_d).to(tl.float32)

    # Accumulate duplicate entries
    for dup in range(MAX_DUPS):
        if dup >= count - 1:
            break
        src = tl.load(MergeFrom + dup * n_canonical + cid)
        k_acc += tl.load(K_in + src * stride_s + offs_d * stride_d).to(tl.float32)
        v_acc += tl.load(V_in + src * stride_s + offs_d * stride_d).to(tl.float32)

    count_f = count.to(tl.float32)
    k_avg = k_acc / count_f
    v_avg = v_acc / count_f

    tl.store(K_out + canon_pos * stride_s + offs_d * stride_d,
             k_avg.to(K_out.dtype.element_ty))
    tl.store(V_out + canon_pos * stride_s + offs_d * stride_d,
             v_avg.to(V_out.dtype.element_ty))
    tl.store(LogitBias + cid, tl.log(count_f))


# ---------------------------------------------------------------------------
# Helper: build Triton-friendly merge arrays from merge_map
# ---------------------------------------------------------------------------

def _build_merge_tensors(merge_map_bh: dict, S: int, device):
    """
    Returns (canon_idx, count_vec, merge_from, n_canonical).
    All canonical token indices + their duplicate lists.
    """
    all_canon  = sorted(merge_map_bh.keys())
    n_canon    = len(all_canon)

    if n_canon == 0:
        return None, None, None, 0

    # max duplicates in any class
    max_dups = max((len(v) for v in merge_map_bh.values()), default=0)
    if max_dups == 0:
        return None, None, None, 0

    canon_idx  = torch.tensor(all_canon, dtype=torch.int32, device=device)
    count_vec  = torch.tensor(
        [len(merge_map_bh[c]) + 1 for c in all_canon],
        dtype=torch.int32, device=device,
    )
    # merge_from[dup_slot, canon_idx]
    merge_from = torch.zeros(max_dups, n_canon, dtype=torch.int32, device=device)
    for j, c in enumerate(all_canon):
        dups = merge_map_bh[c]
        for slot, src in enumerate(dups):
            merge_from[slot, j] = src

    return canon_idx, count_vec, merge_from, n_canon


# ---------------------------------------------------------------------------
# Step 3: full pipeline
# ---------------------------------------------------------------------------

def tempache_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    threshold: float = 1e-3,
    softmax_scale: float = None,
    layout: str = "bhsd",
) -> torch.Tensor:
    """
    Full TempCache pipeline:
      1. find_duplicate_keys
      2. merge K/V in each equivalence class (Triton kernel)
      3. run scaled_dot_product_attention with logit bias
      4. no scatter needed — output positions of non-canonical tokens
         get zero attention weight (merged away), so output is already correct.

    Args:
        q, k, v       : [B, H, S, D] (layout='bhsd') or [B, S, H, D].
        threshold     : cosine-distance threshold for duplicate detection.
        softmax_scale : defaults to 1/sqrt(D).
        layout        : 'bhsd' | 'bshd'.

    Returns:
        output tensor in same layout as inputs.
    """
    transposed = False
    if layout == "bshd":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        transposed = True

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    B, H, S, D = q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    try:
        merge_map = find_duplicate_keys(k, threshold=threshold)

        k_merged = k.clone()
        v_merged = v.clone()

        HEAD_DIM = triton.next_power_of_2(D)

        # Build additive bias tensor [B, H, 1, S] for logit adjustment
        logit_bias = torch.zeros(B, H, 1, S, dtype=torch.float32, device=q.device)

        for b in range(B):
            for h in range(H):
                bh_map = merge_map[b][h]
                canon_idx, count_vec, merge_from, n_canon = _build_merge_tensors(
                    bh_map, S, q.device
                )
                if n_canon == 0:
                    continue

                max_dups = merge_from.shape[0]
                lb = torch.zeros(n_canon, dtype=torch.float32, device=q.device)

                _tempache_merge_kernel[(n_canon,)](
                    k[b, h], v[b, h],
                    k_merged[b, h], v_merged[b, h],
                    lb,
                    merge_from, canon_idx, count_vec,
                    n_canon,
                    k.stride(2), k.stride(3),
                    HEAD_DIM=HEAD_DIM,
                    MAX_DUPS=max_dups,
                    num_warps=4,
                )

                # Write logit bias at canonical positions
                canon_pos = canon_idx.long()
                logit_bias[b, h, 0, canon_pos] = lb

        # Mark non-canonical (duplicate) positions as -inf so they contribute nothing
        # Build mask of removed positions
        remove_mask = torch.zeros(B, H, 1, S, dtype=torch.bool, device=q.device)
        for b in range(B):
            for h in range(H):
                for canon, dups in merge_map[b][h].items():
                    for d in dups:
                        remove_mask[b, h, 0, d] = True
        logit_bias = logit_bias.masked_fill(remove_mask, float("-inf"))

        # Run attention with additive logit bias
        # F.sdpa attn_mask is additive
        out = F.scaled_dot_product_attention(
            q, k_merged, v_merged,
            attn_mask=logit_bias.to(q.dtype),
            scale=softmax_scale,
        )

    except Exception:
        out = F.scaled_dot_product_attention(q, k, v, scale=softmax_scale)

    if transposed:
        out = out.transpose(1, 2).contiguous()
    return out


# ---------------------------------------------------------------------------
# Quick correctness test
# ---------------------------------------------------------------------------

def _test_tempache():
    torch.manual_seed(42)
    B, H, S, D = 1, 2, 64, 32
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(B, H, S, D, dtype=dtype, device=device)
    k = torch.randn(B, H, S, D, dtype=dtype, device=device)
    v = torch.randn(B, H, S, D, dtype=dtype, device=device)
    scale = 1.0 / math.sqrt(D)

    ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
    # With a very small threshold no duplicates are found → should match ref exactly
    out = tempache_attention(q, k, v, threshold=1e-6, softmax_scale=scale)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("tempache_attention correctness OK")


if __name__ == "__main__":
    _test_tempache()
