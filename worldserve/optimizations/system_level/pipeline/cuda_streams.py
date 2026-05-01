"""
CUDA stream pipelining for Matrix-Game 2.0.

Overlaps VAE decode of block N with denoising of block N+1, exploiting
the fact that VAE decode is memory-bound while DiT denoising is
compute-bound -- they can share the GPU without contention.

This is MG2-specific because MG2 generates video in temporal blocks
that can be pipelined. Oasis generates all frames in a single pass.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Guard CUDA-specific operations for CPU-only environments
_CUDA_AVAILABLE = torch.cuda.is_available()


class CUDAStreamPipeline:
    """
    Overlaps VAE decoding with DiT denoising using CUDA streams.

    For MG2's block-by-block video generation:
      - Stream 1 (denoise_stream): runs DiT denoising for block N+1
      - Stream 2 (decode_stream): runs VAE decode for block N

    The two operations overlap on the GPU because they stress different
    hardware units (tensor cores vs memory bandwidth).

    Usage::

        pipeline = CUDAStreamPipeline()
        pipeline.setup_streams()
        frames = pipeline.run_overlapped(denoise_fn, decode_fn, num_blocks=8)
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """
        Args:
            device: CUDA device. Defaults to cuda:0 if available.
        """
        if device is not None:
            self.device = device
        elif _CUDA_AVAILABLE:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            logger.warning(
                "CUDA not available. CUDAStreamPipeline will run sequentially on CPU."
            )

        self.denoise_stream: Optional[torch.cuda.Stream] = None
        self.decode_stream: Optional[torch.cuda.Stream] = None
        self._streams_ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup_streams(self) -> None:
        """
        Create dedicated CUDA streams for denoising and decoding.

        Uses high-priority streams to minimize scheduling latency.
        """
        if not _CUDA_AVAILABLE:
            logger.warning("CUDA not available; streams not created.")
            return

        # Use high priority (-1) for both streams to avoid being preempted
        self.denoise_stream = torch.cuda.Stream(device=self.device, priority=-1)
        self.decode_stream = torch.cuda.Stream(device=self.device, priority=-1)
        self._streams_ready = True

        logger.info(
            "Created CUDA streams on %s (denoise: %s, decode: %s)",
            self.device,
            self.denoise_stream,
            self.decode_stream,
        )

    def run_overlapped(
        self,
        denoise_fn: Callable[[int], torch.Tensor],
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        num_blocks: int,
    ) -> List[torch.Tensor]:
        """
        Run denoising and decoding in overlapped fashion.

        Timeline (for blocks 0..N-1)::

            Block 0: denoise[0] on denoise_stream
            Block 1: decode[0] on decode_stream  ||  denoise[1] on denoise_stream
            Block 2: decode[1] on decode_stream  ||  denoise[2] on denoise_stream
            ...
            Block N:  decode[N-1] on decode_stream

        Args:
            denoise_fn: Callable that takes block_index (int) and returns
                        a latent tensor (the denoised output for that block).
            decode_fn:  Callable that takes a latent tensor and returns
                        decoded frames (pixel-space).
            num_blocks: Number of temporal blocks to process.

        Returns:
            List of decoded frame tensors, one per block.
        """
        if not self._streams_ready or not _CUDA_AVAILABLE:
            # Sequential fallback
            return self._run_sequential(denoise_fn, decode_fn, num_blocks)

        assert self.denoise_stream is not None
        assert self.decode_stream is not None

        decoded_frames: List[torch.Tensor] = []
        latent_buffer: Optional[torch.Tensor] = None

        # Per-iteration events — created fresh each loop to avoid
        # waiting on unrecorded events.
        prev_denoise_done: Optional[torch.cuda.Event] = None
        prev_decode_done: Optional[torch.cuda.Event] = None

        for block_idx in range(num_blocks):
            denoise_done = torch.cuda.Event()

            # --- Denoise block_idx on denoise_stream ---
            with torch.cuda.stream(self.denoise_stream):
                if prev_decode_done is not None:
                    # Wait for previous decode to finish before reusing
                    # any shared resources (e.g., KV cache)
                    self.denoise_stream.wait_event(prev_decode_done)

                new_latent = denoise_fn(block_idx)
                # Mark this tensor as used on the denoise stream for
                # memory safety when it's consumed on decode_stream
                new_latent.record_stream(self.decode_stream)
                denoise_done.record(self.denoise_stream)

            # --- Decode previous block on decode_stream (overlapped) ---
            if latent_buffer is not None and prev_denoise_done is not None:
                decode_done = torch.cuda.Event()
                with torch.cuda.stream(self.decode_stream):
                    # Wait for the denoise that produced latent_buffer
                    self.decode_stream.wait_event(prev_denoise_done)
                    frames = decode_fn(latent_buffer)
                    frames.record_stream(torch.cuda.default_stream(self.device))
                    decode_done.record(self.decode_stream)
                    decoded_frames.append(frames)
                prev_decode_done = decode_done

            prev_denoise_done = denoise_done
            latent_buffer = new_latent

        # --- Decode the last block ---
        if latent_buffer is not None and prev_denoise_done is not None:
            with torch.cuda.stream(self.decode_stream):
                self.decode_stream.wait_event(prev_denoise_done)
                frames = decode_fn(latent_buffer)
                frames.record_stream(torch.cuda.default_stream(self.device))
                decoded_frames.append(frames)

        # Synchronize before returning
        torch.cuda.synchronize(self.device)

        logger.info(
            "Overlapped pipeline: processed %d blocks (%d decoded frames).",
            num_blocks, len(decoded_frames),
        )
        return decoded_frames

    def benchmark_overlap_vs_sequential(
        self,
        denoise_fn: Callable[[int], torch.Tensor],
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        num_blocks: int = 8,
        warmup_runs: int = 1,
        benchmark_runs: int = 3,
    ) -> dict:
        """
        Measure the actual speedup of overlapped vs sequential execution.

        Args:
            denoise_fn: See run_overlapped.
            decode_fn: See run_overlapped.
            num_blocks: Number of temporal blocks.
            warmup_runs: Warmup iterations (not timed).
            benchmark_runs: Timed iterations.

        Returns:
            Dict with keys: sequential_ms, overlapped_ms, speedup, overlap_efficiency.
        """
        if not _CUDA_AVAILABLE:
            logger.warning("CUDA not available; returning dummy benchmark results.")
            return {
                "sequential_ms": 0.0,
                "overlapped_ms": 0.0,
                "speedup": 1.0,
                "overlap_efficiency": 0.0,
            }

        if not self._streams_ready:
            self.setup_streams()

        # --- Warmup ---
        for _ in range(warmup_runs):
            self._run_sequential(denoise_fn, decode_fn, num_blocks)
            self.run_overlapped(denoise_fn, decode_fn, num_blocks)

        # --- Benchmark sequential ---
        torch.cuda.synchronize(self.device)
        seq_times = []
        for _ in range(benchmark_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._run_sequential(denoise_fn, decode_fn, num_blocks)
            end.record()
            torch.cuda.synchronize(self.device)
            seq_times.append(start.elapsed_time(end))

        # --- Benchmark overlapped ---
        overlap_times = []
        for _ in range(benchmark_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.run_overlapped(denoise_fn, decode_fn, num_blocks)
            end.record()
            torch.cuda.synchronize(self.device)
            overlap_times.append(start.elapsed_time(end))

        seq_ms = sum(seq_times) / len(seq_times)
        ovl_ms = sum(overlap_times) / len(overlap_times)
        speedup = seq_ms / max(ovl_ms, 1e-6)

        # Overlap efficiency: 1.0 means perfect overlap (decode is free),
        # 0.0 means no overlap at all.
        overlap_efficiency = max(0.0, 1.0 - ovl_ms / max(seq_ms, 1e-6))

        results = {
            "sequential_ms": seq_ms,
            "overlapped_ms": ovl_ms,
            "speedup": speedup,
            "overlap_efficiency": overlap_efficiency,
        }

        logger.info(
            "Benchmark: sequential=%.1fms, overlapped=%.1fms, "
            "speedup=%.2fx, efficiency=%.1f%%",
            seq_ms, ovl_ms, speedup, overlap_efficiency * 100,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_sequential(
        self,
        denoise_fn: Callable[[int], torch.Tensor],
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        num_blocks: int,
    ) -> List[torch.Tensor]:
        """Run denoise -> decode sequentially for each block (baseline)."""
        decoded_frames: List[torch.Tensor] = []
        for block_idx in range(num_blocks):
            latent = denoise_fn(block_idx)
            frames = decode_fn(latent)
            decoded_frames.append(frames)
        if _CUDA_AVAILABLE:
            torch.cuda.synchronize(self.device)
        return decoded_frames

    def cleanup(self) -> None:
        """Release CUDA streams."""
        self.denoise_stream = None
        self.decode_stream = None
        self._streams_ready = False
        logger.info("CUDA streams released.")
