"""torch.profiler helpers used by training workspaces."""

from __future__ import annotations

import os
from typing import Callable

import torch.profiler as torch_profiler


def make_profiler_trace_handler(
    output_dir: str,
    device_peak_tflops: float = 312.0,
    table_row_limit: int = 20,
) -> Callable:
    """Build an ``on_trace_ready`` handler for ``torch.profiler.profile``.

    The handler prints GPU/CPU bottleneck tables, a FLOPS / MFU summary
    when the profiler captured FLOP counts, and exports a Chrome trace
    JSON to ``{output_dir}/trace/trace_step_{step}.json``.

    Args:
        output_dir: Workspace output directory; the ``trace/`` subdir is
            created on first call if missing.
        device_peak_tflops: Peak TFLOPS of the target device for MFU%.
            Default is 312 (A800 bf16 peak).
        table_row_limit: Number of rows printed in each bottleneck table.
    """
    trace_dir = os.path.join(output_dir, "trace")

    def trace_handler(p: torch_profiler.profile) -> None:
        # GPU bottlenecks
        output_gpu = p.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=table_row_limit
        )
        print("--- GPU Bottlenecks ---")
        print(output_gpu)

        # CPU bottlenecks
        output_cpu = p.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=table_row_limit
        )
        print("\n--- CPU Bottlenecks ---")
        print(output_cpu)

        # FLOPS summary: total measured FLOPS across all CUDA kernels.
        # FunctionEventAvg uses device_time_total / self_device_time_total
        # (not cuda_time_total which only exists on FunctionEvent).
        # Source: pytorch/torch/autograd/profiler_util.py FunctionEventAvg
        try:
            events = p.key_averages()
            total_flops = sum(e.flops for e in events if e.flops > 0)
            total_device_us = sum(e.self_device_time_total for e in events)
            total_device_time_s = total_device_us / 1e6
            print("\n--- FLOPS Summary ---")
            if total_flops > 0 and total_device_time_s > 0:
                tflops_per_sec = total_flops / total_device_time_s / 1e12
                print(f"  Total FLOPS: {total_flops / 1e12:.2f} TFLOPS")
                print(f"  Device time: {total_device_time_s:.3f}s")
                print(f"  Throughput:  {tflops_per_sec:.1f} TFLOPS/s")
                print(
                    f"  MFU (vs {device_peak_tflops:g} TFLOPS peak): "
                    f"{tflops_per_sec / device_peak_tflops * 100:.1f}%"
                )
            else:
                print(
                    "  No FLOPS data (with_flops only covers matmul/conv2d, "
                    "compiled kernels may not report)"
                )
        except Exception as flops_err:
            print(f"\n[WARN] FLOPS summary failed: {flops_err}")

        os.makedirs(trace_dir, exist_ok=True)
        p.export_chrome_trace(os.path.join(trace_dir, f"trace_step_{p.step_num}.json"))

    return trace_handler
