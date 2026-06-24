from __future__ import annotations

import logging
import pathlib

import torch


logger = logging.getLogger(__name__)


class InferenceProfiler:
    def __init__(
        self,
        output_dir: str | pathlib.Path,
        steps: int,
        skip_first: int,
        device: torch.device,
        compile_active: bool = False,
        count_flops: bool = False,
    ) -> None:
        self.output_dir = pathlib.Path(output_dir).expanduser()
        self.steps = int(steps)
        self.skip_first = int(skip_first)
        self.max_steps = self.steps + self.skip_first
        self.device = device
        self.compile_active = compile_active
        self.count_flops = count_flops
        self.profiler = None
        self.flop_counter = None
        self.step_count = 0
        self.done = False

    def start(self) -> None:
        if self.steps <= 0:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=0, warmup=0, active=self.steps, repeat=1, skip_first=self.skip_first,
            ),
            record_shapes=True,
            profile_memory=True,
        )
        self.profiler.__enter__()
        if self.count_flops:
            try:
                from torch.utils.flop_counter import FlopCounterMode
                self.flop_counter = FlopCounterMode(display=False, depth=4)
                self.flop_counter.__enter__()
            except Exception as exc:
                logger.warning("FlopCounterMode unavailable: %s", exc)
                self.flop_counter = None
        logger.info(
            "InferenceProfiler started: window=%d, skip_first=%d, dir=%s",
            self.steps, self.skip_first, self.output_dir,
        )

    def step(self) -> None:
        if self.profiler is None or self.done:
            return
        self.profiler.step()
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.finalize()

    def finalize(self) -> None:
        if self.profiler is None or self.done:
            return
        self.done = True
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        summary = self._render_summary()
        (self.output_dir / "summary.txt").write_text(summary, encoding="utf-8")
        self.profiler.export_chrome_trace(str(self.output_dir / "trace.json"))
        self.profiler.__exit__(None, None, None)
        self.profiler = None
        if self.flop_counter is not None:
            self.flop_counter.__exit__(None, None, None)
        logger.info("Profile saved: %s", self.output_dir)

    def _render_summary(self) -> str:
        sort_by = "self_cuda_time_total" if self.device.type == "cuda" else "self_cpu_time_total"
        parts = [self.profiler.key_averages().table(sort_by=sort_by, row_limit=30)]
        if self.flop_counter is not None:
            parts.append(self._render_flops())
        return "\n\n".join(parts)

    def _render_flops(self) -> str:
        counts = self.flop_counter.flop_counts
        total = int(self.flop_counter.get_total_flops())
        n = max(1, self.step_count)
        lines = [
            "=" * 70,
            f"FlopCounterMode over {n} request(s)",
            "=" * 70,
            f"Total:       {total/1e12:.3f} TFLOPs ({total:,})",
            f"Per request: {total/n/1e12:.3f} TFLOPs",
            "",
            f"{'Module':<70} {'GFLOPs':>10} {'% total':>9}",
            "-" * 91,
        ]
        rows = []
        for name, op_dict in counts.items():
            depth = name.count(".") if name else 0
            if depth <= 4:
                rows.append((name or "Global", sum(int(v) for v in op_dict.values())))
        rows.sort(key=lambda x: -x[1])
        for name, fl in rows[:40]:
            pct = 100.0 * fl / total if total else 0.0
            lines.append(f"{name[:70]:<70} {fl/1e9:>10.2f} {pct:>8.2f}%")
        lines.append("")
        lines.append("Top operators:")
        ops: dict[str, int] = {}
        for op_dict in counts.values():
            for op, v in op_dict.items():
                ops[str(op)] = ops.get(str(op), 0) + int(v)
        for op, fl in sorted(ops.items(), key=lambda x: -x[1])[:15]:
            pct = 100.0 * fl / total if total else 0.0
            lines.append(f"  {op:<60s} {fl/1e9:>10.2f} GFLOPs ({pct:>6.2f}%)")
        return "\n".join(lines)
