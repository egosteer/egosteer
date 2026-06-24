if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import math
import os
import pathlib
import pickle
import random
import time
from contextlib import nullcontext

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import wandb
from omegaconf import OmegaConf
from torch.profiler import (
    ProfilerActivity,
    profile as torch_profile,
    schedule as profiler_schedule,
)
from torch.utils.data import DataLoader

from .base_workspace import BaseWorkspace
from src.policy.egosteer import EgoSteer
from src.utils.checkpoint_util import (
    TopKCheckpointManager,
    enable_pathlib_local_pickle_compat,
    load_checkpoint,
)
from src.utils.distributed_utils import (
    apply_fsdp2,
    build_mixed_precision_policy,
    init_distributed,
)
from src.utils.fsdp_app_state import APP_STATE_KEY, FSDPWorkspaceAppState
from src.utils.profiler_utils import make_profiler_trace_handler
from src.utils.scheduler_utils import build_lr_scheduler
from src.utils.training_utils import (
    DeviceTransferWrapper,
    FullMemoryTracker,
    GarbageCollection,
    TrainingState,
    build_param_groups,
    build_training_step_log,
    capture_output_to_training_log,
    clip_and_check_grads,
    data_worker_init,
    reset_run_seed,
)
from src.workspace.eval_utils import (
    evaluation,
    save_checkpoint_native,
    save_interval_ckpt,
    save_topk_ckpt,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainEgoSteerWorkspace(BaseWorkspace):

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: EgoSteer
        self.model = hydra.utils.instantiate(cfg.policy)
        gc_cfg = cfg.training.get("gradient_checkpointing", False)
        if not isinstance(gc_cfg, bool) and gc_cfg is not None:
            gc_cfg = OmegaConf.to_container(gc_cfg, resolve=True)
        self.model.enable_gradient_checkpointing(gc_cfg)
        self.tracker = FullMemoryTracker(self.model)

        self.dtype = torch.bfloat16 if cfg.training.use_bf16 else torch.float32
        self.training_state = TrainingState()
        self.epoch = 0
        self.update_step = 0
        self.global_step = 0
        # Per-cycle accumulator for raw_loss components, used to log the
        # cycle mean to wandb instead of just the sync-step microbatch.
        # Only populated when grad_accum_steps > 1.
        self._cycle_loss_accum: dict = {}
        self.compile_cfg = cfg.training.get("compile", {})

    def maybe_compile_model(self, rank):
        if not self.compile_cfg.get("enabled", False):
            return

        # apply_patch() runs at qwen3_vl_backbone import time; no need to call here.
        compile_cfg = OmegaConf.to_container(self.compile_cfg, resolve=True)
        compile_kwargs = {
            key: value
            for key, value in compile_cfg.items()
            if key != "enabled" and value is not None
        }
        if rank == 0:
            print(f"Compiling blocks with kwargs: {compile_kwargs}")
        self.model.compile_blocks(compile_kwargs)

    @staticmethod
    def unwrap_optimizer(optimizer):
        return getattr(optimizer, "optimizer", optimizer)

    @staticmethod
    def get_vlm_stage_steps(training_cfg) -> tuple[int, int]:
        freeze_steps = int(training_cfg.get("vlm_freeze_steps", 0))
        rewarmup_steps = int(training_cfg.get("vlm_rewarmup_steps", 0))
        return freeze_steps, rewarmup_steps

    def is_vlm_freeze_active(self) -> bool:
        return bool(self.vlm_group_indices) and self.update_step < self.vlm_freeze_updates

    def get_param_group_lrs(self) -> tuple[list[float], list[int]]:
        optimizer = self.unwrap_optimizer(self.optimizer)
        current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        non_vlm_group_indices = [
            idx for idx in range(len(current_lrs)) if idx not in self.vlm_group_indices
        ]
        return current_lrs, non_vlm_group_indices

    # ------------------------------------------------------------------
    # Top-level orchestrator + 5 sub-methods (setup_runtime, setup_training,
    # train_step, maybe_log_and_ckpt, train_loop).
    # ------------------------------------------------------------------

    @capture_output_to_training_log
    def run(self):
        cfg = copy.deepcopy(self.cfg)
        ctx, profiler = self.setup_runtime(cfg)
        train_dataloader, val_dataloader, micro_batches_per_epoch, topk_manager = self.setup_training(cfg, ctx)
        self.train_loop(cfg, ctx, profiler, train_dataloader, val_dataloader, micro_batches_per_epoch, topk_manager)
        if ctx.rank == 0:
            wandb.finish()
        dist.destroy_process_group()

    def setup_runtime(self, cfg):
        """Distributed init, profiler, wandb, output-dir broadcast, seed reset."""
        ctx = init_distributed(backend="nccl")
        rank, world_size, device = ctx.rank, ctx.world_size, ctx.device

        if rank == 0:
            mesh_desc = (
                f"HSDP ({ctx.mesh.mesh.shape[0]} nodes x {ctx.mesh.mesh.shape[1]} GPUs/node)"
                if ctx.mesh is not None else f"plain FSDP ({world_size} GPUs)"
            )
            print("=" * 80)
            print("Distributed Training Info:")
            print(f"  backend: nccl")
            print(f"  dtype: {'bf16' if cfg.training.use_bf16 else 'fp32'}")
            print(f"  world_size: {world_size}")
            print(f"  rank: {rank}")
            print(f"  device: {device}")
            print(f"  mesh: {mesh_desc}")
            print("=" * 80)

        profiler = None
        if cfg.training.profile:
            os.makedirs(f"{self.output_dir}/trace", exist_ok=True)
            profiler = torch_profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=profiler_schedule(wait=1, warmup=2, active=10, repeat=3, skip_first=50),
                on_trace_ready=make_profiler_trace_handler(self.output_dir),
                with_flops=True,
            )

        if rank == 0:
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            project_name = wandb_cfg.pop("project")
            wandb.init(
                project=project_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                **wandb_cfg,
            )

        # Align output_dir across ranks so checkpoints land in one place.
        objects_to_broadcast = [self.output_dir if rank == 0 else None]
        dist.broadcast_object_list(objects_to_broadcast, src=0)
        self._output_dir = objects_to_broadcast[0]
        dist.barrier()

        self.run_seed = reset_run_seed(
            base_seed=int(cfg.training.seed),
            dynamic_data_seed=bool(cfg.training.get("dynamic_data_seed", False)),
            rank=rank,
        )
        return ctx, profiler

    def setup_training(self, cfg, ctx):
        """Build model state, dataloaders, optimizer, scheduler, checkpoint manager."""
        rank, world_size, device = ctx.rank, ctx.world_size, ctx.device
        model = self.model

        # Finetune weights (model only — optimizer + scheduler start fresh).
        if cfg.training.finetune_checkpoint_path:
            if rank == 0:
                print(f"[ckpt] finetune: loading model weights from {cfg.training.finetune_checkpoint_path}")
            load_checkpoint(model, cfg.training.finetune_checkpoint_path)
            if rank == 0:
                print("[ckpt] finetune: loaded (model weights only)")

        # Pre-FSDP freezes. Param groups are collected AFTER fully_shard()
        # because FSDP2 replaces module._parameters[name] with new Parameter
        # objects wrapping sharded DTensors; pre-shard refs would be orphaned.
        if not cfg.training.train_vlm:
            model.freeze_vlm_weights()

        # Without VLM data, language_model.norm.weight has no grad path;
        # freeze it so it stays out of the optimizer and the DCP schema is
        # stable across runs (no missing Adam-state keys on resume).
        if OmegaConf.select(cfg, "dataset.vlm_dataset", default=None) is None:
            model.freeze_final_lm_norm()

        self.vlm_freeze_updates, self.vlm_rewarmup_updates = self.get_vlm_stage_steps(cfg.training)

        # Dataset + dataloaders.
        print("--> Configure WebDataset dataset and dataloader...")
        dataset = hydra.utils.instantiate(cfg.dataset)
        self.use_relative_action = dataset.vla_dataset.use_relative_action
        print("--> dataset instantiated")
        dist.barrier()

        data_collator = hydra.utils.instantiate(cfg.data_collator)
        dataset.vla_dataset.set_collator(data_collator)
        if dataset.vlm_dataset is not None:
            dataset.vlm_dataset.set_collator(data_collator)

        assert cfg.training.normalizer_path is not None, (
            "WebDataset training requires a pre-computed normalizer_path."
        )
        print("Loading normalizer...")
        with open(cfg.training.normalizer_path, "rb") as f:
            normalizer = pickle.load(f)
        dataset.vla_dataset.set_normalizer(normalizer)
        self.normalizer = normalizer

        webloader_cfg = cfg.dataloader.get("webloader", {}) or {}
        use_webloader = bool(webloader_cfg.get("use_webloader", False))
        if use_webloader:
            import webdataset as wds
            train_loader_kwargs = dict(cfg.dataloader.loader)
            train_batch_size = train_loader_kwargs.pop("batch_size")
            cross_worker_shuffle = int(webloader_cfg.get("cross_worker_shuffle", 0))
            train_dataloader = wds.WebLoader(
                dataset=dataset,
                batch_size=None,
                worker_init_fn=data_worker_init,
                **train_loader_kwargs,
            )
            if cross_worker_shuffle > 0:
                train_dataloader = train_dataloader.shuffle(cross_worker_shuffle)
            train_dataloader = train_dataloader.batched(
                train_batch_size, collation_fn=dataset.get_collator(),
            )
        else:
            train_dataloader = DataLoader(
                dataset=dataset,
                collate_fn=dataset.get_collator(),
                worker_init_fn=data_worker_init,
                **cfg.dataloader.loader,
            )
        # Eval is purely local (eval_with_unsharded_model pre-unshards FSDP params),
        # so unequal batch counts across ranks are safe.
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(
            dataset=val_dataset,
            collate_fn=val_dataset.get_collator(),
            worker_init_fn=data_worker_init,
            **cfg.val_dataloader.loader,
        )

        update_steps_per_epoch = cfg.training.get("steps_per_epoch", 100000)
        train_dataloader = DeviceTransferWrapper(train_dataloader, device)
        val_dataloader = DeviceTransferWrapper(val_dataloader, device)

        # LR schedule — no accelerate wrapping, so scheduler steps map 1:1 to update steps.
        grad_accum_steps = int(cfg.training.get("gradient_accumulation_steps", 1))

        # steps_per_epoch is configured in optimizer-update steps. The dataloader loop
        # counts micro-batches, so it must run grad_accum micro-batches per update step.
        num_update_steps_per_epoch = update_steps_per_epoch
        max_train_steps = num_update_steps_per_epoch * cfg.training.num_epochs
        micro_batches_per_epoch = update_steps_per_epoch * grad_accum_steps
        if cfg.training.max_train_steps is not None:
            max_train_steps = cfg.training.max_train_steps
        num_warmup_steps = cfg.training.lr_warmup_steps
        if rank == 0:
            print(f"num_warmup_steps: {num_warmup_steps}, max_train_steps: {max_train_steps}")
            if self.vlm_freeze_updates > 0:
                print(
                    f"VLM staged training: freeze for {self.vlm_freeze_updates} update steps, "
                    f"then re-warmup for {self.vlm_rewarmup_updates} steps"
                )
        self.lr_schedule_kwargs = dict(
            schedule_name=cfg.training.lr_scheduler,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
            vlm_freeze_steps=self.vlm_freeze_updates,
            vlm_rewarmup_steps=self.vlm_rewarmup_updates,
        )

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        # FSDP2 wrap. Sub-modules sharded first, then the root module.
        from src.model.vlm.qwen3_vl_backbone import Qwen3VLTextDecoderLayerWithKV
        from src.model.vlm.qwen3_expert import DiTQwen3DecoderLayer
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock
        except ImportError as e:
            raise ImportError(
                "FSDP wrap requires transformers shipping Qwen3VLVisionBlock; "
                "please upgrade transformers."
            ) from e
        fsdp_wrap_classes: tuple[type, ...] = (
            Qwen3VLTextDecoderLayerWithKV,
            DiTQwen3DecoderLayer,
            Qwen3VLVisionBlock,
        )
        # Master params stay fp32 before wrapping; MixedPrecisionPolicy
        # casts to bf16 only for compute.
        self.model.to(device=device, dtype=torch.float32)
        apply_fsdp2(
            model=self.model,
            wrap_classes=fsdp_wrap_classes,
            mesh=ctx.mesh,
            reshard_after_forward=cfg.training.fsdp.reshard_after_forward,
            mp_policy=build_mixed_precision_policy(cfg.training.use_bf16),
            enable_prefetch=cfg.training.fsdp.enable_prefetch,
        )

        # Param groups (after fully_shard) + optimizer + scheduler.
        all_trainable_parameters, self.vlm_group_indices, self.vlm_param_refs = build_param_groups(
            model,
            cfg.optimizer,
            train_vlm=cfg.training.train_vlm,
        )

        self.optimizer = torch.optim.AdamW(all_trainable_parameters, fused=True)
        self.lr_scheduler = build_lr_scheduler(
            optimizer=self.optimizer,
            vlm_group_indices=self.vlm_group_indices,
            **self.lr_schedule_kwargs,
        )

        # Resume BEFORE compile so state_dict keys don't carry the _orig_mod prefix.
        if cfg.training.resume_checkpoint_path:
            if rank == 0:
                print(f"[ckpt] resume: loading full workspace state from {cfg.training.resume_checkpoint_path}")
            # Same as load_checkpoint(): DCP .metadata pickle compat (e.g. Py3.13 pathlib on Py3.10 workers).
            enable_pathlib_local_pickle_compat()
            app_state = FSDPWorkspaceAppState(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                training_state=self.training_state,
            )
            dcp.load({APP_STATE_KEY: app_state}, checkpoint_id=cfg.training.resume_checkpoint_path)
            self.update_step = self.training_state.update_step
            self.global_step = self.training_state.global_step
            self.epoch = self.training_state.epoch
            if rank == 0:
                print(
                    f"[ckpt] resume: restored update_step={self.update_step} "
                    f"global_step={self.global_step} epoch={self.epoch}"
                )

        self.maybe_compile_model(rank)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_eval_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.eval_every = 1

        return train_dataloader, val_dataloader, micro_batches_per_epoch, topk_manager

    def train_step(self, batch, batch_idx, grad_accum_steps, cfg, rank):
        """Forward / backward / grad-clip / optimizer-step for one micro-batch.

        Returns (sync_gradients, step_skipped, raw_loss, part_grad_norms).
        """
        if batch_idx == 10 and rank == 0 and cfg.training.profile:
            self.tracker.track()

        is_accumulating = (batch_idx + 1) % grad_accum_steps != 0
        sync_gradients = not is_accumulating

        # FSDP MixedPrecisionPolicy handles bf16 cast of floating-point
        # inputs (cast_forward_inputs=True); int/bool/uint8 pass through.
        raw_loss = self.model("train", batch)
        loss = raw_loss["total_loss"]
        if grad_accum_steps > 1:
            loss = loss / grad_accum_steps
        loss.backward()

        if grad_accum_steps > 1:
            for k, v in raw_loss.items():
                if isinstance(v, torch.Tensor):
                    self._cycle_loss_accum[k] = (
                        self._cycle_loss_accum.get(k, 0) + v.detach() / grad_accum_steps
                    )
            if sync_gradients:
                raw_loss = self._cycle_loss_accum
                self._cycle_loss_accum = {}


        if self.vlm_param_refs and self.is_vlm_freeze_active():
            for param in self.vlm_param_refs:
                param.grad = None

        if batch_idx == 10 and rank == 0 and cfg.training.profile:
            torch.cuda.empty_cache()
            print(torch.cuda.memory_summary())
            self.tracker.report()
            self.tracker.stop()

        part_grad_norms = None
        step_skipped = False
        if sync_gradients:
            part_grad_norms, step_skipped = clip_and_check_grads(
                self.model,
                cfg.training.clipping,
                train_vlm=cfg.training.train_vlm,
                vlm_freeze_active=self.is_vlm_freeze_active(),
                rank=rank,
                update_step=self.update_step,
                global_step=self.global_step,
            )

        if sync_gradients and not step_skipped:
            self.optimizer.step()
            self.lr_scheduler.step()
        if sync_gradients:
            self.optimizer.zero_grad(set_to_none=True)

        return sync_gradients, step_skipped, raw_loss, part_grad_norms

    def maybe_log_and_ckpt(
        self, cfg, ctx, batch, raw_loss, part_grad_norms,
        step_perf_start, data_wait_sec, training_start_time,
        total_samples_processed, log_interval,
        topk_manager, val_dataloader,
    ):
        """Bump counters and run eval / ckpt / wandb.log on update-step boundaries."""
        rank, device = ctx.rank, ctx.device

        should_record = self.update_step % log_interval == 0
        self.update_step += 1

        should_eval = val_dataloader is not None and self.update_step % cfg.training.eval_every == 0
        should_ckpt = self.update_step % cfg.training.checkpoint_every == 0
        should_interval_ckpt = self.update_step % cfg.training.ckpt_save_interval == 0

        step_log = None
        if should_record or should_eval or should_ckpt or should_interval_ckpt:
            step_log = build_training_step_log(
                self,
                include_full_metrics=should_record,
                raw_loss=raw_loss,
                part_grad_norms=part_grad_norms,
                batch=batch,
                step_perf_start=step_perf_start,
                data_wait_sec=data_wait_sec,
                training_start_time=training_start_time,
                total_samples_processed=total_samples_processed,
                train_vlm=cfg.training.train_vlm,
            )

        if should_eval:
            evaluation(self, rank, device, val_dataloader, step_log)
        if should_ckpt:
            save_topk_ckpt(self, rank, topk_manager, step_log)
        if should_interval_ckpt:
            save_interval_ckpt(self, rank)

        if step_log is not None and rank == 0:
            wandb.log(step_log, step=self.update_step)
        return step_log

    def train_loop(self, cfg, ctx, profiler, train_dataloader, val_dataloader, micro_batches_per_epoch, topk_manager):
        rank = ctx.rank
        grad_accum_steps = int(cfg.training.get("gradient_accumulation_steps", 1))
        log_interval = int(getattr(cfg.training, "log_interval", 50))
        profile_context = profiler if (cfg.training.profile and rank == 0) else nullcontext()

        gc_handler = GarbageCollection(gc_freq=100, full_gc_freq=2000)
        training_start_time = None
        total_samples_processed = 0

        with profile_context as prof:
            if rank == 0:
                print(
                    f"Training with {micro_batches_per_epoch // grad_accum_steps} update steps/epoch "
                    f"({micro_batches_per_epoch} micro-batches, grad_accum={grad_accum_steps}) "
                    f"(WebDataset streaming)"
                )
            for _epoch_idx in range(self.epoch, cfg.training.num_epochs):
                self.model.train()
                if rank == 0:
                    print(f"Training epoch {self.epoch} started")
                step_perf_end = time.perf_counter()
                for batch_idx, batch in enumerate(train_dataloader):
                    data_wait_sec = time.perf_counter() - step_perf_end
                    if batch_idx >= micro_batches_per_epoch:
                        break

                    step_perf_start = time.perf_counter()
                    if training_start_time is None:
                        training_start_time = time.time()
                    if cfg.training.profile and torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    sync_gradients, step_skipped, raw_loss, part_grad_norms = self.train_step(
                        batch, batch_idx, grad_accum_steps, cfg, rank,
                    )

                    if step_skipped:
                        step_perf_end = time.perf_counter()
                        continue

                    # Count every successful microbatch: per-microbatch for
                    # global_step / sample throughput, per-update for update_step.
                    self.global_step += 1
                    total_samples_processed += batch["input_ids"].shape[0]

                    if not sync_gradients:
                        step_perf_end = time.perf_counter()
                        continue

                    self.maybe_log_and_ckpt(
                        cfg, ctx, batch, raw_loss, part_grad_norms,
                        step_perf_start, data_wait_sec, training_start_time,
                        total_samples_processed, log_interval,
                        topk_manager, val_dataloader,
                    )

                    if cfg.training.max_train_steps and self.update_step >= cfg.training.max_train_steps:
                        if rank == 0:
                            print(f"Max train steps {cfg.training.max_train_steps} reached, stopping training.")
                        break

                    if self.global_step % 100 == 0 and rank == 0:
                        if grad_accum_steps > 1:
                            print(
                                f"Global step {self.global_step} "
                                f"(update_step {self.update_step}) completed"
                            )
                        else:
                            print(f"Global step {self.global_step} completed")
                    if self.global_step % 500 == 0 and rank == 0:
                        _print_worker_rss(self.global_step)

                    if cfg.training.profile and rank == 0:
                        prof.step()

                    gc_handler.run(self.global_step)
                    step_perf_end = time.perf_counter()

                # Epoch-end defense: drop any residual grads accumulated past
                # the last sync step (e.g., if the dataloader exhausted before
                if grad_accum_steps > 1:
                    self.optimizer.zero_grad(set_to_none=True)
                    self._cycle_loss_accum = {}

                if cfg.training.max_train_steps and self.update_step >= cfg.training.max_train_steps:
                    break
                self.epoch += 1

        gc_handler.finalize()


def _print_worker_rss(global_step: int) -> None:
    """Log main process + top worker RSS to diagnose dataloader leaks."""
    import psutil
    proc = psutil.Process()
    children = proc.children(recursive=True)
    worker_rss = sorted(
        ((c.pid, c.memory_info().rss / 1e9) for c in children),
        key=lambda x: -x[1],
    )
    print(f"[Step {global_step}] Main RSS: {proc.memory_info().rss / 1e9:.2f}GB")
    for pid, rss in worker_rss[:6]:
        print(f"  Worker PID {pid}: {rss:.2f}GB")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainEgoSteerWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
