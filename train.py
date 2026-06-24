import os
import sys
import resource

# Raise MEMLOCK to hard cap so NCCL/RDMA pinned memory registration is not
# capped at the 64MB Linux default. No-op if hard cap is already low.
_, _memlock_hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
try:
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (_memlock_hard, _memlock_hard))
except (ValueError, OSError):
    pass

# Avoid fd-based shm EAGAIN under heavy DataLoader prefetch; leaks tmpfiles
# in TMPDIR on ungraceful crash, sweep before launch.
# https://pytorch.org/docs/stable/multiprocessing.html#sharing-strategies
import torch.multiprocessing as _torch_mp
_torch_mp.set_sharing_strategy("file_system")

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from src.workspace.base_workspace import BaseWorkspace
import torch

torch.set_float32_matmul_precision('high')

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'src','config')), 
    config_name="train_config"
)
def main(cfg: OmegaConf):
    # Allow raising torch.compile recompile limits via env (no native env
    # support in torch._dynamo.config). Must run before any torch.compile call.
    import torch._dynamo.config as _dynamo_cfg
    _dynamo_cfg.recompile_limit = int(32)
    _dynamo_cfg.accumulated_recompile_limit = int(1024)

    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()