import logging
import pathlib
import socket
from typing import Any

from omegaconf import OmegaConf

from src.serving.serving_recorder import ServingRecorder
from src.serving.websocket_policy_server import (
    WebsocketPolicyServer,
    create_engine,
    create_env_wrapper,
)


OmegaConf.register_new_resolver("eval", eval, replace=True)
logger = logging.getLogger(__name__)


def _load_config() -> OmegaConf:
    config_path = pathlib.Path(__file__).resolve().parents[1] / "config" / "experiment" / "inference.yaml"
    logger.info("Loading inference config from: %s", config_path)
    cfg = OmegaConf.load(config_path)
    assert "serving" in cfg and "policy" in cfg and "env_wrapper" in cfg, (
        "Missing policy/serving/env_wrapper config after config load"
    )
    return cfg


def _create_recorder(serving_cfg: OmegaConf, wrapper_cfg: OmegaConf | None = None) -> ServingRecorder | None:
    if not bool(serving_cfg.record_enabled):
        logger.info("Serving recorder is disabled")
        return None

    record_root_dir = serving_cfg.record_root_dir
    if not record_root_dir:
        return None

    root_dir = pathlib.Path(record_root_dir).expanduser()
    if not root_dir.is_absolute():
        project_root = pathlib.Path(__file__).resolve().parents[2]
        root_dir = project_root / root_dir
    logger.info("Recording serving requests to %s", root_dir)

    image_key = getattr(wrapper_cfg, "image_key", "image")
    depth_key = getattr(wrapper_cfg, "depth_key", "depth_image")
    return ServingRecorder(root_dir, image_key=image_key, depth_key=depth_key)


def _warmup_policy(policy: Any, serving_cfg: OmegaConf) -> None:
    if not serving_cfg.warmup_enabled:
        return

    warmup_iters = serving_cfg.warmup_iters
    warmup_instruction = serving_cfg.warmup_instruction
    logger.info("Starting policy warmup")
    policy.warmup(warmup_iters=warmup_iters, instruction=warmup_instruction)


def _enable_profiler(policy: Any, serving_cfg: OmegaConf) -> None:
    if not serving_cfg.profile_enabled:
        return

    profile_dir = pathlib.Path(serving_cfg.profile_dir).expanduser()
    if not profile_dir.is_absolute():
        project_root = pathlib.Path(__file__).resolve().parents[2]
        profile_dir = project_root / profile_dir
    policy.enable_profiling(
        profile_dir,
        int(serving_cfg.profile_steps),
        int(serving_cfg.profile_skip_first),
        count_flops=bool(getattr(serving_cfg, "flops_count_enabled", False)),
    )


def main() -> None:
    try:
        cfg = _load_config()
        logger.info("Initializing policy engine...")
        policy = create_engine(cfg.policy, cfg.serving)
        _warmup_policy(policy, cfg.serving)
        _enable_profiler(policy, cfg.serving)
        wrapper_cfg = None
        if getattr(cfg, "env_wrapper", None) and cfg.env_wrapper.enabled:
            wrapper_cfg = cfg.env_wrapper
            logger.info("Enabling env wrapper: %s", wrapper_cfg)
            policy = create_env_wrapper(policy, wrapper_cfg)
        policy_metadata = policy.metadata

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logger.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

        server = WebsocketPolicyServer(
            policy=policy,
            host=cfg.serving.host,
            port=cfg.serving.port,
            metadata=policy_metadata,
            recorder=_create_recorder(cfg.serving, wrapper_cfg),
            log_obs_details=bool(cfg.serving.log_obs_details),
        )
        logger.info("Serving websocket policy on %s:%s", cfg.serving.host, cfg.serving.port)
        server.serve_forever()
    except Exception:
        logger.exception("Policy server failed during startup.")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()