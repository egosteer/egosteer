from typing import Optional
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


class BaseWorkspace:
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        self.cfg = cfg
        self._output_dir = output_dir

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def run(self):
        """
        Create any resource that should not be serialized as local variables.
        """
        pass
