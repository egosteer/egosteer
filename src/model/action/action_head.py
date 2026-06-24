import torch
import torch.nn as nn
from typing import Optional


class MLPEncoder(nn.Module):
    """MLP encoder for continuous state/action inputs."""

    def __init__(
        self,
        action_dim: int,
        width: int,
        time_cond: bool = False,
        mlp_depth: int = 2,
        final_layer_norm: bool = True,
        time_emb_dim: Optional[int] = None,
        use_mlp_layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        assert mlp_depth > 0, "mlp_depth must be > 0"
        self.time_cond = time_cond
        if self.time_cond:
            if time_emb_dim is None:
                raise ValueError("time_emb_dim must be provided when time_cond=True")
        else:
            time_emb_dim = 0

        mlp_input_dim = action_dim + time_emb_dim
        if mlp_depth == 1: 
            self.mlp = nn.Identity()
            self.projector = nn.Linear(mlp_input_dim, width)
        else:
            layers = []
            for layer_idx in range(mlp_depth - 1):
                in_dim = mlp_input_dim if layer_idx == 0 else width
                layers.append(nn.Linear(in_dim, width))
                if use_mlp_layer_norm:
                    layers.append(nn.LayerNorm(width))
                layers.append(nn.SiLU())
            self.mlp = nn.Sequential(*layers)
            self.projector = nn.Linear(width, width)
            
        self.final_layer_norm = nn.LayerNorm(width) if final_layer_norm else None
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        self.projector.weight.data.normal_(mean=0.0, std=0.02)
        self.projector.bias.data.zero_()

    def forward(
        self,
        action: torch.FloatTensor,
        time_emb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            action: [Batch_Size, Seq_Len, Action_Dim]
            time_emb: [Batch_Size, Time_Dim]
        Returns:
            emb: [Batch_Size, Seq_Len, Width]
        """
        emb = action
        if self.time_cond:
            if time_emb is None:
                raise ValueError("time_emb must be provided when time_cond=True")
            if time_emb.ndim == 2:
                time_emb_full = time_emb.unsqueeze(1).expand(-1, action.size(1), -1)
            else:
                time_emb_full = time_emb
            emb = torch.cat([time_emb_full, emb], dim=-1)
        emb = self.mlp(emb)
        emb = self.projector(emb)
        if self.final_layer_norm is not None:
            emb = self.final_layer_norm(emb)
        return emb


class MLPDecoder(nn.Module):
    """
    MLP decoder/projector with optional final LayerNorm and SiLU hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 1024,
        depth: int = 3,
        final_layer_norm: bool = True,
        use_mlp_layer_norm: bool = False,
    ):
        super().__init__()
        assert depth >= 0, "depth must be >= 0"
        if depth == 0:
            self.mlp = nn.Identity()
            self.final_layer_norm = None
        else:
            layers = []
            for layer_idx in range(depth):
                in_dim = input_dim if layer_idx == 0 else width
                out_dim = width if layer_idx < depth - 1 else output_dim
                layers.append(nn.Linear(in_dim, out_dim))
                if layer_idx < depth - 1:
                    if use_mlp_layer_norm:
                        layers.append(nn.LayerNorm(width))
                    layers.append(nn.SiLU())
            self.mlp = nn.Sequential(*layers)
            self.final_layer_norm = nn.LayerNorm(output_dim) if final_layer_norm else None
        self.initialize_weights()
        
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, latent: torch.FloatTensor) -> torch.FloatTensor:
        emb = self.mlp(latent)
        if self.final_layer_norm is not None:
            emb = self.final_layer_norm(emb)
        return emb

