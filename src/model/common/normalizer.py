from typing import Union, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from .dict_of_tensor_mixin import DictOfTensorMixin

class StreamingStats:
    """
    class for streaming statistics calculation, supporting dynamic data addition and updating statistics
    """

    def __init__(self, last_n_dims=1, reservoir_size=1000000):
        self.last_n_dims = last_n_dims
        self.reservoir_size = reservoir_size
        self.n_samples = 0
        self.sum = None
        self.sum_sq = None
        self.min_vals = None
        self.max_vals = None
        self.dim = None
        self.reservoir = None
        self.reservoir_count = 0

    def update(self, data: Union[torch.Tensor, np.ndarray]):
        """
        add new data and update statistics
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        if self.dim is None:
            if self.last_n_dims > 0:
                self.dim = int(torch.tensor(data.shape[-self.last_n_dims:]).prod().item())
            else:
                self.dim = 1
            data = data.reshape(-1, self.dim)

            self.sum = torch.zeros(self.dim, dtype=torch.float64)
            self.sum_sq = torch.zeros(self.dim, dtype=torch.float64)
            self.min_vals = torch.full((self.dim,), float('inf'), dtype=torch.float64)
            self.max_vals = torch.full((self.dim,), float('-inf'), dtype=torch.float64)
            self.reservoir = torch.empty(self.reservoir_size, self.dim, dtype=torch.float64)
        else:
            data = data.reshape(-1, self.dim)

        data = data.to(torch.float64)
        n_new = data.shape[0]

        # reservoir sampling (batch version)
        if self.reservoir_count < self.reservoir_size:
            space = self.reservoir_size - self.reservoir_count
            direct = min(space, n_new)
            self.reservoir[self.reservoir_count:self.reservoir_count + direct] = data[:direct]
            self.reservoir_count += direct
            remaining = data[direct:]
        else:
            remaining = data

        if remaining.shape[0] > 0:
            n_remaining = remaining.shape[0]
            base = self.n_samples + (n_new - n_remaining)
            # generate random indices for the entire batch
            sample_indices = base + torch.arange(n_remaining)
            rand_j = (torch.rand(n_remaining) * (sample_indices + 1).float()).long()
            # only keep samples that map into the reservoir
            mask = rand_j < self.reservoir_size
            if mask.any():
                self.reservoir[rand_j[mask]] = remaining[mask]

        self.n_samples += n_new

        self.sum += data.sum(dim=0)
        self.sum_sq += (data ** 2).sum(dim=0)

        self.min_vals = torch.minimum(self.min_vals, data.min(dim=0).values)
        self.max_vals = torch.maximum(self.max_vals, data.max(dim=0).values)

    def get_stats(self):
        """
        get current statistics including quantiles from reservoir
        """
        if self.n_samples == 0:
            return None

        mean = self.sum / self.n_samples
        if self.n_samples > 1:
            variance = (self.sum_sq - self.n_samples * mean ** 2) / (self.n_samples - 1)
            std = torch.sqrt(torch.clamp(variance, min=0))
        else:
            std = torch.zeros_like(mean)

        # compute quantiles from reservoir
        valid_reservoir = self.reservoir[:self.reservoir_count]
        q01 = torch.quantile(valid_reservoir, 0.01, dim=0)
        q99 = torch.quantile(valid_reservoir, 0.99, dim=0)

        return {
            'min': self.min_vals.clone(),
            'max': self.max_vals.clone(),
            'mean': mean,
            'std': std,
            'q01': q01,
            'q99': q99,
            'n_samples': self.n_samples
        }

    def reset(self):
        """
        reset statistics
        """
        self.n_samples = 0
        self.sum = None
        self.sum_sq = None
        self.min_vals = None
        self.max_vals = None
        self.dim = None
        self.reservoir = None
        self.reservoir_count = 0


class LinearNormalizer(DictOfTensorMixin):
    
    def __init__(self):
        super().__init__()
        self.streaming_stats = {}
    
    def start_streaming_fit(self,
                           keys: Optional[list] = None,
                           last_n_dims=1,
                           dtype=torch.float32,
                           mode='limits',
                           output_max=1.,
                           output_min=-1.,
                           range_eps=1e-4,
                           fit_offset=True,
                           reservoir_size=1000000):
        """
        start streaming fit, initialize streaming statistics
        """
        self.streaming_config = {
            'last_n_dims': last_n_dims,
            'dtype': dtype,
            'mode': mode,
            'output_max': output_max,
            'output_min': output_min,
            'range_eps': range_eps,
            'fit_offset': fit_offset
        }
        
        if keys is None:
            keys = ['_default']
        
        for key in keys:
            self.streaming_stats[key] = StreamingStats(
                last_n_dims=last_n_dims, reservoir_size=reservoir_size
            )
    
    def update_streaming_fit(self, data: Union[Dict, torch.Tensor, np.ndarray]):
        """
        update streaming statistics
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if key in self.streaming_stats:
                    self.streaming_stats[key].update(value)
        else:
            if '_default' in self.streaming_stats:
                self.streaming_stats['_default'].update(data)
    
    def finish_streaming_fit(self):
        """
        finish streaming fit, calculate final normalizer parameters
        """
        if not hasattr(self, 'streaming_config'):
            raise RuntimeError("Must call start_streaming_fit first")
        
        config = self.streaming_config
        
        for key, stats in self.streaming_stats.items():
            stats_dict = stats.get_stats()
            if stats_dict is None:
                continue
                
            self.params_dict[key] = _fit_from_stats(
                stats_dict,
                last_n_dims=config['last_n_dims'],
                dtype=config['dtype'],
                mode=config['mode'],
                output_max=config['output_max'],
                output_min=config['output_min'],
                range_eps=config['range_eps'],
                fit_offset=config['fit_offset']
            )
        
        self.streaming_stats = {}
        delattr(self, 'streaming_config')

    def ignore_dim(self, key: str, dim: slice):
        """
        ignore some dimensions when normalizing, e.g. the wrist rotation
        """
        if key not in self.params_dict:
            raise RuntimeError(f"Not initialized with key: {key}")
        params = self.params_dict[key]
        params['scale'][dim] = 1.0
        params['offset'][dim] = 0.0

        ignored_dim_mask = params.get('ignored_dim_mask')
        if ignored_dim_mask is None:
            ignored_dim_mask = torch.zeros_like(params['scale'])
            params['ignored_dim_mask'] = nn.Parameter(ignored_dim_mask, requires_grad=False)
        params['ignored_dim_mask'][dim] = 1.0
    
    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> Union[Dict, torch.Tensor]:
        return self.normalize(x)
    
    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                if key not in self.params_dict:
                    raise RuntimeError(f"Not initialized with key: {key}")
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> Union[Dict, torch.Tensor]:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> Union[Dict, torch.Tensor]:
        return self._normalize_impl(x, forward=False)

class SingleFieldLinearNormalizer(DictOfTensorMixin):
    
    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)


def _fit_from_stats(stats_dict: Dict,
                   last_n_dims=1,
                   dtype=torch.float32,
                   mode='limits',
                   output_max=1.,
                   output_min=-1.,
                   range_eps=1e-4,
                   fit_offset=True):
    """
    calculate normalizer parameters from statistics dictionary
    """
    assert mode in ['limits', 'gaussian']
    assert last_n_dims >= 0
    assert output_max > output_min

    # extract data from statistics dictionary and convert to torch
    def to_float32(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(torch.float32)
        return x.to(torch.float32)

    input_min = to_float32(stats_dict['min'])
    input_max = to_float32(stats_dict['max'])
    input_mean = to_float32(stats_dict['mean'])
    input_std = to_float32(stats_dict['std'])

    # quantiles for clipping (optional, from reservoir sampling)
    has_quantiles = 'q01' in stats_dict and 'q99' in stats_dict
    if has_quantiles:
        input_q01 = to_float32(stats_dict['q01'])
        input_q99 = to_float32(stats_dict['q99'])

    # compute scale and offset
    # use q01/q99 when available for robustness to outliers, fallback to min/max
    lo = input_q01 if has_quantiles else input_min
    hi = input_q99 if has_quantiles else input_max

    if mode == 'limits':
        if fit_offset:
            # unit scale based on q01/q99
            input_range = hi - lo
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * lo
            offset[ignore_dim] = (output_max + output_min) / 2 - lo[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(lo), torch.abs(hi))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == 'gaussian':
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = - input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)
    
    # save
    input_stats_dict = {
        'min': input_min,
        'max': input_max,
        'mean': input_mean,
        'std': input_std
    }
    if has_quantiles:
        input_stats_dict['q01'] = input_q01
        input_stats_dict['q99'] = input_q99

    this_params = nn.ParameterDict({
        'scale': scale,
        'offset': offset,
        'input_stats': nn.ParameterDict(input_stats_dict)
    })
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward=True):
    assert 'scale' in params
    is_numpy = isinstance(x, np.ndarray)
    scale = params['scale']
    offset = params['offset']

    if is_numpy:
        scale = scale.cpu().numpy()
        offset = offset.cpu().numpy()
    else:
        scale = scale.to(x.device)
        offset = offset.to(x.device)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x
