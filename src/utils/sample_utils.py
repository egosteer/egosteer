"""Centralized sampling primitives for training and inference.

All stochastic sampling operations (flow time, geometric spans, biased
positions, RTC delays) are collected here so that each behaviour has a
single authoritative implementation.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Flow-matching time sampling
# ---------------------------------------------------------------------------

def sample_flow_time(
    batch_size: int,
    num_samples: int = 1,
    *,
    sampling: str = "beta",
    alpha: float = 1.5,
    beta: float = 1.0,
    sig_min: float = 0.001,
) -> torch.FloatTensor:
    """Sample flow-matching time steps.

    Supports two strategies:
      - "uniform": stratified sampling — batch elements evenly spaced across
        [0, 1), each column shifted by a shared random offset.
      - "beta": Beta(alpha, beta) distribution, mapped to t via
        t = (1 - sig_min) * (1 - z).

    Args:
        batch_size: Number of samples (B).
        num_samples: Number of independent time draws per sample (T).
        sampling: "uniform" or "beta".
        alpha, beta: Shape parameters for Beta distribution.
        sig_min: Minimum noise level (only used in "beta" mode).

    Returns:
        [B] when num_samples == 1, else [B, T].
    """
    if sampling == "uniform":
        eps = 1e-5
        ranks = torch.arange(batch_size, dtype=torch.float32) / batch_size
        offsets = torch.rand(num_samples, dtype=torch.float32)
        t = (ranks.unsqueeze(1) + offsets.unsqueeze(0)) % (1 - eps)  # [B, T]
        return t.squeeze(1) if num_samples == 1 else t

    if sampling == "beta":
        beta_dist = torch.distributions.Beta(alpha, beta)
        z = beta_dist.sample((batch_size, num_samples))
        t = (1 - sig_min) * (1 - z)
        return t.squeeze(1) if num_samples == 1 else t

    raise ValueError(f"Unsupported flow time sampling strategy: {sampling}")


# ---------------------------------------------------------------------------
# Geometric distribution (inverse-CDF)
# ---------------------------------------------------------------------------

def sample_geometric(
    shape: tuple[int, ...],
    mean: float,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Sample from a geometric distribution via inverse CDF.

    P(k) = (1-p)^{k-1} * p  with  p = 1/mean,  so E[k] = mean.
    Implemented as k = ceil(log(U) / log(1-p)) for U ~ Uniform(0, 1).

    Args:
        shape: Output tensor shape.
        mean: Mean of the geometric distribution (must be >= 1).
        device: Target device.

    Returns:
        LongTensor of the given shape, values >= 1.
    """
    assert mean >= 1.0, f"sample_geometric requires mean >= 1.0, got {mean}"
    p = 1.0 / mean
    u = torch.rand(shape, device=device).clamp_(1e-7, 1 - 1e-7)
    # Matches PyTorch Geometric.sample(); log1p(-p) is more stable than log(1-p)
    return torch.ceil(u.log() / torch.tensor(-p, device=device).log1p()).long()


# ---------------------------------------------------------------------------
# Beta-biased discrete position sampling
# ---------------------------------------------------------------------------

def sample_beta_positions(
    batch_size: int,
    num_samples: int,
    seq_len: int,
    alpha: float,
    prefer_early: bool = True,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Sample discrete positions from a Beta-biased distribution.

    - prefer_early=True  → Beta(1, alpha), density concentrated near 0.
    - prefer_early=False → Beta(alpha, 1), density concentrated near seq_len-1.

    The continuous Beta sample is scaled to [0, seq_len) then truncated to long.

    Args:
        batch_size: B.
        num_samples: Number of positions per sample.
        seq_len: Length of the discrete sequence to index into.
        alpha: Shape parameter controlling bias strength.
        prefer_early: Direction of the bias.
        device: Target device.

    Returns:
        [B, num_samples] LongTensor in [0, seq_len).
    """
    if prefer_early:
        beta_dist = torch.distributions.Beta(1.0, alpha)
    else:
        beta_dist = torch.distributions.Beta(alpha, 1.0)
    positions = (beta_dist.sample((batch_size, num_samples)).to(device) * seq_len).long()
    positions.clamp_(max=seq_len - 1)
    return positions


# ---------------------------------------------------------------------------
# RTC prefix delay sampling
# ---------------------------------------------------------------------------

def sample_rtc_delay(
    valid_action_len: torch.LongTensor,
    strategy: str = "uniform",
    max_delay: int | None = None,
    forced_delay: torch.LongTensor | None = None,
) -> torch.LongTensor:
    """Sample one RTC prefix delay per batch element.

    The sampled delay determines how many leading action tokens are treated as
    known action-prefix conditions during RTC flow training.

    Args:
        valid_action_len:
            [B] Number of valid action steps for each sample. Values are expected
            to be in the range [0, horizon_steps].
        strategy:
            Delay sampling strategy. ``uniform`` samples all valid delays with
            equal probability. ``exp`` biases toward smaller delays via
            ``exp(arange(upper)[::-1])``.
        max_delay:
            Optional upper bound for the sampled delay. When provided, the actual
            sampling upper bound becomes min(valid_action_len, max_delay) for each
            sample.
        forced_delay:
            Optional [B] tensor used to bypass random sampling. This is intended
            for deterministic tests and debugging.

    Returns:
        torch.LongTensor:
            [B] Sampled prefix delays. Each entry is in the range
            [0, min(valid_action_len_i, max_delay)) when the upper bound is
            positive, or 0 when the sample has no valid action tokens.
    """
    if forced_delay is not None:
        return forced_delay.to(device=valid_action_len.device, dtype=torch.long)

    delay_upper = valid_action_len.clamp(min=0)
    if max_delay is not None:
        delay_upper = torch.minimum(
            delay_upper,
            torch.full_like(delay_upper, max_delay),
        )
    delay_upper = delay_upper.clamp(min=0)
    strategy = str(strategy).lower()

    if strategy == "uniform":
        random_delay = torch.rand(delay_upper.shape, device=valid_action_len.device, dtype=torch.float32)
        return torch.floor(random_delay * delay_upper.to(torch.float32)).to(dtype=torch.long)

    if strategy == "exp":
        device = valid_action_len.device
        out = torch.zeros_like(delay_upper, device=device, dtype=torch.long)
        # Only VLA samples (delay_upper > 0) participate in multinomial sampling.
        # VLM padding samples have delay_upper == 0 and keep the default 0, both
        # avoiding the all-zero-row CUDA assert in torch.multinomial and skipping
        # useless work for samples whose flow loss is masked out anyway.
        active = delay_upper > 0
        if not active.any():
            return out
        active_upper = delay_upper[active]
        max_upper = int(active_upper.max().item())
        w = torch.exp(
            torch.arange(max_upper - 1, -1, -1, device=device, dtype=torch.float32)
        )
        w = w.unsqueeze(0).expand(active_upper.shape[0], -1).clone()
        idx = torch.arange(max_upper, device=device).unsqueeze(0)
        w[idx >= active_upper.unsqueeze(1)] = 0.0
        sampled = torch.multinomial(w, num_samples=1).squeeze(1)
        out[active] = sampled.to(dtype=torch.long)
        return out

    raise ValueError(f"Unsupported RTC delay strategy: {strategy}")


# ---------------------------------------------------------------------------
# Multi-span mask generation
# ---------------------------------------------------------------------------

def generate_bernoulli_mask(
    batch_size: int,
    seq_len: int,
    config,
    *,
    device: torch.device = torch.device("cpu"),
) -> torch.BoolTensor:
    """Per-frame independent Bernoulli mask for short state histories.

    Reads ``mask_prob``, ``keep_last``, ``p_no_mask`` from ``config``. Each frame
    is masked independently with probability ``mask_prob``; the last frame
    (current state) is kept when ``keep_last``; a fraction ``p_no_mask`` of
    samples are left fully unmasked.

    Args:
        batch_size: Number of samples in the batch.
        seq_len: Sequence length (H_state).
        config: A StateMaskConfig (or duck-typed equivalent).
        device: Target device for the output tensor.

    Returns:
        [B, seq_len] BoolTensor where True = masked position.
    """
    if config.mask_prob <= 0 or seq_len <= 0:
        return torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    mask = torch.rand(batch_size, seq_len, device=device) < config.mask_prob
    if config.keep_last:
        mask[:, -1] = False
    if config.p_no_mask > 0:
        unmasked = torch.rand(batch_size, device=device) < config.p_no_mask
        mask[unmasked] = False
    return mask


def generate_multi_span_mask(
    batch_size: int,
    seq_len: int,
    config,
    *,
    device: torch.device = torch.device("cpu"),
) -> torch.BoolTensor:
    """Generate multi-span boolean masks for input-side corruption.

    Reads mask parameters directly from ``config`` attributes:
    ``mask_ratio``, ``mean_span_len``, ``start_bias_alpha``,
    ``p_no_mask``, ``keep_last``, ``prefer_early``.

    Args:
        batch_size: Number of samples in the batch.
        seq_len: Total sequence length (H_action or H_state).
        config: A SpanMaskConfig (or duck-typed equivalent).
        device: Target device for the output tensor.

    Returns:
        [B, seq_len] BoolTensor where True = masked position.
    """
    mask_ratio = config.mask_ratio
    mean_span_len = config.mean_span_len
    start_bias_alpha = config.start_bias_alpha
    p_no_mask = config.p_no_mask
    keep_last = config.keep_last
    prefer_early = config.prefer_early

    effective_len = seq_len - 1 if keep_last else seq_len
    if effective_len <= 0 or mask_ratio <= 0:
        return torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    expected_n_spans = max(1.0, mask_ratio * effective_len / mean_span_len)

    # Per-element span count from Poisson, so n_spans varies across batch
    n_spans_per_elem = torch.poisson(
        torch.full((batch_size,), expected_n_spans, device=device)
    ).long().clamp_(min=0, max=effective_len)
    max_n_spans = max(1, int(n_spans_per_elem.max().item()))

    # Biased start positions and geometric span lengths (padded to max_n_spans)
    starts = sample_beta_positions(
        batch_size, max_n_spans, effective_len, start_bias_alpha,
        prefer_early=prefer_early, device=device,
    )  # [B, max_n_spans]
    span_lengths = sample_geometric(
        (batch_size, max_n_spans), mean_span_len, device=device,
    ).clamp_(min=1, max=effective_len)  # [B, max_n_spans]

    # Mask out inactive span slots per element
    active = torch.arange(max_n_spans, device=device).unsqueeze(0) < n_spans_per_elem.unsqueeze(1)

    # Expand spans via 3D offset grid and scatter into output mask
    max_span = min(int(span_lengths.max().item()), effective_len)
    offsets = torch.arange(max_span, device=device).view(1, 1, -1)
    abs_pos = starts.unsqueeze(-1) + offsets  # [B, max_n_spans, max_span]
    valid = (abs_pos < effective_len) & (offsets < span_lengths.unsqueeze(-1)) & active.unsqueeze(-1)
    abs_pos = abs_pos.clamp(0, effective_len - 1)

    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    mask.scatter_(1, abs_pos.reshape(batch_size, -1), valid.reshape(batch_size, -1))

    # Some samples get no masking at all
    no_mask_selector = torch.rand(batch_size, device=device) < p_no_mask
    mask[no_mask_selector] = False

    if keep_last:
        mask[:, -1] = False

    return mask
