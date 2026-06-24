from typing import List, Optional, Dict, Tuple
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


def get_smart_ticklabels(length: int, max_ticks: int = 8) -> List:
    """
    Generate smart tick labels that only show numbers at selected positions.
    
    Args:
        length: Total number of ticks
        max_ticks: Maximum number of ticks to show
    
    Returns:
        List of labels, where most are empty strings and only selected positions have numbers
    """
    if length <= max_ticks:
        return [i for i in range(length)]
    
    # Select evenly spaced indices
    tick_indices = np.linspace(0, length - 1, max_ticks, dtype=int)
    labels = [''] * length
    for idx in tick_indices:
        labels[idx] = str(idx)
    
    return labels


def get_action_accuracy(
    gt: torch.FloatTensor,  # [Batch_Size, Horizon, Action_Dim]
    pred: torch.FloatTensor,
    thresholds: List[float] = [0.1, 0.2],
    valid_mask: Optional[torch.Tensor] = None,  # [B,H,D] or [B,H]; padding excluded from denom
) -> torch.FloatTensor:
    device = gt.device
    assert gt.shape == pred.shape, "GT and pred must have the same shape"
    diff = torch.abs(gt - pred)  # [B, H, D]

    if valid_mask is not None:
        valid_step = valid_mask.any(dim=-1) if valid_mask.dim() == 3 else valid_mask
        denom = valid_step.sum().clamp(min=1).float()
    else:
        valid_step = None

    accuracies = torch.zeros(len(thresholds), device=device)
    for idx, threshold in enumerate(thresholds):
        per_step_ok = (diff < threshold).all(dim=-1)  # [B, H]
        if valid_step is not None:
            accuracies[idx] = (per_step_ok & valid_step).sum().float() / denom
        else:
            accuracies[idx] = per_step_ok.float().mean()
    return accuracies


def compute_smoothness_metrics(
    gt: torch.FloatTensor,  # [Batch_Size, Horizon, Action_Dim]
    pred: torch.FloatTensor,
) -> Dict[str, torch.FloatTensor]:
    """
    Compute smoothness metrics using first and second order differences.
    
    Returns:
        Dict containing:
            - gt_first_diff: First order difference for GT [Batch_Size, Horizon-1, Action_Dim]
            - pred_first_diff: First order difference for Pred
            - gt_second_diff: Second order difference for GT [Batch_Size, Horizon-2, Action_Dim]
            - pred_second_diff: Second order difference for Pred
            - first_diff_error: Global L1 error of first differences (scalar)
            - second_diff_error: Global L1 error of second differences (scalar)
            - first_diff_error_heatmap: L1 error heatmap [Horizon-1, Action_Dim] averaged over batch
            - second_diff_error_heatmap: L1 error heatmap [Horizon-2, Action_Dim] averaged over batch
    """
    assert gt.ndim == 3, "GT must have 3 dimensions"
    assert gt.shape == pred.shape, "GT and pred must have the same shape"
    
    # First order difference (velocity-like)
    gt_first_diff = gt[:, 1:, :] - gt[:, :-1, :]  # [B, H-1, D]
    pred_first_diff = pred[:, 1:, :] - pred[:, :-1, :]
    
    # Second order difference (acceleration-like)
    gt_second_diff = gt_first_diff[:, 1:, :] - gt_first_diff[:, :-1, :]  # [B, H-2, D]
    pred_second_diff = pred_first_diff[:, 1:, :] - pred_first_diff[:, :-1, :]
    
    # Compute errors (L1)
    first_diff_error = torch.mean(torch.abs(gt_first_diff - pred_first_diff))
    second_diff_error = torch.mean(torch.abs(gt_second_diff - pred_second_diff))
    
    # Compute error heatmaps: average over batch dimension
    first_diff_error_heatmap = torch.mean(torch.abs(gt_first_diff - pred_first_diff), dim=0)  # [H-1, D]
    second_diff_error_heatmap = torch.mean(torch.abs(gt_second_diff - pred_second_diff), dim=0)  # [H-2, D]
    
    return {
        'gt_first_diff': gt_first_diff,
        'pred_first_diff': pred_first_diff,
        'gt_second_diff': gt_second_diff,
        'pred_second_diff': pred_second_diff,
        'first_diff_error': first_diff_error,
        'second_diff_error': second_diff_error,
        'first_diff_error_heatmap': first_diff_error_heatmap,
        'second_diff_error_heatmap': second_diff_error_heatmap,
    }


def compute_error_heatmap(
    gt: torch.FloatTensor,  # [Batch_Size, Horizon, Action_Dim]
    pred: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Compute error heatmap: [Horizon, Action_Dim] showing mean absolute error.
    
    Returns:
        Error heatmap [Horizon, Action_Dim]
    """
    assert gt.ndim == 3, "GT must have 3 dimensions"
    assert gt.shape == pred.shape, "GT and pred must have the same shape"
    error = torch.abs(gt - pred)  # [B, H, D]
    error_heatmap = torch.mean(error, dim=0)  # [H, D]
    return error_heatmap


def compute_covariance_matrix(
    gt: torch.FloatTensor,  # [Batch_Size, Horizon, Action_Dim]
    pred: torch.FloatTensor,
) -> Dict[str, torch.FloatTensor]:
    """
    Compute covariance matrices for GT and Pred.
    
    Returns:
        Dict containing:
            - gt_cov: GT covariance matrix [Action_Dim, Action_Dim]
            - pred_cov: Pred covariance matrix [Action_Dim, Action_Dim]
            - error_cov: Error covariance matrix [Action_Dim, Action_Dim]
    """
    assert gt.ndim == 3, "GT must have 3 dimensions"
    assert gt.shape == pred.shape, "GT and pred must have the same shape"
    B, H, D = gt.shape
    
    # Flatten to [B*H, D]
    gt_flat = gt.reshape(-1, D)
    pred_flat = pred.reshape(-1, D)
    error_flat = (gt - pred).reshape(-1, D)
    
    # Compute covariance matrices
    gt_mean = torch.mean(gt_flat, dim=0, keepdim=True)
    pred_mean = torch.mean(pred_flat, dim=0, keepdim=True)
    error_mean = torch.mean(error_flat, dim=0, keepdim=True)
    
    gt_centered = gt_flat - gt_mean
    pred_centered = pred_flat - pred_mean
    error_centered = error_flat - error_mean
    
    gt_cov = torch.mm(gt_centered.t(), gt_centered) / (B * H - 1)
    pred_cov = torch.mm(pred_centered.t(), pred_centered) / (B * H - 1)
    error_cov = torch.mm(error_centered.t(), error_centered) / (B * H - 1)
    
    return {
        'gt_cov': gt_cov,
        'pred_cov': pred_cov,
        'error_cov': error_cov,
    }


def compute_loss_over_time(
    gt: torch.FloatTensor,  # [Batch_Size, Horizon, Action_Dim]
    pred: torch.FloatTensor,
    loss_type: str = 'l1',
) -> torch.FloatTensor:
    """
    Compute loss for each time step.
    
    Args:
        loss_type: 'l1' (default), 'l2', or 'mse'
    
    Returns:
        Loss per time step [Horizon]
    """
    assert gt.ndim == 3, "GT must have 3 dimensions"
    assert gt.shape == pred.shape, "GT and pred must have the same shape"
    
    if loss_type == 'l1':
        loss = torch.mean(torch.abs(gt - pred), dim=(0, 2))  # [H]
    elif loss_type == 'l2' or loss_type == 'mse':
        loss = torch.mean((gt - pred) ** 2, dim=(0, 2))  # [H]
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return loss


def compute_trajectory_metrics(
    gt: torch.FloatTensor,  # [Batch_Size, Horizon, Action_Dim]
    pred: torch.FloatTensor,
) -> Dict[str, torch.FloatTensor]:
    """
    Compute trajectory-level metrics.
    
    Returns:
        Dict containing:
            - endpoint_error: Error at the last time step [Batch_Size]
            - trajectory_length_error: Difference in trajectory lengths
            - mean_error: Mean error over trajectory [Batch_Size]
            - max_error: Maximum error over trajectory [Batch_Size]
    """
    assert gt.ndim == 3, "GT must have 3 dimensions."
    assert gt.shape == pred.shape, "GT and pred must have the same shape"
    B, H, D = gt.shape
    
    error = torch.abs(gt - pred)  # [B, H, D]
    
    # Endpoint error
    endpoint_error = torch.mean(error[:, -1, :], dim=1)  # [B]
    
    # Trajectory length (L2 norm of differences)
    gt_length = torch.sum(torch.norm(gt[:, 1:, :] - gt[:, :-1, :], dim=2), dim=1)  # [B]
    pred_length = torch.sum(torch.norm(pred[:, 1:, :] - pred[:, :-1, :], dim=2), dim=1)  # [B]
    trajectory_length_error = torch.abs(gt_length - pred_length)  # [B]
    
    # Mean and max error per trajectory
    mean_error = torch.mean(error.reshape(B, -1), dim=1)  # [B]
    max_error = torch.max(error.reshape(B, -1), dim=1)[0]  # [B]
    
    return {
        'endpoint_error': endpoint_error,
        'trajectory_length_error': trajectory_length_error,
        'mean_error': mean_error,
        'max_error': max_error,
    }


def compute_per_dimension_metrics(
    gt: torch.FloatTensor,  # [Batch_Size, Horizon, Action_Dim]
    pred: torch.FloatTensor,
) -> Dict[str, torch.FloatTensor]:
    """
    Compute metrics per action dimension.
    
    Returns:
        Dict containing:
            - mae_per_dim: Mean Absolute Error per dimension [Action_Dim]
    """
    assert gt.shape == pred.shape, "GT and pred must have the same shape"
    
    abs_error = torch.abs(gt - pred)  # [B, H, D]
    mae_per_dim = torch.mean(abs_error, dim=(0, 1))  # [D]
    
    return {
        'mae_per_dim': mae_per_dim,
    }


def plot_smoothness_analysis(
    gt: torch.FloatTensor,
    pred: torch.FloatTensor,
    save_path: Optional[str] = None,
    action_dim_names: Optional[List[str]] = None,
):
    """
    Plot smoothness analysis comparing GT and Pred first/second order differences,
    including error heatmaps showing where errors are largest.
    """
    smoothness = compute_smoothness_metrics(gt, pred)
    
    B, H, D = gt.shape
    if action_dim_names is None:
        action_dim_names = [f'Dim {i}' for i in range(D)]
    
    # Convert to numpy for plotting
    gt_first_diff = smoothness['gt_first_diff'].cpu().numpy()
    pred_first_diff = smoothness['pred_first_diff'].cpu().numpy()
    gt_second_diff = smoothness['gt_second_diff'].cpu().numpy()
    pred_second_diff = smoothness['pred_second_diff'].cpu().numpy()
    first_diff_error_heatmap = smoothness['first_diff_error_heatmap'].cpu().numpy()  # [H-1, D]
    second_diff_error_heatmap = smoothness['second_diff_error_heatmap'].cpu().numpy()  # [H-2, D]
    
    # Average over batch dimension
    gt_first_diff_mean = np.mean(gt_first_diff, axis=0)  # [H-1, D]
    pred_first_diff_mean = np.mean(pred_first_diff, axis=0)
    gt_second_diff_mean = np.mean(gt_second_diff, axis=0)  # [H-2, D]
    pred_second_diff_mean = np.mean(pred_second_diff, axis=0)
    
    # Define dimension ranges for averaging
    dim_ranges = [
        (0, 3, '[0, 3)'),
        (3, 6, '[3, 6)'),
        (6, 12, '[6, 12)'),
        (12, 18, '[12, 18)'),
        (18, 33, '[18, 33)'),
        (33, 48, '[33, 48)'),
    ]
    
    # Create figure with 4 subplots: 2 for line plots, 2 for heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot first order difference (velocity) - line plot
    ax1 = axes[0, 0]
    time_steps_1 = np.arange(H - 1)
    for start, end, label in dim_ranges:
        if start >= D:
            continue
        end = min(end, D)
        if start >= end:
            continue
        
        # Average over dimensions in this range
        gt_mean_segment = np.mean(gt_first_diff_mean[:, start:end], axis=1)  # [H-1]
        pred_mean_segment = np.mean(pred_first_diff_mean[:, start:end], axis=1)  # [H-1]
        
        ax1.plot(time_steps_1, gt_mean_segment, 
                label=f'GT {label}', linestyle='--', alpha=0.7, linewidth=2)
        ax1.plot(time_steps_1, pred_mean_segment, 
                label=f'Pred {label}', linestyle='-', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('First Order Difference (Velocity)')
    ax1.set_title('First Order Difference Comparison (Averaged by Dimension Ranges)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot first order difference error heatmap
    ax2 = axes[0, 1]
    sns.heatmap(
        first_diff_error_heatmap.T,  # Transpose to have dimensions on y-axis
        cmap='YlOrRd',
        xticklabels=get_smart_ticklabels(H - 1),
        yticklabels=get_smart_ticklabels(D),
        cbar_kws={'label': 'L1 Error'},
        ax=ax2,
    )
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Action Dimension')
    ax2.set_title('First Order Difference Error Heatmap (L1)')
    
    # Plot second order difference (acceleration) - line plot
    ax3 = axes[1, 0]
    time_steps_2 = np.arange(H - 2)
    for start, end, label in dim_ranges:
        if start >= D:
            continue
        end = min(end, D)
        if start >= end:
            continue
        
        # Average over dimensions in this range
        gt_mean_segment = np.mean(gt_second_diff_mean[:, start:end], axis=1)  # [H-2]
        pred_mean_segment = np.mean(pred_second_diff_mean[:, start:end], axis=1)  # [H-2]
        
        ax3.plot(time_steps_2, gt_mean_segment, 
                label=f'GT {label}', linestyle='--', alpha=0.7, linewidth=2)
        ax3.plot(time_steps_2, pred_mean_segment, 
                label=f'Pred {label}', linestyle='-', alpha=0.7, linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Second Order Difference (Acceleration)')
    ax3.set_title('Second Order Difference Comparison (Averaged by Dimension Ranges)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot second order difference error heatmap
    ax4 = axes[1, 1]
    sns.heatmap(
        second_diff_error_heatmap.T,  # Transpose to have dimensions on y-axis
        cmap='YlOrRd',
        xticklabels=get_smart_ticklabels(H - 2),
        yticklabels=get_smart_ticklabels(D),
        cbar_kws={'label': 'L1 Error'},
        ax=ax4,
    )
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Action Dimension')
    ax4.set_title('Second Order Difference Error Heatmap (L1)')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_smoothness_error_heatmaps(
    gt: torch.FloatTensor,
    pred: torch.FloatTensor,
    save_path: Optional[str] = None,
    action_dim_names: Optional[List[str]] = None,
):
    """
    Plot error heatmaps for first and second order differences.
    Shows where smoothness errors are largest across time and dimensions.
    """
    smoothness = compute_smoothness_metrics(gt, pred)
    
    B, H, D = gt.shape
    if action_dim_names is None:
        action_dim_names = [f'Dim {i}' for i in range(D)]
    
    first_diff_error_heatmap = smoothness['first_diff_error_heatmap'].cpu().numpy()  # [H-1, D]
    second_diff_error_heatmap = smoothness['second_diff_error_heatmap'].cpu().numpy()  # [H-2, D]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # First order difference error heatmap
    sns.heatmap(
        first_diff_error_heatmap.T,  # Transpose to have dimensions on y-axis
        cmap='YlOrRd',
        xticklabels=get_smart_ticklabels(H - 1),
        yticklabels=get_smart_ticklabels(D),
        cbar_kws={'label': 'L1 Error'},
        ax=axes[0],
    )
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Action Dimension')
    axes[0].set_title('First Order Difference Error Heatmap (L1)\nShows velocity smoothness errors')
    
    # Second order difference error heatmap
    sns.heatmap(
        second_diff_error_heatmap.T,  # Transpose to have dimensions on y-axis
        cmap='YlOrRd',
        xticklabels=get_smart_ticklabels(H - 2),
        yticklabels=get_smart_ticklabels(D),
        cbar_kws={'label': 'L1 Error'},
        ax=axes[1],
    )
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Action Dimension')
    axes[1].set_title('Second Order Difference Error Heatmap (L1)\nShows acceleration smoothness errors')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_error_heatmap(
    gt: torch.FloatTensor,
    pred: torch.FloatTensor,
    save_path: Optional[str] = None,
    action_dim_names: Optional[List[str]] = None,
):
    """
    Plot error heatmap showing mean absolute error across time and dimensions.
    """
    error_heatmap = compute_error_heatmap(gt, pred)
    error_np = error_heatmap.cpu().numpy()  # [H, D]
    
    H, D = error_np.shape
    if action_dim_names is None:
        action_dim_names = [f'Dim {i}' for i in range(D)]
    
    plt.figure(figsize=(max(8, D * 0.8), max(6, H * 0.3)))
    sns.heatmap(
        error_np.T,  # Transpose to have dimensions on y-axis
        cmap='YlOrRd',
        xticklabels=get_smart_ticklabels(H),
        yticklabels=get_smart_ticklabels(D),
        cbar_kws={'label': 'Mean Absolute Error'},
    )
    plt.xlabel('Time Step')
    plt.ylabel('Action Dimension')
    plt.title('Error Heatmap: Mean Absolute Error over Time and Dimensions')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_covariance_matrices(
    gt: torch.FloatTensor,
    pred: torch.FloatTensor,
    save_path: Optional[str] = None,
    action_dim_names: Optional[List[str]] = None,
):
    """
    Plot covariance matrices for GT, Pred, and Error.
    """
    cov_dict = compute_covariance_matrix(gt, pred)
    
    D = gt.shape[-1]
    if action_dim_names is None:
        action_dim_names = [f'Dim {i}' for i in range(D)]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    matrices = [
        ('GT Covariance', cov_dict['gt_cov']),
        ('Pred Covariance', cov_dict['pred_cov']),
        ('Error Covariance', cov_dict['error_cov']),
    ]
    
    for idx, (title, matrix) in enumerate(matrices):
        matrix_np = matrix.cpu().numpy()
        sns.heatmap(
            matrix_np,
            cmap='coolwarm',
            center=0,
            xticklabels=get_smart_ticklabels(D),
            yticklabels=get_smart_ticklabels(D),
            cbar_kws={'label': 'Covariance'},
            ax=axes[idx],
        )
        axes[idx].set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_loss_over_time(
    gt: torch.FloatTensor,
    pred: torch.FloatTensor,
    save_path: Optional[str] = None,
    loss_type: str = 'l1',
):
    """
    Plot loss curve over time steps.
    """
    loss = compute_loss_over_time(gt, pred, loss_type=loss_type)
    loss_np = loss.cpu().numpy()
    
    H = len(loss_np)
    time_steps = np.arange(H)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, loss_np, marker='o', linewidth=2, markersize=4)
    plt.xlabel('Time Step')
    plt.ylabel(f'{loss_type.upper()} Loss')
    plt.title(f'Loss vs Time Curve ({loss_type.upper()})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_trajectory_comparison(
    gt: torch.FloatTensor,
    pred: torch.FloatTensor,
    save_path: Optional[str] = None,
    action_dim_names: Optional[List[str]] = None,
    num_samples: int = 5,
):
    """
    Plot trajectory comparison for GT and Pred for a few sample trajectories.
    Uses dimension ranges to reduce the number of subplots.
    """
    B, H, D = gt.shape
    if action_dim_names is None:
        action_dim_names = [f'Dim {i}' for i in range(D)]
    
    num_samples = min(num_samples, B)
    time_steps = np.arange(H)
    
    # Define dimension ranges for averaging
    dim_ranges = [
        (0, 3, '[0, 3)'),
        (3, 6, '[3, 6)'),
        (6, 12, '[6, 12)'),
        (12, 18, '[12, 18)'),
        (18, 33, '[18, 33)'),
        (33, 48, '[33, 48)'),
    ]
    
    # Filter valid ranges based on actual dimension size
    valid_ranges = []
    for start, end, label in dim_ranges:
        if start >= D:
            continue
        end = min(end, D)
        if start < end:
            valid_ranges.append((start, end, label))
    
    num_dim_ranges = len(valid_ranges)
    
    # Select random samples
    sample_indices = np.random.choice(B, num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, num_dim_ranges, figsize=(max(15, num_dim_ranges * 3), 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if num_dim_ranges == 1:
        axes = axes.reshape(-1, 1)
    
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    for i, sample_idx in enumerate(sample_indices):
        for j, (start, end, label) in enumerate(valid_ranges):
            ax = axes[i, j] if num_samples > 1 or num_dim_ranges > 1 else axes[j]
            
            # Average over dimensions in this range
            gt_mean_segment = np.mean(gt_np[sample_idx, :, start:end], axis=1)  # [H]
            pred_mean_segment = np.mean(pred_np[sample_idx, :, start:end], axis=1)  # [H]
            
            ax.plot(time_steps, gt_mean_segment, 
                   label='GT', marker='o', linestyle='--', alpha=0.7, linewidth=2)
            ax.plot(time_steps, pred_mean_segment, 
                   label='Pred', marker='s', linestyle='-', alpha=0.7, linewidth=2)
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'Avg Value')
            ax.set_title(f'Sample {sample_idx}, Dims {label}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_all_metrics(
    gt: torch.FloatTensor,
    pred: torch.FloatTensor,
) -> Dict[str, torch.FloatTensor]:
    """
    Compute all metrics and return as a dictionary.
    """
    metrics = {}
    
    # Basic metrics
    metrics.update(compute_per_dimension_metrics(gt, pred))
    metrics.update(compute_trajectory_metrics(gt, pred))
    
    # Smoothness
    smoothness = compute_smoothness_metrics(gt, pred)
    metrics['first_diff_error'] = smoothness['first_diff_error']
    metrics['second_diff_error'] = smoothness['second_diff_error']
    
    # Overall errors (L1)
    error = torch.abs(gt - pred)
    metrics['overall_mae'] = torch.mean(error)
    
    return metrics


def plot_all_visualizations(
    gt: torch.FloatTensor,
    pred: torch.FloatTensor,
    output_dir: str,
    action_dim_names: Optional[List[str]] = None,
    prefix: str = '',
):
    """
    Generate all visualization plots and save to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Smoothness analysis
    plot_smoothness_analysis(
        gt, pred,
        save_path=os.path.join(output_dir, f'{prefix}smoothness_analysis.png'),
        action_dim_names=action_dim_names,
    )
    
    # Error heatmap
    plot_error_heatmap(
        gt, pred,
        save_path=os.path.join(output_dir, f'{prefix}error_heatmap.png'),
        action_dim_names=action_dim_names,
    )
    
    # Covariance matrices
    plot_covariance_matrices(
        gt, pred,
        save_path=os.path.join(output_dir, f'{prefix}covariance_matrices.png'),
        action_dim_names=action_dim_names,
    )
    
    # Loss over time
    plot_loss_over_time(
        gt, pred,
        save_path=os.path.join(output_dir, f'{prefix}loss_over_time.png'),
        loss_type='l1',
    )
    
    # Trajectory comparison
    plot_trajectory_comparison(
        gt, pred,
        save_path=os.path.join(output_dir, f'{prefix}trajectory_comparison.png'),
        action_dim_names=action_dim_names,
    )
    
    # Smoothness error heatmaps
    plot_smoothness_error_heatmaps(
        gt, pred,
        save_path=os.path.join(output_dir, f'{prefix}smoothness_error_heatmaps.png'),
        action_dim_names=action_dim_names,
    )


def _prepare_metrics_data_from_memory(data, dataset_name):
    pred_actions = data.get('pred_actions')
    gt_actions = data.get('gt_actions')
    if pred_actions is None:
        print(f"  Warning: {dataset_name} has no pred_actions, skipping")
        return None
    if torch.is_tensor(pred_actions):
        pred_actions = pred_actions.detach().cpu().numpy()
    if gt_actions is not None and torch.is_tensor(gt_actions):
        gt_actions = gt_actions.detach().cpu().numpy()
    return {
        'pred_actions': pred_actions,
        'gt_actions': gt_actions,
        'has_gt': gt_actions is not None,
    }


def _compute_and_save_metrics_from_data(metrics_data, output_dir, dataset_name):
    pred_actions = metrics_data['pred_actions']
    gt_actions = metrics_data['gt_actions']

    # Save basic information
    info_dict = {
        'dataset_name': str(dataset_name),
        'num_samples': int(pred_actions.shape[0]),
        'horizon': int(pred_actions.shape[1]),
        'action_dim': int(pred_actions.shape[2]),
        'has_gt': bool(metrics_data['has_gt']),
    }
    info_path = output_dir / "info.json"
    with open(info_path, 'w') as f:
        json.dump(info_dict, f, indent=2)

    # If there is ground truth, compute metrics and save visualizations
    if gt_actions is None:
        print(f"  Warning: {dataset_name} has no ground truth data, skipping metrics computation")
        return

    # Convert to torch tensor
    pred_tensor = torch.from_numpy(pred_actions).float()
    gt_tensor = torch.from_numpy(gt_actions).float()

    # Compute all metrics
    print(f"  Computing statistical metrics...")
    
    # 1. Basic metrics (the metrics that compute_all_metrics contains)
    basic_metrics = compute_all_metrics(gt_tensor, pred_tensor)
    
    # 2. Action accuracy
    accuracy_thresholds = [0.1, 0.2, 0.3, 0.5]
    action_accuracy = get_action_accuracy(gt_tensor, pred_tensor, thresholds=accuracy_thresholds)
    
    # 3. Smoothness metrics (detailed version)
    smoothness = compute_smoothness_metrics(gt_tensor, pred_tensor)
    
    # 4. Error heatmap
    error_heatmap = compute_error_heatmap(gt_tensor, pred_tensor)
    
    # 5. Covariance matrix
    covariance = compute_covariance_matrix(gt_tensor, pred_tensor)
    
    # 6. Loss over time
    loss_l1_over_time = compute_loss_over_time(gt_tensor, pred_tensor, loss_type='l1')
    loss_l2_over_time = compute_loss_over_time(gt_tensor, pred_tensor, loss_type='l2')
    
    # 7. Trajectory level metrics (already in basic_metrics, but keep detailed version)
    trajectory_metrics = compute_trajectory_metrics(gt_tensor, pred_tensor)
    
    # 8. Per dimension metrics (already in basic_metrics, but keep detailed version)
    per_dim_metrics = compute_per_dimension_metrics(gt_tensor, pred_tensor)
    
    # Summarize all metrics to dictionary
    metrics_dict = {
        # Basic metrics
        'overall_mae': float(basic_metrics['overall_mae'].item()),
        'first_diff_error': float(basic_metrics['first_diff_error'].item()),
        'second_diff_error': float(basic_metrics['second_diff_error'].item()),
        
        # Action accuracy
        'action_accuracy': {
            f'threshold_{t}': float(action_accuracy[i].item())
            for i, t in enumerate(accuracy_thresholds)
        },
        
        # Smoothness detailed metrics
        'smoothness': {
            'first_diff_error': float(smoothness['first_diff_error'].item()),
            'second_diff_error': float(smoothness['second_diff_error'].item()),
            'first_diff_error_heatmap_mean': float(torch.mean(smoothness['first_diff_error_heatmap']).item()),
            'second_diff_error_heatmap_mean': float(torch.mean(smoothness['second_diff_error_heatmap']).item()),
        },
        
        # Error heatmap statistics
        'error_heatmap': {
            'mean': float(torch.mean(error_heatmap).item()),
            'std': float(torch.std(error_heatmap).item()),
            'max': float(torch.max(error_heatmap).item()),
            'min': float(torch.min(error_heatmap).item()),
        },
        
        # Covariance matrix statistics
        'covariance': {
            'gt_cov_trace': float(torch.trace(covariance['gt_cov']).item()),
            'pred_cov_trace': float(torch.trace(covariance['pred_cov']).item()),
            'error_cov_trace': float(torch.trace(covariance['error_cov']).item()),
            'gt_cov_det': float(torch.det(covariance['gt_cov']).item()),
            'pred_cov_det': float(torch.det(covariance['pred_cov']).item()),
            'error_cov_det': float(torch.det(covariance['error_cov']).item()),
        },
        
        # Loss over time
        'loss_over_time': {
            'l1_mean': float(torch.mean(loss_l1_over_time).item()),
            'l1_std': float(torch.std(loss_l1_over_time).item()),
            'l1_max': float(torch.max(loss_l1_over_time).item()),
            'l1_min': float(torch.min(loss_l1_over_time).item()),
            'l2_mean': float(torch.mean(loss_l2_over_time).item()),
            'l2_std': float(torch.std(loss_l2_over_time).item()),
            'l2_max': float(torch.max(loss_l2_over_time).item()),
            'l2_min': float(torch.min(loss_l2_over_time).item()),
        },
        
        # Trajectory level metrics
        'trajectory': {
            'endpoint_error_mean': float(torch.mean(trajectory_metrics['endpoint_error']).item()),
            'endpoint_error_std': float(torch.std(trajectory_metrics['endpoint_error']).item()),
            'trajectory_length_error_mean': float(torch.mean(trajectory_metrics['trajectory_length_error']).item()),
            'trajectory_length_error_std': float(torch.std(trajectory_metrics['trajectory_length_error']).item()),
            'mean_error_mean': float(torch.mean(trajectory_metrics['mean_error']).item()),
            'mean_error_std': float(torch.std(trajectory_metrics['mean_error']).item()),
            'max_error_mean': float(torch.mean(trajectory_metrics['max_error']).item()),
            'max_error_std': float(torch.std(trajectory_metrics['max_error']).item()),
        },
        
        # Per dimension metrics
        'per_dimension': {
            'mae_per_dim': per_dim_metrics['mae_per_dim'].cpu().numpy().tolist(),
            'mae_per_dim_mean': float(torch.mean(per_dim_metrics['mae_per_dim']).item()),
            'mae_per_dim_std': float(torch.std(per_dim_metrics['mae_per_dim']).item()),
            'mae_per_dim_max': float(torch.max(per_dim_metrics['mae_per_dim']).item()),
            'mae_per_dim_min': float(torch.min(per_dim_metrics['mae_per_dim']).item()),
        },
    }
    
    # Save metrics to JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")
    
    # Print key metrics
    print(f"    Overall MAE: {metrics_dict['overall_mae']:.4f}")
    print(f"    First order difference error: {metrics_dict['first_diff_error']:.4f}")
    print(f"    Second order difference error: {metrics_dict['second_diff_error']:.4f}")
    print(f"    Average endpoint error: {metrics_dict['trajectory']['endpoint_error_mean']:.4f}")
    print(f"    Action accuracy (threshold=0.1): {metrics_dict['action_accuracy']['threshold_0.1']:.4f}")
    print(f"    Action accuracy (threshold=0.2): {metrics_dict['action_accuracy']['threshold_0.2']:.4f}")
    
    # Generate all visualizations
    print(f"  Generating visualizations...")
    plot_all_visualizations(
        gt_tensor,
        pred_tensor,
        str(output_dir),
        prefix='',
    )
    print(f"  Visualizations saved to: {output_dir}")


def compute_and_save_metrics_from_data(data, output_dir, dataset_name):
    """
    Read data from in-memory dict, compute all metrics and save visualizations.
    """
    metrics_data = _prepare_metrics_data_from_memory(data, dataset_name)
    if metrics_data is None:
        return
    _compute_and_save_metrics_from_data(metrics_data, output_dir, dataset_name)

