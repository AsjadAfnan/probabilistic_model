"""Utility functions for probabilistic inference circuits."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_log(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute safe logarithm with small epsilon to avoid log(0).
    
    Args:
        x: Input tensor
        eps: Small epsilon value to add
        
    Returns:
        Safe log tensor
    """
    return torch.log(x + eps)


def safe_exp(x: Tensor, max_val: float = 20.0) -> Tensor:
    """Compute safe exponential with clipping to avoid overflow.
    
    Args:
        x: Input tensor
        max_val: Maximum value before clipping
        
    Returns:
        Safe exp tensor
    """
    return torch.exp(torch.clamp(x, max=max_val))


def log_sum_exp_stable(
    log_values: Tensor,
    dim: int = -1,
    keepdim: bool = False
) -> Tensor:
    """Compute stable log-sum-exp operation.
    
    Args:
        log_values: Log values tensor
        dim: Dimension along which to sum
        keepdim: Whether to keep the reduced dimension
        
    Returns:
        Log-sum-exp result
    """
    max_val = torch.max(log_values, dim=dim, keepdim=True)[0]
    stable_log_values = log_values - max_val
    exp_values = torch.exp(stable_log_values)
    sum_exp = torch.sum(exp_values, dim=dim, keepdim=True)
    log_sum = torch.log(sum_exp) + max_val
    
    if not keepdim:
        log_sum = log_sum.squeeze(dim)
    
    return log_sum


def softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """Compute softplus function with optional thresholding.
    
    Args:
        x: Input tensor
        beta: Beta parameter for softplus
        threshold: Threshold for linear approximation
        
    Returns:
        Softplus result
    """
    if beta != 1.0:
        x = x * beta
    
    # Use linear approximation for large values
    mask = x > threshold
    result = torch.zeros_like(x)
    result[mask] = x[mask]
    result[~mask] = torch.log1p(torch.exp(x[~mask]))
    
    if beta != 1.0:
        result = result / beta
    
    return result


def inv_softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """Compute inverse softplus function.
    
    Args:
        x: Input tensor
        beta: Beta parameter for softplus
        threshold: Threshold for linear approximation
        
    Returns:
        Inverse softplus result
    """
    if beta != 1.0:
        x = x * beta
    
    # Use linear approximation for large values
    mask = x > threshold
    result = torch.zeros_like(x)
    result[mask] = x[mask]
    result[~mask] = torch.log(torch.expm1(x[~mask]))
    
    return result


def batch_diag(x: Tensor) -> Tensor:
    """Create batch diagonal matrix from vector.
    
    Args:
        x: Input tensor of shape (..., n)
        
    Returns:
        Diagonal matrix of shape (..., n, n)
    """
    n = x.shape[-1]
    eye = torch.eye(n, device=x.device, dtype=x.dtype)
    return x.unsqueeze(-1) * eye


def batch_trace(x: Tensor) -> Tensor:
    """Compute batch trace of matrices.
    
    Args:
        x: Input tensor of shape (..., n, n)
        
    Returns:
        Trace tensor of shape (...)
    """
    return torch.diagonal(x, dim1=-2, dim2=-1).sum(dim=-1)


def batch_det(x: Tensor) -> Tensor:
    """Compute batch determinant of matrices.
    
    Args:
        x: Input tensor of shape (..., n, n)
        
    Returns:
        Determinant tensor of shape (...)
    """
    return torch.linalg.det(x)


def batch_inv(x: Tensor) -> Tensor:
    """Compute batch inverse of matrices.
    
    Args:
        x: Input tensor of shape (..., n, n)
        
    Returns:
        Inverse tensor of shape (..., n, n)
    """
    return torch.linalg.inv(x)


def save_tensor(tensor: Tensor, filepath: Union[str, Path]) -> None:
    """Save tensor to file.
    
    Args:
        tensor: Tensor to save
        filepath: Path to save file
    """
    torch.save(tensor, filepath)


def load_tensor(filepath: Union[str, Path]) -> Tensor:
    """Load tensor from file.
    
    Args:
        filepath: Path to load file
        
    Returns:
        Loaded tensor
    """
    return torch.load(filepath)


def save_dict(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    # Convert tensors to lists for JSON serialization
    json_data = {}
    for key, value in data.items():
        if isinstance(value, Tensor):
            json_data[key] = value.tolist()
        else:
            json_data[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=2)


def load_dict(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load dictionary from JSON file.
    
    Args:
        filepath: Path to load file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to tensors where appropriate
    for key, value in data.items():
        if isinstance(value, list):
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError):
                # Keep as list if conversion fails
                pass
    
    return data


def check_gradients(model: torch.nn.Module, max_norm: float = 10.0) -> bool:
    """Check if gradients are finite and within bounds.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        
    Returns:
        True if gradients are valid, False otherwise
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                print(f"Non-finite gradients in {name}")
                return False
            
            grad_norm = param.grad.norm()
            if grad_norm > max_norm:
                print(f"Large gradient norm in {name}: {grad_norm}")
                return False
    
    return True


def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0) -> float:
    """Clip gradients to maximum norm.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        
    Returns:
        Total gradient norm before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm


def count_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name()
        info['device_capability'] = torch.cuda.get_device_capability()
    
    return info


def to_device(x: Union[Tensor, Dict, List], device: torch.device) -> Union[Tensor, Dict, List]:
    """Recursively move tensors to device.
    
    Args:
        x: Tensor, dict, or list to move
        device: Target device
        
    Returns:
        Object with tensors moved to device
    """
    if isinstance(x, Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {key: to_device(value, device) for key, value in x.items()}
    elif isinstance(x, list):
        return [to_device(item, device) for item in x]
    else:
        return x


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer for model.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ("adam", "sgd", "adamw")
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "step",
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ("step", "cosine", "exponential")
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_type.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_type.lower() == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_type.lower() == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
