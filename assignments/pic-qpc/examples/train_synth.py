#!/usr/bin/env python3
"""Synthetic training script for Tree-PIC to QPC library.

This script demonstrates training on synthetic data with both linear-Gaussian
and neural energy-based conditionals, providing reproducible results.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pic import (
    LatentTree, Quadrature, TreePIC, QPC,
    LinearGaussian, NeuralEnergyConditional,
    GaussianLeaf, BernoulliLeaf
)
from pic.utils import set_seed, create_optimizer, create_scheduler


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('train_synth.log')
        ]
    )


def create_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 3,
    noise_std: float = 0.1,
    seed: int = 42
) -> torch.Tensor:
    """Create synthetic data for training.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        noise_std: Standard deviation of noise
        seed: Random seed for reproducibility
        
    Returns:
        Synthetic data tensor
    """
    set_seed(seed)
    
    # Generate latent variables
    z1 = torch.randn(n_samples, 1) * 1.0
    z2 = torch.randn(n_samples, 1) * 0.5 + z1 * 0.3
    
    # Generate observed variables
    x1 = z1 + torch.randn(n_samples, 1) * noise_std
    x2 = z1 * 0.8 + z2 * 0.6 + torch.randn(n_samples, 1) * noise_std
    x3 = z2 * 1.2 + torch.randn(n_samples, 1) * noise_std
    
    # Combine features
    data = torch.cat([x1, x2, x3], dim=1)
    
    return data


def create_linear_gaussian_model() -> Tuple[TreePIC, Dict[str, torch.nn.Parameter]]:
    """Create a Tree-PIC model with linear-Gaussian conditionals.
    
    Returns:
        Tuple of (TreePIC model, parameter dictionary)
    """
    # Define tree structure
    parents = {
        "root": None,
        "z1": "root",
        "z2": "z1",
        "x1": "z1",
        "x2": "z1",
        "x3": "z2"
    }
    
    scopes = {
        "root": {"z1", "z2", "x1", "x2", "x3"},
        "z1": {"z1", "x1", "x2"},
        "z2": {"z2", "x3"},
        "x1": {"x1"},
        "x2": {"x2"},
        "x3": {"x3"}
    }
    
    # Create tree
    tree = LatentTree.from_parents(parents, scopes)
    
    # Create learnable parameters
    params = {}
    
    # Root distribution parameters
    params["root_mu"] = nn.Parameter(torch.zeros(1))
    params["root_sigma"] = nn.Parameter(torch.ones(1))
    
    # Conditional parameters
    params["z1_A"] = nn.Parameter(torch.tensor([[0.5]]))
    params["z1_b"] = nn.Parameter(torch.zeros(1))
    params["z1_Sigma"] = nn.Parameter(torch.tensor([[1.0]]))
    
    params["z2_A"] = nn.Parameter(torch.tensor([[0.3]]))
    params["z2_b"] = nn.Parameter(torch.zeros(1))
    params["z2_Sigma"] = nn.Parameter(torch.tensor([[0.5]]))
    
    # Leaf parameters
    params["x1_mu"] = nn.Parameter(torch.zeros(1))
    params["x1_sigma"] = nn.Parameter(torch.ones(1))
    
    params["x2_mu"] = nn.Parameter(torch.zeros(1))
    params["x2_sigma"] = nn.Parameter(torch.ones(1))
    
    params["x3_mu"] = nn.Parameter(torch.zeros(1))
    params["x3_sigma"] = nn.Parameter(torch.ones(1))
    
    # Create conditionals
    conditionals = {
        "z1": LinearGaussian("z1", params["z1_A"], params["z1_b"], params["z1_Sigma"]),
        "z2": LinearGaussian("z2", params["z2_A"], params["z2_b"], params["z2_Sigma"])
    }
    
    # Create leaves
    leaves = {
        "x1": GaussianLeaf("x1", mu=params["x1_mu"], sigma=params["x1_sigma"]),
        "x2": GaussianLeaf("x2", mu=params["x2_mu"], sigma=params["x2_sigma"]),
        "x3": GaussianLeaf("x3", mu=params["x3_mu"], sigma=params["x3_sigma"])
    }
    
    # Create TreePIC
    tree_pic = TreePIC(tree, conditionals, leaves)
    
    return tree_pic, params


def create_neural_energy_model() -> Tuple[TreePIC, Dict[str, torch.nn.Parameter]]:
    """Create a Tree-PIC model with neural energy-based conditionals.
    
    Returns:
        Tuple of (TreePIC model, parameter dictionary)
    """
    # Define tree structure (same as linear-Gaussian)
    parents = {
        "root": None,
        "z1": "root",
        "z2": "z1",
        "x1": "z1",
        "x2": "z1",
        "x3": "z2"
    }
    
    scopes = {
        "root": {"z1", "z2", "x1", "x2", "x3"},
        "z1": {"z1", "x1", "x2"},
        "z2": {"z2", "x3"},
        "x1": {"x1"},
        "x2": {"x2"},
        "x3": {"x3"}
    }
    
    # Create tree
    tree = LatentTree.from_parents(parents, scopes)
    
    # Create learnable parameters
    params = {}
    
    # Root distribution parameters
    params["root_mu"] = nn.Parameter(torch.zeros(1))
    params["root_sigma"] = nn.Parameter(torch.ones(1))
    
    # Leaf parameters
    params["x1_mu"] = nn.Parameter(torch.zeros(1))
    params["x1_sigma"] = nn.Parameter(torch.ones(1))
    
    params["x2_mu"] = nn.Parameter(torch.zeros(1))
    params["x2_sigma"] = nn.Parameter(torch.ones(1))
    
    params["x3_mu"] = nn.Parameter(torch.zeros(1))
    params["x3_sigma"] = nn.Parameter(torch.ones(1))
    
    # Create conditionals (neural energy-based)
    conditionals = {
        "z1": NeuralEnergyConditional("z1", 0, 1, hidden_dim=32, n_layers=2),
        "z2": NeuralEnergyConditional("z2", 1, 1, hidden_dim=32, n_layers=2)
    }
    
    # Create leaves
    leaves = {
        "x1": GaussianLeaf("x1", mu=params["x1_mu"], sigma=params["x1_sigma"]),
        "x2": GaussianLeaf("x2", mu=params["x2_mu"], sigma=params["x2_sigma"]),
        "x3": GaussianLeaf("x3", mu=params["x3_mu"], sigma=params["x3_sigma"])
    }
    
    # Create TreePIC
    tree_pic = TreePIC(tree, conditionals, leaves)
    
    return tree_pic, params


def train_model(
    model: TreePIC,
    params: Dict[str, nn.Parameter],
    train_loader: DataLoader,
    val_loader: DataLoader,
    quadrature: Quadrature,
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu")
) -> Dict[str, List[float]]:
    """Train the Tree-PIC model.
    
    Args:
        model: TreePIC model
        params: Model parameters
        train_loader: Training data loader
        val_loader: Validation data loader
        quadrature: Quadrature for integration
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Dictionary with training history
    """
    # Move quadrature to device
    quadrature = quadrature.to(device)
    
    # Create optimizer
    optimizer = create_optimizer(
        nn.ParameterList(params.values()),
        optimizer_type="adam",
        lr=lr,
        weight_decay=1e-5
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type="step",
        step_size=30,
        gamma=0.5
    )
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_nll": [],
        "val_nll": []
    }
    
    logging.info("Starting training...")
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_nlls = []
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Compile to QPC
            qpc = model.compile_to_qpc(quadrature)
            
            # Compute log probabilities
            log_probs = qpc.log_prob(data)
            
            # Negative log-likelihood loss
            nll = -torch.mean(log_probs)
            
            # Add regularization
            reg_loss = 0.0
            for param in params.values():
                reg_loss += torch.sum(param**2)
            reg_loss *= 1e-4
            
            total_loss = nll + reg_loss
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(params.values(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(total_loss.item())
            train_nlls.append(nll.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        val_nlls = []
        
        with torch.no_grad():
            for data, in val_loader:
                data = data.to(device)
                
                qpc = model.compile_to_qpc(quadrature)
                log_probs = qpc.log_prob(data)
                nll = -torch.mean(log_probs)
                
                val_losses.append(nll.item())
                val_nlls.append(nll.item())
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(np.mean(val_losses))
        history["train_nll"].append(np.mean(train_nlls))
        history["val_nll"].append(np.mean(val_nlls))
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            logging.info(
                f"Epoch {epoch+1}/{n_epochs}: "
                f"Train NLL: {history['train_nll'][-1]:.4f}, "
                f"Val NLL: {history['val_nll'][-1]:.4f}"
            )
    
    logging.info("Training completed!")
    return history


def plot_training_history(history: Dict[str, List[float]], save_path: str) -> None:
    """Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Plot NLL
    ax2.plot(history["train_nll"], label="Train NLL")
    ax2.plot(history["val_nll"], label="Val NLL")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Negative Log-Likelihood")
    ax2.set_title("Training and Validation NLL")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Tree-PIC on synthetic data")
    parser.add_argument("--model", choices=["linear", "neural"], default="linear",
                       help="Model type: linear-Gaussian or neural energy")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of synthetic samples")
    parser.add_argument("--n-epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--quadrature-points", type=int, default=32,
                       help="Number of quadrature points")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", default="cpu",
                       help="Device to use (cpu/cuda)")
    parser.add_argument("--output-dir", default="outputs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    set_seed(args.seed)
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logging.info(f"Using device: {device}")
    logging.info(f"Model type: {args.model}")
    logging.info(f"Number of samples: {args.n_samples}")
    logging.info(f"Number of epochs: {args.n_epochs}")
    
    # Create synthetic data
    logging.info("Creating synthetic data...")
    data = create_synthetic_data(n_samples=args.n_samples, seed=args.seed)
    
    # Split data
    n_train = int(0.8 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]
    
    # Create data loaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logging.info(f"Train samples: {len(train_data)}")
    logging.info(f"Val samples: {len(val_data)}")
    
    # Create quadrature
    quadrature = Quadrature.gauss_legendre(-3.0, 3.0, args.quadrature_points)
    
    # Create model
    logging.info("Creating model...")
    if args.model == "linear":
        model, params = create_linear_gaussian_model()
    else:
        model, params = create_neural_energy_model()
    
    # Train model
    history = train_model(
        model=model,
        params=params,
        train_loader=train_loader,
        val_loader=val_loader,
        quadrature=quadrature,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device
    )
    
    # Plot results
    plot_path = output_dir / f"training_history_{args.model}.png"
    plot_training_history(history, str(plot_path))
    logging.info(f"Training plot saved to: {plot_path}")
    
    # Save final model
    model_path = output_dir / f"model_{args.model}.pt"
    torch.save({
        "model_state": model,
        "params": params,
        "history": history,
        "args": args
    }, model_path)
    logging.info(f"Model saved to: {model_path}")
    
    # Print final results
    final_train_nll = history["train_nll"][-1]
    final_val_nll = history["val_nll"][-1]
    
    logging.info("Final Results:")
    logging.info(f"  Final Train NLL: {final_train_nll:.4f}")
    logging.info(f"  Final Val NLL: {final_val_nll:.4f}")
    
    # Test model on validation data
    model.eval()
    with torch.no_grad():
        qpc = model.compile_to_qpc(quadrature)
        val_log_probs = qpc.log_prob(val_data)
        val_nll = -torch.mean(val_log_probs).item()
        logging.info(f"  Test Val NLL: {val_nll:.4f}")


if __name__ == "__main__":
    main()
