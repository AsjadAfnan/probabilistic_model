#!/usr/bin/env python3
"""UCI dataset training script for Tree-PIC to QPC library.

This script demonstrates training on real UCI datasets with both linear-Gaussian
and neural energy-based conditionals.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_boston, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
            logging.FileHandler('train_uci.log')
        ]
    )


def load_uci_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load UCI dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (features, target)
    """
    if dataset_name == "boston":
        data = load_boston()
        return data.data, data.target.reshape(-1, 1)
    elif dataset_name == "wine":
        data = load_wine()
        return data.data, data.target.reshape(-1, 1)
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target.reshape(-1, 1)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    """Preprocess data for training.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction of data for testing
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler


def create_uci_model(
    n_features: int,
    model_type: str = "linear",
    n_latent: int = 2
) -> Tuple[TreePIC, Dict[str, nn.Parameter]]:
    """Create a Tree-PIC model for UCI data.
    
    Args:
        n_features: Number of input features
        model_type: Type of model (linear or neural)
        n_latent: Number of latent variables
        
    Returns:
        Tuple of (TreePIC model, parameter dictionary)
    """
    # Define tree structure
    parents = {"root": None}
    scopes = {"root": set()}
    
    # Add latent variables
    for i in range(n_latent):
        z_name = f"z{i+1}"
        if i == 0:
            parents[z_name] = "root"
        else:
            parents[z_name] = f"z{i}"
        scopes[z_name] = {z_name}
        scopes["root"].add(z_name)
    
    # Add observed variables
    for i in range(n_features):
        x_name = f"x{i+1}"
        # Connect to first latent variable
        parents[x_name] = "z1"
        scopes[x_name] = {x_name}
        scopes["root"].add(x_name)
        scopes["z1"].add(x_name)
    
    # Create tree
    tree = LatentTree.from_parents(parents, scopes)
    
    # Create learnable parameters
    params = {}
    
    # Root distribution parameters
    params["root_mu"] = nn.Parameter(torch.zeros(1))
    params["root_sigma"] = nn.Parameter(torch.ones(1))
    
    # Create conditionals
    conditionals = {}
    for i in range(n_latent):
        z_name = f"z{i+1}"
        if i == 0:
            # First latent variable
            params[f"{z_name}_A"] = nn.Parameter(torch.tensor([[0.5]]))
            params[f"{z_name}_b"] = nn.Parameter(torch.zeros(1))
            params[f"{z_name}_Sigma"] = nn.Parameter(torch.tensor([[1.0]]))
            conditionals[z_name] = LinearGaussian(
                z_name, params[f"{z_name}_A"], params[f"{z_name}_b"], params[f"{z_name}_Sigma"]
            )
        else:
            # Subsequent latent variables
            if model_type == "linear":
                params[f"{z_name}_A"] = nn.Parameter(torch.tensor([[0.3]]))
                params[f"{z_name}_b"] = nn.Parameter(torch.zeros(1))
                params[f"{z_name}_Sigma"] = nn.Parameter(torch.tensor([[0.5]]))
                conditionals[z_name] = LinearGaussian(
                    z_name, params[f"{z_name}_A"], params[f"{z_name}_b"], params[f"{z_name}_Sigma"]
                )
            else:
                conditionals[z_name] = NeuralEnergyConditional(
                    z_name, 1, 1, hidden_dim=32, n_layers=2
                )
    
    # Create leaves
    leaves = {}
    for i in range(n_features):
        x_name = f"x{i+1}"
        params[f"{x_name}_mu"] = nn.Parameter(torch.zeros(1))
        params[f"{x_name}_sigma"] = nn.Parameter(torch.ones(1))
        leaves[x_name] = GaussianLeaf(
            x_name, mu=params[f"{x_name}_mu"], sigma=params[f"{x_name}_sigma"]
        )
    
    # Create TreePIC
    tree_pic = TreePIC(tree, conditionals, leaves)
    
    return tree_pic, params


def train_uci_model(
    model: TreePIC,
    params: Dict[str, nn.Parameter],
    train_loader: DataLoader,
    val_loader: DataLoader,
    quadrature: Quadrature,
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu")
) -> Dict[str, List[float]]:
    """Train the Tree-PIC model on UCI data.
    
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
    parser = argparse.ArgumentParser(description="Train Tree-PIC on UCI datasets")
    parser.add_argument("--dataset", choices=["boston", "wine", "breast_cancer"], 
                       default="boston", help="UCI dataset to use")
    parser.add_argument("--model", choices=["linear", "neural"], default="linear",
                       help="Model type: linear-Gaussian or neural energy")
    parser.add_argument("--n-latent", type=int, default=2,
                       help="Number of latent variables")
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
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Model type: {args.model}")
    logging.info(f"Number of latent variables: {args.n_latent}")
    logging.info(f"Number of epochs: {args.n_epochs}")
    
    # Load dataset
    logging.info(f"Loading {args.dataset} dataset...")
    X, y = load_uci_dataset(args.dataset)
    
    logging.info(f"Dataset shape: {X.shape}")
    logging.info(f"Target shape: {y.shape}")
    
    # Preprocess data
    logging.info("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, random_state=args.seed)
    
    # Combine features and target for training
    train_data = torch.cat([X_train, y_train], dim=1)
    test_data = torch.cat([X_test, y_test], dim=1)
    
    # Create data loaders
    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    logging.info(f"Train samples: {len(train_data)}")
    logging.info(f"Test samples: {len(test_data)}")
    logging.info(f"Features: {X_train.shape[1]}")
    
    # Create quadrature
    quadrature = Quadrature.gauss_legendre(-3.0, 3.0, args.quadrature_points)
    
    # Create model
    logging.info("Creating model...")
    model, params = create_uci_model(
        n_features=X_train.shape[1],
        model_type=args.model,
        n_latent=args.n_latent
    )
    
    # Train model
    history = train_uci_model(
        model=model,
        params=params,
        train_loader=train_loader,
        val_loader=test_loader,
        quadrature=quadrature,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device
    )
    
    # Plot results
    plot_path = output_dir / f"training_history_{args.dataset}_{args.model}.png"
    plot_training_history(history, str(plot_path))
    logging.info(f"Training plot saved to: {plot_path}")
    
    # Save final model
    model_path = output_dir / f"model_{args.dataset}_{args.model}.pt"
    torch.save({
        "model_state": model,
        "params": params,
        "history": history,
        "args": args,
        "scaler": scaler
    }, model_path)
    logging.info(f"Model saved to: {model_path}")
    
    # Print final results
    final_train_nll = history["train_nll"][-1]
    final_test_nll = history["test_nll"][-1]
    
    logging.info("Final Results:")
    logging.info(f"  Final Train NLL: {final_train_nll:.4f}")
    logging.info(f"  Final Test NLL: {final_test_nll:.4f}")
    
    # Test model on test data
    model.eval()
    with torch.no_grad():
        qpc = model.compile_to_qpc(quadrature)
        test_log_probs = qpc.log_prob(test_data)
        test_nll = -torch.mean(test_log_probs).item()
        logging.info(f"  Test NLL: {test_nll:.4f}")
    
    # Save results summary
    results = {
        "dataset": args.dataset,
        "model_type": args.model,
        "n_latent": args.n_latent,
        "n_features": X_train.shape[1],
        "n_train": len(train_data),
        "n_test": len(test_data),
        "final_train_nll": final_train_nll,
        "final_test_nll": final_test_nll,
        "test_nll": test_nll
    }
    
    results_path = output_dir / f"results_{args.dataset}_{args.model}.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
