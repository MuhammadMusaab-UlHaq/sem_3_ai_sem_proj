"""Utility functions"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def save_checkpoint(model, optimizer, epoch, rmse, ncc, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rmse': rmse,
        'ncc': ncc
    }, path)


def load_checkpoint(path):
    """Load model checkpoint"""
    return torch.load(path)


def plot_training_history(history, save_dir):
    """Plot and save training metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('MSE Loss', fontsize=11)
    axes[0].set_title('Training and Validation Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RMSE
    axes[1].plot(history['val_rmse'], color='green', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('RMSE', fontsize=11)
    axes[1].set_title('Validation RMSE', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # NCC
    axes[2].plot(history['val_ncc'], color='orange', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('NCC (Correlation)', fontsize=11)
    axes[2].set_title('Validation NCC', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'training_history.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to {save_path}")
    plt.show()
