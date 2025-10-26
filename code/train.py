"""Training script for Whisper intelligibility model"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import config
from data_loader import CadenzaDataset, collate_fn
from model_baseline_D import WhisperIntelligibilityPredictor
from utils import plot_training_history, save_checkpoint, load_checkpoint


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    losses = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        audio = batch['audio'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        predictions = model(audio)
        loss = criterion(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return np.mean(losses)


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    losses = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            audio = batch['audio'].to(device)
            targets = batch['target'].to(device)
            
            predictions = model(audio)
            loss = criterion(predictions, targets)
            
            losses.append(loss.item())
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    rmse = np.sqrt(np.mean((all_preds - all_targets)**2))
    ncc = np.corrcoef(all_preds, all_targets)[0, 1]
    
    return np.mean(losses), rmse, ncc, all_preds, all_targets


def train_model():
    """Main training function"""
    
    # Create output directories
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    Path(config.PLOTS_DIR).mkdir(exist_ok=True)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*50}")
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*50}\n")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = CadenzaDataset(config.DATA_ROOT, split='train')
    val_dataset = CadenzaDataset(config.DATA_ROOT, split='valid')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    # Initialize model
    model = WhisperIntelligibilityPredictor(
        model_cache=config.MODEL_CACHE,
        freeze_encoder=config.FREEZE_ENCODER
    )
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.regression_head.parameters(),
        lr=config.LEARNING_RATE
    )
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_ncc': []
    }
    
    best_rmse = float('inf')
    
    # Training loop
    print(f"\n{'='*50}")
    print(f"Starting training for {config.EPOCHS} epochs")
    print(f"{'='*50}\n")
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, rmse, ncc, preds, targets = validate_epoch(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(rmse)
        history['val_ncc'].append(ncc)
        
        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val RMSE:   {rmse:.4f}")
        print(f"  Val NCC:    {ncc:.4f}")
        
        # Save best model
        if rmse < best_rmse:
            best_rmse = rmse
            save_checkpoint(
                model, optimizer, epoch, rmse, ncc,
                config.MODEL_SAVE_PATH
            )
            print(f"  ✓ Saved best model! (RMSE: {rmse:.4f})")
    
    # Plot training history
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"Best RMSE: {best_rmse:.4f}")
    print(f"{'='*50}\n")
    
    plot_training_history(history, config.PLOTS_DIR)
    
    return model, history


if __name__ == '__main__':
    train_model()
