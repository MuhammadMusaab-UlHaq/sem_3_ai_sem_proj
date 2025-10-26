"""Evaluation script"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import config
from data_loader import CadenzaDataset, collate_fn
from model_baseline_D import WhisperIntelligibilityPredictor
from utils import load_checkpoint


def evaluate_model():
    """Evaluate trained model on validation set"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = CadenzaDataset(config.DATA_ROOT, split='valid')
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS
    )
    
    # Load model
    print("Loading trained model...")
    model = WhisperIntelligibilityPredictor(model_cache=config.MODEL_CACHE)
    checkpoint = load_checkpoint(config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint RMSE: {checkpoint['rmse']:.4f}")
    print(f"Checkpoint NCC: {checkpoint['ncc']:.4f}\n")
    
    # Make predictions
    predictions = []
    ground_truth = []
    signals = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            audio = batch['audio'].to(device)
            targets = batch['target']
            
            preds = model(audio).cpu().numpy()
            
            predictions.extend(preds)
            ground_truth.extend(targets.numpy())
            signals.extend(batch['signals'])
    
    # Calculate final metrics
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    rmse = np.sqrt(np.mean((predictions - ground_truth)**2))
    ncc = np.corrcoef(predictions, ground_truth)[0, 1]
    mae = np.mean(np.abs(predictions - ground_truth))
    
    print(f"\n{'='*50}")
    print("Final Evaluation Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  NCC:  {ncc:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"{'='*50}\n")
    
    # Save predictions
    results_df = pd.DataFrame({
        'signal': signals,
        'ground_truth': ground_truth,
        'prediction': predictions,
        'error': np.abs(predictions - ground_truth)
    })
    results_df.to_csv(config.PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {config.PREDICTIONS_PATH}")
    
    # Create scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(ground_truth, predictions, alpha=0.5, s=20)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction', linewidth=2)
    plt.xlabel('Ground Truth Intelligibility', fontsize=12)
    plt.ylabel('Predicted Intelligibility', fontsize=12)
    plt.title(f'Whisper-tiny Predictions\nRMSE: {rmse:.4f}, NCC: {ncc:.4f}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    
    plot_path = Path(config.PLOTS_DIR) / 'predictions_scatter.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to {plot_path}")
    plt.show()
    
    return results_df, rmse, ncc


if __name__ == '__main__':
    evaluate_model()
