# test_setup.py
import config
from data_loader import CadenzaDataset
from model_baseline_D import WhisperIntelligibilityPredictor
import torch

print("Testing data loading...")
dataset = CadenzaDataset(config.DATA_ROOT, 'train')
sample = dataset[0]
print(f"✓ Audio shape: {sample['audio'].shape}")
print(f"✓ Target: {sample['target']}")

print("\nTesting model...")
model = WhisperIntelligibilityPredictor(model_cache=config.MODEL_CACHE)
with torch.no_grad():
    pred = model(sample['audio'].unsqueeze(0))
print(f"✓ Prediction: {pred.item():.4f}")
print("\n✓ All tests passed!")
