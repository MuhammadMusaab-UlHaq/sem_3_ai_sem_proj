"""Data loading utilities for Cadenza dataset"""

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
import librosa
import numpy as np
from typing import Dict, Any
import config

class CadenzaDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train'):
        """
        Args:
            root_dir: Base directory containing data
            split: 'train' or 'valid'
        """
        self.root = Path(root_dir)
        self.split = split
        
        # Set paths based on data structure
        if split == 'train':
            metadata_path = self.root / 'cadenza_data_train' / 'metadata' / 'train_metadata.json'
            self.audio_dir = self.root / 'cadenza_data_train' / 'train' / 'signals'
        else:
            metadata_path = self.root / 'cadenza_data_valid' / 'metadata' / 'valid_metadata.json'
            self.audio_dir = self.root / 'cadenza_data_valid' / 'valid' / 'signals'
        
        # Verify paths exist
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.records = json.load(f)
        
        print(f"✓ Loaded {len(self.records)} {split} samples from {metadata_path}")
        
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        signal_name = record['signal']
        
        # Load audio
        audio_path = self.audio_dir / f'{signal_name}.flac'
        
        try:
            # Load and resample to 16kHz (Whisper's requirement)
            audio, _ = librosa.load(
                str(audio_path), 
                sr=config.SAMPLE_RATE, 
                mono=True
            )
            
            # Pad or truncate to max length
            max_length = config.MAX_AUDIO_LENGTH * config.SAMPLE_RATE
            if len(audio) > max_length:
                audio = audio[:max_length]
            else:
                audio = np.pad(audio, (0, max_length - len(audio)))
                
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            audio = np.zeros(config.MAX_AUDIO_LENGTH * config.SAMPLE_RATE)
        
        return {
            'audio': torch.FloatTensor(audio),
            'target': float(record['correctness']),
            'prompt': record['prompt'],
            'hearing_loss': record['hearing_loss'],
            'signal': signal_name
        }


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    audios = torch.stack([item['audio'] for item in batch])
    targets = torch.FloatTensor([item['target'] for item in batch])
    
    return {
        'audio': audios,
        'target': targets,
        'prompts': [item['prompt'] for item in batch],
        'hearing_loss': [item['hearing_loss'] for item in batch],
        'signals': [item['signal'] for item in batch]
    }
