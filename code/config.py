"""Configuration file for paths and hyperparameters"""

# Data paths - ADJUST THESE TO YOUR LOCAL PATHS
DATA_ROOT = r"G:\My Drive\musab\BSCS 14B Musab\Musab BSCS 14B files\Semester 3\Artificial Intelligence\sem_3_ai_sem_proj\data"  # Windows
# DATA_ROOT = "/home/username/path/to/data"  # Linux/Mac

# Model cache directory
MODEL_CACHE = r"G:\My Drive\musab\BSCS 14B Musab\Musab BSCS 14B files\Semester 3\Artificial Intelligence\sem_3_ai_sem_proj\whisper_models"  # Windows
# MODEL_CACHE = "/home/username/whisper_models"  # Linux/Mac

# Training hyperparameters
BATCH_SIZE = 8  # Increase if you have good GPU
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_WORKERS = 4  # Adjust based on your CPU cores

# Model settings
MODEL_NAME = "tiny.en"
FREEZE_ENCODER = True

# Audio settings
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 30  # seconds

# Output paths
OUTPUT_DIR = "./outputs"
# MODEL_SAVE_PATH = "./outputs/best_whisper_tiny_model.pt"
MODEL_SAVE_PATH = "./outputs/best_whisper_tiny_model.pt"
PREDICTIONS_PATH = "./outputs/predictions.csv"
PLOTS_DIR = "./plots"
