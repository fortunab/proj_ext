from pathlib import Path
import torch

BASE_DIR = Path("./data")
HYPERKVASIR_DIR = BASE_DIR / "hyperkvasir"
KVASIR_SEG_DIR = BASE_DIR / "kvasir_seg"
OUTPUTS_DIR = Path("./outputs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 2
NUM_WORKERS = 2
VQA_BATCH_SIZE = 4
VQA_EPOCHS = 2
