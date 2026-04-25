
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "sam2"

SAM2_BASE_MODEL_ID = "facebook/sam2.1-hiera-tiny"
SAM2_DATASET_ID = "RGarrido03/kvasir-seg-augmented"
SAM2_BATCH_SIZE = 4
SAM2_LEARNING_RATE = 1e-4
SAM2_EPOCHS = 1
SAM2_LORA_R = 8
SAM2_LORA_ALPHA = 16
SAM2_LORA_DROPOUT = 0.05
SAM2_MERGED_DIR = PROJECT_ROOT / "sam2_kvasir_merged"

for sub in ["images", "charts", "reports"]:
    (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
