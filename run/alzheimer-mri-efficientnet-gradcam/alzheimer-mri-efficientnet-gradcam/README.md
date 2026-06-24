# Alzheimer MRI EfficientNetB0 Classification with Grad-CAM

This repository contains a GitHub-ready implementation for Alzheimer MRI image classification using **EfficientNetB0** and visual explanation using an **Grad-CAM-style full attention heatmap**.

The project is based on the notebook: 

```text
notebooks/Copy_of_GradCAM_others.ipynb
```

## Main Features

- EfficientNetB0 transfer learning for 4-class Alzheimer MRI classification
- Fine-tuning from `block5a_expand_activation`
- Training with Early Stopping and ReduceLROnPlateau
- Classification report
- Confusion matrix
- Multi-class ROC-AUC curve
- Grad-CAM-style attention heatmap visualization
- Portable dataset paths through `configs/config.yaml`
- Works on Windows, Linux, Google Colab, and Kaggle with path adjustment

## Dataset

The notebook uses the KaggleHub dataset:

```text
lukechugh/best-alzheimer-mri-dataset-99-accuracy
```

Expected local structure:

```text
data/
├── train/
│   ├── MildDemented/
│   ├── ModerateDemented/
│   ├── NonDemented/
│   └── VeryMildDemented/
└── test/
    ├── MildDemented/
    ├── ModerateDemented/
    ├── NonDemented/
    └── VeryMildDemented/
```

## Clone Repository

```bash
git clone https://github.com/fortunab/els
cd els/alzheimer-mri-efficientnet-gradcam/alzheimer-mri-efficientnet-gradcam
```

## Windows PowerShell Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell

pip install --upgrade pip
pip install -r requirements.txt 
pip install -e .
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Download Dataset

```powershell
python scripts/download_dataset.py --config configs/config.yaml
```

If automatic copying does not match the dataset structure, manually copy the dataset so that `data/train` and `data/test` exist.

## Configuration

Edit:

```text
configs/config.yaml
```

Default configuration:

```yaml
data_dir: data
train_dir: data/train
test_dir: data/test
model_path: outputs/models/alzheimers_efficientnet_model.keras
```

On Windows, you may use absolute paths with forward slashes:

```yaml
data_dir: D:/Datasets/AlzheimerMRI
train_dir: D:/Datasets/AlzheimerMRI/train
test_dir: D:/Datasets/AlzheimerMRI/test
model_path: outputs/models/alzheimers_efficientnet_model.keras
```

```powershell
@"
data_dir: data
train_dir: data/train
test_dir: data/test

image_size: [224, 224]
batch_size: 32
epochs: 50
num_classes: 4
learning_rate: 0.001
fine_tune_from_layer: block5a_expand_activation

model_path: outputs/models/alzheimers_efficientnet_model.keras
reports_dir: outputs/reports
figures_dir: outputs/figures

early_stopping:
  monitor: val_loss
  min_delta: 0.01
  patience: 11
  restore_best_weights: true

reduce_lr:
  monitor: val_loss
  factor: 0.2
  patience: 5
  min_lr: 0.000001
"@ | Set-Content .\configs\config.yaml
```


## Train Model

```powershell
python scripts/train.py --config configs/config.yaml
```

The trained model is saved to:

```text
outputs/models/alzheimers_efficientnet_model.keras
```

Training outputs are saved in:

```text
outputs/figures/
outputs/reports/
```

## Evaluate Saved Model

```powershell
python scripts/evaluate.py --config configs/config.yaml
```

Generated files:

```text
outputs/reports/classification_report.txt
outputs/figures/confusion_matrix.png
outputs/figures/roc_curve.png
```

## Generate Eigen-CAM Heatmaps

```powershell
python scripts/eigencam.py --config configs/config.yaml
```

Output:

```text
outputs/figures/eigencam_grid.png
```

## Push to GitHub

After creating an empty GitHub repository:

```powershell
git init
git add .
git commit -m "Initial commit"

git branch -M main
git remote add origin https://github.com/<your-username>/alzheimer-mri-efficientnet-eigencam.git
git push -u origin main
```

Then others can clone it with:

```bash
git clone https://github.com/<your-username>/alzheimer-mri-efficientnet-eigencam.git
```

## Citation

Follow-up.

## Notes

This project is intended for research and educational purposes. It is not a medical diagnostic system.
