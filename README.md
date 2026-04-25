# Bionnica Structured Project

This refactor splits the original notebook into multiple Python files so the project is easier to read, maintain, and extend.

## Structure

- `bionnica/config.py` — paths and shared constants
- `bionnica/datasets/` — dataset classes
- `bionnica/models/` — neural network definitions
- `bionnica/metrics/` — evaluation utilities
- `bionnica/utils/` — helpers for paths, reports, and plotting
- `bionnica/scripts/train_classification.py` — HyperKvasir classification workflow
- `bionnica/scripts/train_vqa.py` — Kvasir VQA workflow
- `bionnica/scripts/generate_synthetic_reports.py` — synthetic charts, CSV, and TXT outputs
- `outputs/` — default folder for generated text reports and figures

## TXT outputs

The scripts save plain-text outputs such as:

- dataset summary
- training log
- evaluation metrics
- exact-match score
- synthetic metrics report

## Example usage

### 1. Classification setup

```bash
python -m bionnica.files.train_classification
```

### 2. VQA setup

```bash
python -m bionnica.files.train_vqa
```

### 3. Charts and tables

```bash
python -m bionnica.files.generate_reports
```

## Notes

- I preserved the notebook logic, but turned repeated code into reusable modules.
- Some notebook-specific commands such as `drive.mount(...)`, `files.download(...)`, and inline `display(...)` were intentionally removed or replaced with file-based outputs.
- The original notebook referenced `evaluate_exact_match(...)` without defining it. A reusable implementation was added in `bionnica/metrics/vqa_metrics.py`.
