from __future__ import annotations

from datasets import load_dataset
from transformers import BlipForQuestionAnswering, BlipProcessor

from bionnica.config import OUTPUTS_DIR, DEVICE
from bionnica.metrics.vqa_metrics import evaluate_exact_match
from bionnica.utils.io_utils import ensure_dir, save_text_report


def main() -> None:
    output_dir = ensure_dir(OUTPUTS_DIR / "blip_vqa_base")
    dataset = load_dataset("SimulaMet/Kvasir-VQA-x1")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(DEVICE)

    n_eval = min(300, len(dataset["test"]))
    em = evaluate_exact_match(model, processor, dataset["test"], device=DEVICE, n_samples=n_eval)

    lines = [
        "Model: BLIP-VQA Base",
        f"DEVICE = {DEVICE}",
        "Setup:",
        "- Pretrained Salesforce/blip-vqa-base",
        "- No fine-tuning applied",
        f"Test samples evaluated: {n_eval}",
        "",
        f"Exact Match: {em:.4f}",
    ]

    save_text_report(output_dir / "blip_vqa_base_report.txt", lines)
    print(f"Saved: {output_dir / 'blip_vqa_base_report.txt'}")


if __name__ == "__main__":
    main()
