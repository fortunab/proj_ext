
from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Sam2Model, Sam2Processor


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_processor(model_id: str):
    return Sam2Processor.from_pretrained(model_id)


def load_base_model(model_id: str, device: str | None = None):
    device = device or get_device()
    model = Sam2Model.from_pretrained(model_id)
    return model.to(device)


def apply_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    return get_peft_model(model, lora_config)


def load_lora_for_merge(base_model_id: str, adapter_path: str):
    base_model = Sam2Model.from_pretrained(base_model_id)
    return PeftModel.from_pretrained(base_model, adapter_path)


def merge_and_save_lora(base_model_id: str, adapter_path: str, output_dir: str):
    model = load_lora_for_merge(base_model_id, adapter_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    return output_dir
