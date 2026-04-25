from __future__ import annotations

from torch.utils.data import Dataset


class KvasirVQADataset(Dataset):
    def __init__(self, hf_split, processor, max_target_len=32):
        self.data = hf_split
        self.processor = processor
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"].convert("RGB")
        question = item["question"]
        answer = item["answer"]

        encoding = self.processor(
            images=image,
            text=question,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = self.processor.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_len,
            return_tensors="pt",
        ).input_ids

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = labels.squeeze(0)
        return encoding
