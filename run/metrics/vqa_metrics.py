from __future__ import annotations


def evaluate_exact_match(model, processor, dataset_split, device, n_samples=100, max_new_tokens=20):
    model.eval()
    n_samples = min(n_samples, len(dataset_split))
    matches = 0

    for idx in range(n_samples):
        sample = dataset_split[idx]
        image = sample["image"].convert("RGB")
        question = sample["question"]
        gt_answer = str(sample["answer"]).strip().lower()

        inputs = processor(images=image, text=question, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        pred_answer = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip().lower()

        if pred_answer == gt_answer:
            matches += 1

    return matches / max(n_samples, 1)
