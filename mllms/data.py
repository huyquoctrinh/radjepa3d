"""Datasets for M3D MLLM training: VQA, Captioning, and Merged."""

import csv
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset


class M3DVQADataset(Dataset):
    """M3D-VQA: closed-ended visual question answering."""

    def __init__(self, config, tokenizer, split="train"):
        self.config = config
        self.tokenizer = tokenizer
        self.npy_dir = config.npy_dir

        csv_path = config.vqa_train_csv if split == "train" else config.vqa_val_csv

        self.samples = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                parts = row["Image Path"].split("/")
                npy_path = os.path.join(self.npy_dir, parts[1], parts[2], parts[3])
                if not os.path.exists(npy_path):
                    continue
                self.samples.append({
                    "npy_path": npy_path,
                    "question": row["Question"],
                    "choice_a": row["Choice A"],
                    "choice_b": row["Choice B"],
                    "choice_c": row["Choice C"],
                    "choice_d": row["Choice D"],
                    "answer": row["Answer"],
                    "answer_choice": row["Answer Choice"],
                })

        cap = getattr(config, "vqa_max_train", 0)
        if split == "train" and cap and len(self.samples) > cap:
            import random
            random.Random(0).shuffle(self.samples)
            self.samples = self.samples[:cap]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        volume = self._load_volume(s["npy_path"])

        user_content = (
            f"{self.config.vision_start_token}{self.config.vision_end_token}\n"
            f"Based on the 3D CT scan shown above, answer the following question.\n"
            f"Question: {s['question']}\n"
            f"Options: A) {s['choice_a']} B) {s['choice_b']} "
            f"C) {s['choice_c']} D) {s['choice_d']}"
        )
        assistant_content = f"{s['answer_choice']}. {s['answer']}"

        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        target = f"{assistant_content}<|im_end|>"

        return volume, prompt, target

    def _load_volume(self, path):
        try:
            return torch.from_numpy(np.load(path).astype(np.float32))
        except Exception:
            return torch.zeros(1, *self.config.volume_size)


class M3DCapDataset(Dataset):
    """M3D-Cap: report generation / captioning."""

    def __init__(self, config, tokenizer, split="train"):
        self.config = config
        self.tokenizer = tokenizer
        self.npy_dir = config.npy_dir

        with open(config.data_json) as f:
            data = json.load(f)
        entries = data[split]

        self.samples = []
        for entry in entries:
            img_path = entry["image"]
            parts = img_path.split("/")
            npy_path = os.path.join(self.npy_dir, parts[1], parts[2], parts[3])
            text_path = os.path.join(self.npy_dir, parts[1], parts[2], "text.txt")

            if not os.path.exists(npy_path) or not os.path.exists(text_path):
                continue

            self.samples.append({
                "npy_path": npy_path,
                "text_path": text_path,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        volume = self._load_volume(s["npy_path"])

        try:
            with open(s["text_path"], "r") as f:
                report = f.read().strip()
        except Exception:
            report = ""

        # Truncate long reports
        report_tokens = report.split()
        if len(report_tokens) > 256:
            report = " ".join(report_tokens[:256])

        user_content = (
            f"{self.config.vision_start_token}{self.config.vision_end_token}\n"
            f"Describe the findings in this 3D CT scan in detail."
        )
        assistant_content = report

        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        target = f"{assistant_content}<|im_end|>"

        return volume, prompt, target

    def _load_volume(self, path):
        try:
            return torch.from_numpy(np.load(path).astype(np.float32))
        except Exception:
            return torch.zeros(1, *self.config.volume_size)


def build_merged_dataset(config, tokenizer, split="train"):
    """Build merged VQA + Captioning dataset (caption oversampling on train)."""
    vqa = M3DVQADataset(config, tokenizer, split=split)
    cap = M3DCapDataset(config, tokenizer, split=split)
    over = getattr(config, "caption_oversample", 1) if split == "train" else 1
    merged = ConcatDataset([vqa] + [cap] * over)
    print(f"  Merged dataset ({split}): VQA={len(vqa)} + Cap={len(cap)}x{over} = {len(merged)}")
    return merged


def collate_fn(batch, tokenizer, max_length=2048):
    """Collate: tokenize prompts + targets, build labels with proper masking."""
    volumes, prompts, targets = zip(*batch)
    volumes = torch.stack(volumes)

    all_input_ids = []
    all_labels = []

    for prompt, target in zip(prompts, targets):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    max_len = max(len(ids) for ids in all_input_ids)
    pad_id = tokenizer.pad_token_id or 0

    padded_input_ids = []
    padded_labels = []
    padded_mask = []

    for ids, labs in zip(all_input_ids, all_labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append(ids + [pad_id] * pad_len)
        padded_labels.append(labs + [-100] * pad_len)
        padded_mask.append([1] * len(ids) + [0] * pad_len)

    return (
        volumes,
        torch.tensor(padded_input_ids, dtype=torch.long),
        torch.tensor(padded_mask, dtype=torch.long),
        torch.tensor(padded_labels, dtype=torch.long),
    )
