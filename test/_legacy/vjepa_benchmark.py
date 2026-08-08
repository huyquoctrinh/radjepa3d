"""Benchmark V-JEPA embeddings via linear probing and k-NN evaluation."""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import csv
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

torch.backends.cudnn.enabled = False

from vjepa_config import VJEPAConfig
from vjepa_data import M3DCTDataset
from vjepa_model import VJEPA3D


TASKS = {
    "1": "plane",
    "2": "phase",
    "3": "organ",
}

CSV_FILES = {
    "train": "/data2/gdp2988/3d_cot/dataset/m3d_vqa_text/M3D_VQA_train.csv",
    "val": "/data2/gdp2988/3d_cot/dataset/m3d_vqa_text/M3D_VQA_val.csv",
    "test": "/data2/gdp2988/3d_cot/dataset/m3d_vqa_text/M3D_VQA_test.csv",
}


def load_labels(csv_path, qtype, min_count=10):
    """Load (image_path, label) pairs for a given question type."""
    samples = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["Question Type"] == qtype:
                img = row["Image Path"]
                label = row["Answer"].strip()
                samples[img] = label

    # Filter rare labels
    counts = Counter(samples.values())
    valid_labels = {l for l, c in counts.items() if c >= min_count}
    samples = {k: v for k, v in samples.items() if v in valid_labels}
    return samples


def img_path_to_npy(img_path, npy_dir):
    parts = img_path.split("/")
    return os.path.join(npy_dir, parts[1], parts[2], parts[3])


@torch.no_grad()
def extract_embeddings(model, npy_paths, batch_size=32, device="cuda"):
    """Extract mean-pooled embeddings from the target encoder."""
    model.eval()
    encoder = model.target_encoder

    all_embeds = []
    for i in range(0, len(npy_paths), batch_size):
        batch_paths = npy_paths[i:i + batch_size]
        volumes = []
        for p in batch_paths:
            try:
                vol = np.load(p).astype(np.float32)
                volumes.append(torch.from_numpy(vol))
            except Exception:
                volumes.append(torch.zeros(1, 32, 256, 256))

        batch = torch.stack(volumes).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens = encoder(batch, indices=None)  # (B, 1024, dim)
        embeds = tokens.float().mean(dim=1)  # (B, dim) — mean pool
        all_embeds.append(embeds.cpu())

        if (i // batch_size) % 20 == 0:
            print(f"  Extracted {min(i + batch_size, len(npy_paths))}/{len(npy_paths)}",
                  flush=True)

    return torch.cat(all_embeds, dim=0).numpy()


def run_knn(train_embeds, train_labels, test_embeds, test_labels, k=20):
    """k-NN evaluation."""
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(train_embeds, train_labels)
    preds = knn.predict(test_embeds)
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="weighted")
    return acc, f1, preds


def run_linear_probe(train_embeds, train_labels, test_embeds, test_labels,
                     n_classes, dim, epochs=100, lr=1e-3, device="cuda"):
    """Linear probing: train a single linear layer."""
    le = LabelEncoder()
    y_train = torch.tensor(le.fit_transform(train_labels), dtype=torch.long)
    y_test = torch.tensor(le.transform(test_labels), dtype=torch.long)

    X_train = torch.tensor(train_embeds, dtype=torch.float32)
    X_test = torch.tensor(test_embeds, dtype=torch.float32)

    probe = nn.Linear(dim, n_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    probe.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = probe(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    probe.eval()
    with torch.no_grad():
        logits = probe(X_test.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()

    test_labels_enc = y_test.numpy()
    acc = accuracy_score(test_labels_enc, preds)
    f1 = f1_score(test_labels_enc, preds, average="weighted")
    report = classification_report(
        test_labels_enc, preds, target_names=le.classes_, zero_division=0,
    )
    return acc, f1, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tasks", type=str, default="1,2,3",
                        help="Comma-separated qtypes to benchmark")
    parser.add_argument("--max-train", type=int, default=5000,
                        help="Max train samples for speed (0=all)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda"

    config = VJEPAConfig()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = VJEPA3D(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.context_encoder.load_state_dict(ckpt["context_encoder"])
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    model.predictor.load_state_dict(ckpt["predictor"])
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("best_loss", ckpt.get("global_step", "?"))
    print(f"  Loaded (epoch={epoch})")

    task_list = args.tasks.split(",")

    for qtype in task_list:
        task_name = TASKS.get(qtype, f"qtype{qtype}")
        print(f"\n{'='*60}")
        print(f"Task: {task_name} (QType {qtype})")
        print(f"{'='*60}")

        # Load labels
        train_labels_map = load_labels(CSV_FILES["train"], qtype)
        test_labels_map = load_labels(CSV_FILES["val"], qtype)

        # Map to npy paths and filter to existing files
        train_paths, train_labels = [], []
        for img, label in train_labels_map.items():
            npy = img_path_to_npy(img, config.npy_dir)
            if os.path.exists(npy):
                train_paths.append(npy)
                train_labels.append(label)

        test_paths, test_labels = [], []
        for img, label in test_labels_map.items():
            npy = img_path_to_npy(img, config.npy_dir)
            if os.path.exists(npy):
                test_paths.append(npy)
                test_labels.append(label)

        # Align label spaces
        all_labels = sorted(set(train_labels) & set(test_labels))
        train_mask = [l in all_labels for l in train_labels]
        test_mask = [l in all_labels for l in test_labels]
        train_paths = [p for p, m in zip(train_paths, train_mask) if m]
        train_labels = [l for l, m in zip(train_labels, train_mask) if m]
        test_paths = [p for p, m in zip(test_paths, test_mask) if m]
        test_labels = [l for l, m in zip(test_labels, test_mask) if m]

        # Subsample train for speed
        if args.max_train > 0 and len(train_paths) > args.max_train:
            indices = np.random.RandomState(42).permutation(len(train_paths))[:args.max_train]
            train_paths = [train_paths[i] for i in indices]
            train_labels = [train_labels[i] for i in indices]

        n_classes = len(all_labels)
        print(f"  Train: {len(train_paths)}, Test: {len(test_paths)}, Classes: {n_classes}",
              flush=True)
        print(f"  Labels: {all_labels}")

        if n_classes < 2 or len(test_paths) < 10:
            print("  Skipping — not enough data")
            continue

        # Extract embeddings
        print("  Extracting train embeddings...")
        train_embeds = extract_embeddings(model, train_paths, args.batch_size, device)
        print("  Extracting test embeddings...")
        test_embeds = extract_embeddings(model, test_paths, args.batch_size, device)

        # k-NN
        print(f"\n  --- k-NN (k=20) ---")
        knn_acc, knn_f1, _ = run_knn(train_embeds, train_labels, test_embeds, test_labels)
        print(f"  Accuracy: {knn_acc:.4f}")
        print(f"  F1 (weighted): {knn_f1:.4f}")

        # Linear probe
        print(f"\n  --- Linear Probe (100 epochs) ---")
        lp_acc, lp_f1, report = run_linear_probe(
            train_embeds, train_labels, test_embeds, test_labels,
            n_classes, config.dim, device=device,
        )
        print(f"  Accuracy: {lp_acc:.4f}")
        print(f"  F1 (weighted): {lp_f1:.4f}")
        print(f"\n{report}")


if __name__ == "__main__":
    main()
