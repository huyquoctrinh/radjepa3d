"""Dataset for M3D-Seg: 3D multi-class semantic segmentation.

Format: .npy images (1, H, W, D), sparse .npz masks -> (C, H, W, D) foreground
channels (NO background channel; background = all channels zero).

Each sub-dataset (0000-0024) has its own label taxonomy. We build an integer
label map (D, H, W) with 0=background and c+1 for foreground channel c, and
train a per-dataset head so channel meanings stay consistent within a dataset.
"""

import ast
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from torch.utils.data import Dataset, ConcatDataset, Sampler


class M3DSegSubDataset(Dataset):
    """Single sub-dataset from M3D-Seg (multi-class)."""

    def __init__(self, data_dir, dataset_code, split="train", volume_size=(32, 256, 256)):
        self.volume_size = list(volume_size)
        self.data_dir = data_dir
        self.code = dataset_code

        json_candidates = [
            os.path.join(data_dir, dataset_code, dataset_code, f"{dataset_code}.json"),
            os.path.join(data_dir, dataset_code, f"{dataset_code}.json"),
        ]
        code_dir = os.path.join(data_dir, dataset_code)
        if os.path.isdir(code_dir):
            for root, dirs, files in os.walk(code_dir):
                for f in files:
                    if f.endswith(".json") and f.startswith(dataset_code[:4]):
                        json_candidates.append(os.path.join(root, f))
        json_path = None
        for p in json_candidates:
            if os.path.exists(p):
                json_path = p
                break

        if json_path is None:
            self.samples = []
            self.num_classes = 1
            self.labels = {}
            return

        with open(json_path) as f:
            meta = json.load(f)

        # num_classes includes background (index 0)
        self.labels = meta.get("labels", {})
        self.num_classes = max(len(self.labels), 2)
        base_dir = os.path.dirname(json_path)

        entries = meta.get(split, [])
        self.samples = []
        for entry in entries:
            for bd in [base_dir, os.path.dirname(base_dir)]:
                img_path = os.path.join(bd, entry["image"])
                mask_path = os.path.join(bd, entry["label"])
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.samples.append({"image": img_path, "mask": mask_path})
                    break

    def __len__(self):
        return len(self.samples)

    def _load_label_map(self, mask_path):
        """Return integer label map (D, H, W) with 0=bg, c+1 for fg channel c."""
        sp = sparse.load_npz(mask_path)
        shape_str = mask_path.split(".")[-2].split("_")[-1]
        gt_shape = ast.literal_eval(shape_str)
        mask = sp.toarray().reshape(gt_shape)

        # (C, H, W, D) -> (C, D, H, W); or (H, W, D) -> (1, D, H, W)
        if mask.ndim == 4:
            mask = mask.transpose(0, 3, 1, 2)
        elif mask.ndim == 3:
            mask = mask[np.newaxis].transpose(0, 3, 1, 2)

        C = mask.shape[0]
        label_map = np.zeros(mask.shape[1:], dtype=np.int64)  # (D, H, W)
        for c in range(C):
            cls = min(c + 1, self.num_classes - 1)
            label_map[mask[c] > 0] = cls
        return label_map

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = np.load(s["image"]).astype(np.float32)
        if img.ndim == 4:
            img = img.transpose(0, 3, 1, 2)  # (1, H, W, D) -> (1, D, H, W)
        elif img.ndim == 3:
            img = img[np.newaxis].transpose(0, 3, 1, 2)

        img = img - img.min()
        img = img / np.clip(img.max(), 1e-8, None)

        img_t = torch.from_numpy(img).float().unsqueeze(0)
        img_t = F.interpolate(img_t, size=self.volume_size, mode="trilinear").squeeze(0)

        label_map = self._load_label_map(s["mask"])
        lab_t = torch.from_numpy(label_map).float().view(1, 1, *label_map.shape)
        lab_t = F.interpolate(lab_t, size=self.volume_size, mode="nearest")
        lab_t = lab_t.squeeze(0).squeeze(0).long()  # (D, H, W)

        return img_t, lab_t, self.code


def num_classes_map(config):
    """Return {code: num_classes} for sub-datasets that have data."""
    out = {}
    for code in config.dataset_codes:
        ds = M3DSegSubDataset(config.data_dir, code, split="train",
                              volume_size=config.volume_size)
        if len(ds) > 0:
            out[code] = ds.num_classes
    return out


def build_seg_dataset(config, split="train"):
    """Build combined dataset from all sub-datasets."""
    datasets = []
    total = 0
    for code in config.dataset_codes:
        ds = M3DSegSubDataset(
            config.data_dir, code, split=split, volume_size=config.volume_size,
        )
        if len(ds) > 0:
            datasets.append(ds)
            total += len(ds)
            print(f"  {code}: {len(ds)} samples ({ds.num_classes} classes)")

    if not datasets:
        raise ValueError("No datasets found!")

    merged = ConcatDataset(datasets)
    print(f"  Total {split}: {total} samples from {len(datasets)} sub-datasets")
    return merged


class SameDatasetBatchSampler(Sampler):
    """Yield batches whose samples all come from ONE sub-dataset.

    Ensures every batch shares a single taxonomy so one per-dataset head applies.
    """

    def __init__(self, concat_dataset, batch_size, shuffle=True, drop_last=False,
                 seed=0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Map global indices to their sub-dataset via cumulative sizes
        bounds = concat_dataset.cumulative_sizes
        starts = [0] + bounds[:-1]
        self.groups = [list(range(s, e)) for s, e in zip(starts, bounds)]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        batches = []
        for group in self.groups:
            idxs = list(group)
            if self.shuffle:
                perm = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[i] for i in perm]
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]
        return iter(batches)

    def __len__(self):
        n = 0
        for group in self.groups:
            if self.drop_last:
                n += len(group) // self.batch_size
            else:
                n += (len(group) + self.batch_size - 1) // self.batch_size
        return n


class DistSameDatasetBatchSampler(Sampler):
    """Distributed same-dataset batching.

    Every rank iterates the SAME sequence of sub-datasets (shared seed), so at
    each step all ranks use the same taxonomy/head; samples within a global
    batch (batch_size * world_size) are partitioned disjointly across ranks.
    Incomplete global batches per sub-dataset are dropped.
    """

    def __init__(self, concat_dataset, batch_size, num_replicas, rank,
                 shuffle=True, seed=0):
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        bounds = concat_dataset.cumulative_sizes
        starts = [0] + bounds[:-1]
        self.groups = [list(range(s, e)) for s, e in zip(starts, bounds)]
        self.gb = batch_size * num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _global_batches(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        gbatches = []
        for group in self.groups:
            idxs = list(group)
            if self.shuffle:
                perm = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[i] for i in perm]
            for i in range(0, len(idxs) - self.gb + 1, self.gb):
                gbatches.append(idxs[i:i + self.gb])
        if self.shuffle:
            perm = torch.randperm(len(gbatches), generator=g).tolist()
            gbatches = [gbatches[i] for i in perm]
        return gbatches

    def __iter__(self):
        for gb in self._global_batches():
            start = self.rank * self.batch_size
            yield gb[start:start + self.batch_size]

    def __len__(self):
        return len(self._global_batches())


def collate_fn(batch):
    images, labels, codes = zip(*batch)
    return torch.stack(images), torch.stack(labels), codes[0]
