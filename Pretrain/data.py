"""Raw-signal EMG windows read from a simple on-disk format.

Convert any dataset (EMG-1, Ninapro, emg2pose, ...) to this layout once, then
every script in this package works on it:

    <root>/manifest.json
        {"fs": 2000,
         "recordings": [
            {"emg": "rec0.npy",              # float32 (T, C), already filtered / resampled to fs
             "subject": "s01",
             "video": "rec0_video.npy",      # optional, float32 (Tv, Dv) frozen video-encoder latents
             "video_fps": 30.0,              # required with "video"
             "labels": "rec0_labels.npy"},   # optional, int (T,), per-sample class, <0 = ignore
            ...]}

All recordings in one manifest must share fs and channel count.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EMGWindows(Dataset):
    """Fixed-length windows (C, W) cut from memory-mapped recordings."""

    def __init__(self, manifest, window_s=2.0, stride_s=0.5, patch_ms=50,
                 use_video=False, use_labels=False, subjects=None):
        manifest = Path(manifest)
        self.root = manifest.parent
        meta = json.loads(manifest.read_text())
        self.fs = meta["fs"]
        self.use_video = use_video
        self.use_labels = use_labels

        self.patch_len = max(1, round(patch_ms / 1000 * self.fs))
        self.n_patches = round(window_s * 1000 / patch_ms)
        self.window = self.n_patches * self.patch_len
        stride = max(1, round(stride_s * self.fs))

        recs = [r for r in meta["recordings"] if subjects is None or r["subject"] in subjects]
        if not recs:
            raise ValueError(f"No recordings in {manifest} for subjects={subjects}")
        self.recs = recs
        self.subjects = sorted({r["subject"] for r in recs})

        self.starts, self.mean, self.std, self.labels = [], [], [], []
        self.n_channels = None
        self.video_dim = None
        self.num_classes = 0
        for r in recs:
            emg = np.load(self.root / r["emg"], mmap_mode="r")
            T, C = emg.shape
            if self.n_channels is None:
                self.n_channels = C
            assert C == self.n_channels, f"{r['emg']}: {C} channels, expected {self.n_channels}"

            # Per-recording, per-channel statistics from a strided subsample. Per-window
            # z-scoring would erase amplitude (i.e. force), so it is not used.
            sub = np.asarray(emg[:: max(1, T // 200_000)], dtype=np.float32)
            self.mean.append(sub.mean(0))
            self.std.append(sub.std(0) + 1e-6)

            starts = np.arange(max(0, (T - self.window) // stride + 1)) * stride
            labels = None
            if use_labels:
                labels = np.load(self.root / r["labels"]).astype(np.int64).reshape(-1)
                centre = labels[starts + self.window // 2]
                starts = starts[centre >= 0]
                if len(starts):
                    self.num_classes = max(self.num_classes, int(labels.max()) + 1)
            self.labels.append(labels)
            self.starts.append(starts)

            if use_video:
                self.video_dim = np.load(self.root / r["video"], mmap_mode="r").shape[1]
            del emg

        self.cum = np.cumsum([len(s) for s in self.starts])
        self._mm = {}

    def __getstate__(self):
        # memmaps must be reopened inside each DataLoader worker
        state = self.__dict__.copy()
        state["_mm"] = {}
        return state

    def _open(self, i, key):
        if (i, key) not in self._mm:
            self._mm[(i, key)] = np.load(self.root / self.recs[i][key], mmap_mode="r")
        return self._mm[(i, key)]

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, idx):
        i = int(np.searchsorted(self.cum, idx, side="right"))
        start = int(self.starts[i][idx - (self.cum[i - 1] if i else 0)])

        x = np.asarray(self._open(i, "emg")[start:start + self.window], dtype=np.float32)
        x = (x - self.mean[i]) / self.std[i]
        out = {"emg": torch.from_numpy(x.T.copy()),  # (C, W)
               "subject": self.subjects.index(self.recs[i]["subject"])}

        if self.use_labels:
            out["label"] = int(self.labels[i][start + self.window // 2])

        if self.use_video:
            # one video latent per EMG patch: the frame at the patch centre
            video = self._open(i, "video")
            t = (start + (np.arange(self.n_patches) + 0.5) * self.patch_len) / self.fs
            frames = np.minimum((t * self.recs[i]["video_fps"]).astype(np.int64), len(video) - 1)
            out["video"] = torch.from_numpy(np.asarray(video[frames], dtype=np.float32))  # (N, Dv)
        return out


def make_synthetic(root, n_subjects=4, minutes=2.0, fs=200, n_channels=8,
                   n_classes=5, video_dim=32, video_fps=30.0, segment_s=4, seed=0):
    """Write a small fake dataset in the manifest format, for smoke tests only.

    Each segment_s-second segment has a class; the class sets per-channel noise
    amplitude and the paired "video latent", so both objectives and the probe
    have real structure to find.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    gains = rng.uniform(0.3, 2.0, size=(n_classes, n_channels)).astype(np.float32)
    class_latent = rng.normal(size=(n_classes, video_dim)).astype(np.float32)

    recordings = []
    T = int(minutes * 60 * fs)
    for s in range(n_subjects):
        seg_len = segment_s * fs
        seg = rng.integers(0, n_classes, size=T // seg_len + 1)
        labels = np.repeat(seg, seg_len)[:T]
        subject_gain = rng.uniform(0.7, 1.3, size=n_channels).astype(np.float32)
        emg = rng.normal(size=(T, n_channels)).astype(np.float32) * gains[labels] * subject_gain

        frame_t = np.arange(int(T / fs * video_fps)) / video_fps
        frame_label = labels[np.minimum((frame_t * fs).astype(np.int64), T - 1)]
        video = class_latent[frame_label] + 0.1 * rng.normal(size=(len(frame_t), video_dim)).astype(np.float32)

        name = f"s{s + 1:02d}"
        np.save(root / f"{name}_emg.npy", emg)
        np.save(root / f"{name}_labels.npy", labels)
        np.save(root / f"{name}_video.npy", video.astype(np.float32))
        recordings.append({"emg": f"{name}_emg.npy", "subject": name,
                           "labels": f"{name}_labels.npy",
                           "video": f"{name}_video.npy", "video_fps": video_fps})

    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"fs": fs, "recordings": recordings}, indent=2))
    return manifest
