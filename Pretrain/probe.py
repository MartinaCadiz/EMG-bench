"""Evaluate a pretrained encoder on a labelled dataset in the manifest format.

    python -m Pretrain.probe --ckpt runs/emg1_jepa/last.pt --manifest data/db5/manifest.json \
        --val-subjects s08 --test-subjects s09,s10 --mode linear

--mode linear   : frozen encoder, train a linear head on mean-pooled tokens
--mode finetune : train encoder and head
--scratch       : same architecture with random weights (the baseline pretraining must beat)

Subjects are split three ways. The epoch is chosen on the validation subjects
and accuracy is reported on the test subjects, which are never used for selection.
"""

import argparse
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import EMGWindows
from .model import EMGEncoder


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes, freeze):
        super().__init__()
        self.encoder = encoder
        self.freeze = freeze
        self.head = nn.Sequential(nn.BatchNorm1d(encoder.dim, affine=False), nn.Linear(encoder.dim, num_classes))
        if freeze:
            for p in encoder.parameters():
                p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def forward(self, x):
        with torch.set_grad_enabled(not self.freeze and self.training):
            feat = self.encoder(x).mean(1)
        return self.head(feat)


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        pred = model(batch["emg"].to(device)).argmax(1).cpu()
        correct += (pred == batch["label"]).sum().item()
        total += len(pred)
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val-subjects", required=True, help="comma-separated subject ids")
    ap.add_argument("--test-subjects", required=True, help="comma-separated subject ids")
    ap.add_argument("--mode", choices=["linear", "finetune"], default="linear")
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["cfg"]
    m = cfg["model"]

    val_s, test_s = args.val_subjects.split(","), args.test_subjects.split(",")
    assert not set(val_s) & set(test_s), "validation and test subjects overlap"
    every = EMGWindows(args.manifest, use_labels=True, **{k: cfg["data"][k] for k in ("window_s", "stride_s", "patch_ms")})
    train_s = [s for s in every.subjects if s not in val_s + test_s]
    assert every.fs == ckpt["fs"], f"dataset fs={every.fs} but encoder was pretrained at fs={ckpt['fs']}; resample first"

    def split(subjects, shuffle):
        ds = EMGWindows(args.manifest, cfg["data"]["window_s"], cfg["data"]["stride_s"], cfg["data"]["patch_ms"],
                        use_labels=True, subjects=subjects)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)

    train, val, test = split(train_s, True), split(val_s, False), split(test_s, False)
    print(f"train {train_s} ({len(train.dataset)})  val {val_s} ({len(val.dataset)})  "
          f"test {test_s} ({len(test.dataset)})  classes {every.num_classes}")

    encoder = EMGEncoder(ckpt["patch_len"], m["dim"], m["depth"], m["heads"], m["max_channels"])
    if not args.scratch:
        # the EMA target encoder is the one used downstream, as in I-JEPA
        state = {k[len("target_encoder."):]: v for k, v in ckpt["model"].items() if k.startswith("target_encoder.")}
        encoder.load_state_dict(state)
    model = Classifier(encoder, every.num_classes, freeze=args.mode == "linear").to(args.device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_val, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        for batch in train:
            loss = nn.functional.cross_entropy(model(batch["emg"].to(args.device)), batch["label"].to(args.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        v = accuracy(model, val, args.device)
        print(f"epoch {epoch + 1}  loss {loss.item():.4f}  val_acc {v:.4f}")
        if v > best_val:
            best_val, best_state = v, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"RESULT mode={args.mode} scratch={args.scratch} seed={args.seed} "
          f"val_acc={best_val:.4f} test_acc={accuracy(model, test, args.device):.4f}")


if __name__ == "__main__":
    main()
