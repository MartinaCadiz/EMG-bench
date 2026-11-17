"""Self-supervised JEPA pretraining on raw EMG.

    python -m Pretrain.train --config Pretrain/configs/emg1_jepa.yaml
    python -m Pretrain.train --config Pretrain/configs/emg1_jepa.yaml --set loss.video_weight=1.0 out=runs/video

Writes <out>/last.pt after every epoch (resume with --resume) and <out>/log.jsonl.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import EMGWindows
from .model import EMGJEPA


def load_config(path, overrides):
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"unknown config key: {key}")
        node[leaf] = yaml.safe_load(value)
    return cfg


def build_model(cfg, patch_len, video_dim=None):
    return EMGJEPA(patch_len=patch_len, video_dim=video_dim, **cfg["model"],
                   n_blocks=cfg["mask"]["n_blocks"], block_len=cfg["mask"]["block_len"],
                   drop_channels=cfg["mask"]["drop_channels"],
                   self_weight=cfg["loss"]["self_weight"], video_weight=cfg["loss"]["video_weight"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    opt_cfg = cfg["optim"]
    torch.manual_seed(opt_cfg["seed"])
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)

    use_video = cfg["loss"]["video_weight"] > 0
    data = EMGWindows(cfg["data"]["manifest"], cfg["data"]["window_s"], cfg["data"]["stride_s"],
                      cfg["data"]["patch_ms"], use_video=use_video, subjects=cfg["data"]["subjects"])
    loader = DataLoader(data, batch_size=opt_cfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=opt_cfg["num_workers"], pin_memory=args.device.startswith("cuda"),
                        persistent_workers=opt_cfg["num_workers"] > 0)
    if len(loader) == 0:
        raise ValueError(f"{len(data)} windows is fewer than one batch of {opt_cfg['batch_size']}")
    print(f"{len(data)} windows, {data.n_channels} channels, {len(data.subjects)} subjects, "
          f"fs={data.fs}, patch={data.patch_len} samples, {data.n_patches} patches/window")

    model = build_model(cfg, data.patch_len, data.video_dim).to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    amp = opt_cfg["amp"] and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=amp)

    start_epoch = 0
    ckpt_path = out / "last.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"]
        print(f"resumed from epoch {start_epoch}")

    total = opt_cfg["epochs"] * len(loader)
    warmup = opt_cfg["warmup_epochs"] * len(loader)

    def lr_at(step):
        if step < warmup:
            return opt_cfg["lr"] * (step + 1) / warmup
        return opt_cfg["lr"] * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    def ema_at(step):
        return opt_cfg["ema_start"] + (opt_cfg["ema_end"] - opt_cfg["ema_start"]) * step / total

    for epoch in range(start_epoch, opt_cfg["epochs"]):
        model.train()
        sums, n = {}, 0
        for i, batch in enumerate(loader):
            step = epoch * len(loader) + i
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x = batch["emg"].to(args.device, non_blocking=True)
            video = batch["video"].to(args.device, non_blocking=True) if use_video else None

            with torch.autocast(args.device.split(":")[0], enabled=amp):
                res = model(x, video)
            opt.zero_grad(set_to_none=True)
            scaler.scale(res["loss"]).backward()
            scaler.step(opt)
            scaler.update()
            model.update_target(ema_at(step))

            for k, v in res.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n += 1

        row = {"epoch": epoch + 1, "lr": lr_at(step), **{k: v / n for k, v in sums.items()}}
        print("  ".join(f"{k}={v:.4g}" for k, v in row.items()))
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1,
                    "cfg": cfg, "patch_len": data.patch_len, "fs": data.fs}, ckpt_path)


if __name__ == "__main__":
    main()
