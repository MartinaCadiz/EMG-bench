"""Convert EMG-1 (Encord) into the manifest format read by Pretrain/data.py.

    # pretraining corpus: 3-channel "training" layer, each wrist stored as its own 3-channel recording
    python -m Pretrain.convert_emg1 3ch --src /data/emg1/emg1-3ch-v1.0.1 --out data/emg1_3ch --workers 32
    # evaluation corpus: 8-channel "benchmark" layer, with any-digit contact labels (task T1)
    python -m Pretrain.convert_emg1 8ch --src /data/emg1/emg1-8ch-v1.0.1 --out data/emg1_8ch --workers 32

Both datasets are written at FS_OUT = 2000 Hz, so an encoder pretrained on one can be probed on the other
(Pretrain/probe.py requires equal fs). Per recording, the pipeline is:
    select samples -> split at gaps / invalid samples -> 20 Hz high-pass + mains notches (+ anti-alias
    low-pass) at the native rate -> linear interpolation onto a uniform 2000 Hz grid -> float32 (T, C) .npy.
Amplitude is left in the release's units; data.py z-scores per recording and channel.

3ch: samples are kept inside index/segments.parquet quality segments (default: scope "both", tier GOLD or
STANDARD, i.e. the 271.8 h the release advertises), selected on the release's quality clock
(derived/sensors_est qc_t_ns_est). Saturated samples (|v| >= 0.99, +-250 ms) are cut, as the release
requires for STANDARD. There are no person ids in this release; "subject" is station_day, the official split
group, and one manifest is written per official split (manifest_train.json, _val, _test) plus manifest.json.
Pretrain on manifest_train.json only.

8ch: every take.h5 under takes/, subject = person (OP-001..OP-012), so leave-one-person-out probing works with
probe.py --val-subjects / --test-subjects. Labels are any-digit contact_est (1 contact, 0 no contact, -1 ignore)
on the band clock corrected by the release's measured band-mocap lag.

Resumable: each clip/take writes parts/<key>.json when done and is skipped on re-run; manifests are rebuilt
from parts/ at the end. Run with --limit 20 on the starter download first and check the printed summary.
"""

import argparse
import json
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

FS_OUT = 2000


# ---------------------------------------------------------------------------- signal helpers

def as_tc(a, n_ch, name):
    """Return a (T, C) view whatever the stored orientation."""
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == n_ch:
        return a
    if a.ndim == 2 and a.shape[0] == n_ch:
        return a.T
    raise ValueError(f"{name}: shape {a.shape}, expected (T, {n_ch}) or ({n_ch}, T)")


def dilate(mask, r):
    """True wherever a True lies within r samples."""
    if r <= 0 or not mask.any():
        return mask
    cs = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    i = np.arange(len(mask))
    return (cs[np.minimum(len(mask), i + r + 1)] - cs[np.maximum(0, i - r)]) > 0


def runs(valid, t_ns, max_gap_ns, min_len):
    """(start, stop) index pairs of contiguous valid samples without timestamp gaps."""
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return []
    brk = (np.diff(idx) > 1) | (np.diff(t_ns[idx]) > max_gap_ns) | (np.diff(t_ns[idx]) <= 0)
    edges = np.flatnonzero(brk) + 1
    out = []
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(idx)]):
        i0, i1 = int(idx[a]), int(idx[b - 1]) + 1
        if i1 - i0 >= min_len:
            out.append((i0, i1))
    return out


def clean_and_resample(x, t_ns, mains, hp_hz=20.0):
    """Filter one gap-free run at its native rate, then resample to FS_OUT. Returns (t0_ns, y float32 (T, C))."""
    tr = (t_ns - t_ns[0]).astype(np.float64)
    fs_in = 1e9 / np.median(np.diff(tr))
    y = np.asarray(x, dtype=np.float64)
    y = sosfiltfilt(butter(4, hp_hz, "highpass", fs=fs_in, output="sos"), y, axis=0)
    if mains:
        for f in np.arange(mains, min(fs_in / 2, 500.0), mains):
            b, a = iirnotch(f, 30.0, fs=fs_in)
            y = filtfilt(b, a, y, axis=0)
    if fs_in > FS_OUT:
        y = sosfiltfilt(butter(8, 0.45 * FS_OUT, "lowpass", fs=fs_in, output="sos"), y, axis=0)
    grid = np.arange(0.0, tr[-1], 1e9 / FS_OUT)
    out = np.stack([np.interp(grid, tr, y[:, c]) for c in range(y.shape[1])], axis=1)
    return int(t_ns[0]), grid, out.astype(np.float32)


def read_json_ds(f, path):
    if path not in f:
        return None
    v = f[path][()]
    if isinstance(v, np.ndarray):
        v = v.reshape(-1)[0]
    if isinstance(v, bytes):
        v = v.decode()
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return None


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


# ---------------------------------------------------------------------------- 3-channel

def _3ch_clip(job):
    clip, sensor_path, est_path, segs, out, opt = job
    recs, stats = [], {"samples_in": 0, "saturated": 0}
    with h5py.File(sensor_path, "r") as fs, h5py.File(est_path, "r") as fe:
        for side in ("left", "right"):
            side_segs = [s for s in segs if s["side"] == side]
            if not side_segs:
                continue
            g = f"bands/{side}/emg"
            if f"{g}/v" not in fs or f"{g}/t_ns_est" not in fe:
                stats[f"missing_{side}"] = 1
                continue
            v = as_tc(fs[f"{g}/v"][()], 3, f"{clip}:{g}/v")
            t = fe[f"{g}/t_ns_est"][()].astype(np.int64)
            qc = fe[f"{g}/qc_t_ns_est"][()] if f"{g}/qc_t_ns_est" in fe else t.astype(np.float64)
            assert len(t) == len(v) == len(qc), f"{clip} {side}: {len(v)} samples, {len(t)} t_ns_est, {len(qc)} qc"

            sel = np.zeros(len(v), bool)
            for s in side_segs:
                sel |= (qc >= s["start_t_ns"]) & (qc < s["end_t_ns"])
            stats["samples_in"] += int(sel.sum())
            fs_nom = 1e9 / np.median(np.diff(t[: min(len(t), 100_000)]))
            sat = dilate((np.abs(v) >= 0.99).any(1), int(0.25 * fs_nom))
            stats["saturated"] += int((sat & sel).sum())
            valid = sel & ~sat & np.isfinite(v).all(1)

            split = side_segs[0]["split"]
            for k, (i0, i1) in enumerate(runs(valid, t, 2.5e9 / fs_nom, int(opt["min_s"] * fs_nom))):
                t0, _, y = clean_and_resample(v[i0:i1], t[i0:i1], opt["mains"])
                name = f"emg/{clip}_{side}_{k:03d}.npy"
                np.save(out / name, y)
                recs.append({"emg": name, "subject": side_segs[0]["station_day"], "split": split,
                             "clip": clip, "wrist": side, "t0_ns": t0, "seconds": len(y) / FS_OUT})
    return recs, stats


def convert_3ch(args):
    import pandas as pd

    src, out = Path(args.src), Path(args.out)
    segs = pd.read_parquet(src / "index" / "segments.parquet")
    clips = pd.read_parquet(src / "index" / "clips.parquet").set_index("clip_id")
    tiers = [t.strip().upper() for t in args.tiers.split(",")]
    segs = segs[segs["tier"].astype(str).str.upper().isin(tiers)]
    segs = segs[segs["scope"] == "both"] if args.scope == "both" else segs[segs["scope"].isin(["left", "right"])]
    if "physical_side_quarantined" in segs:
        segs = segs[~segs["physical_side_quarantined"].fillna(False).astype(bool)]
    print(f"{len(segs)} segments, {segs['hours'].sum():.1f} h listed (scope={args.scope}, tiers={tiers})")

    def field(row, key, default):
        v = row.get(key, None) if hasattr(row, "get") else None
        return default if v is None or (isinstance(v, float) and np.isnan(v)) or v == "" else v

    jobs = []
    for clip, g in segs.groupby("clip_id"):
        if args.limit and len(jobs) >= args.limit:
            break
        c = clips.loc[clip] if clip in clips.index else {}
        sp = src / field(c, "sensor_path", f"sensors/{clip}.h5")
        ep = src / field(c, "derived_sensor_path", f"derived/sensors_est/{clip}_est.h5")
        if not sp.exists() or not ep.exists():
            continue  # not downloaded (e.g. starter subset)
        rows = []
        for r in g.to_dict("records"):
            sides = ["left", "right"] if r["scope"] == "both" else [r["scope"]]
            if bool(field(r, "failing_band_1325482500024", False)):
                sides = [s for s in sides if s != "left"]  # degraded left band on egox-017
            for side in sides:
                rows.append({"side": side, "start_t_ns": float(r["start_t_ns"]), "end_t_ns": float(r["end_t_ns"]),
                             "split": str(field(r, "split", field(c, "split", "unknown"))),
                             "station_day": str(field(r, "station_day", field(c, "station_day", clip)))})
        jobs.append((clip, sp, ep, rows))
    print(f"{len(jobs)} clips present under {src}")
    run_jobs(_3ch_clip, jobs, out, args)

    recs = gather(out)
    for split in sorted({r["split"] for r in recs}):
        write_manifest(out / f"manifest_{split}.json", [r for r in recs if r["split"] == split])
    write_manifest(out / "manifest.json", recs)
    done = segs[segs["clip_id"].isin({r["clip"] for r in recs})]
    expected = done.groupby("split")["hours"].sum() * (2 if args.scope == "both" else 1)
    for split in sorted({r["split"] for r in recs}):
        got = sum(r["seconds"] for r in recs if r["split"] == split) / 3600
        print(f"  {split:6s} {got:9.2f} wrist-hours written / {expected.get(split, float('nan')):9.2f} listed")


# ---------------------------------------------------------------------------- 8-channel

def _contact_labels(f, t_band_ns):
    """Any-digit contact per band sample: 1 if any digit is in contact, 0 if all digits are clear, else -1."""
    base = "derived/proximity_est"
    if base not in f:
        return None, "no proximity_est"
    lag = read_json_ds(f, "derived/band_mocap_lag_est")
    lag_ms = find_key(lag, "applied_lag_ms") if lag else None
    status = find_key(lag, "status") if lag else None
    shift = int(round(float(lag_ms) * 1e6)) if lag_ms is not None and status in (None, "measured") else 0
    t_q = t_band_ns - shift  # band sample stamped t corresponds to mocap at t - lag

    any_c = np.zeros(len(t_q), bool)
    all_clear = np.ones(len(t_q), bool)
    n = 0
    for digit in f[base]:
        g = f[f"{base}/{digit}"]
        if "contact_est" not in g:
            continue
        c = np.asarray(g["contact_est"][()]).reshape(-1).astype(np.int64)
        tm = g["t_ns"][()] if "t_ns" in g else f["markers/t_ns"][()] if "markers/t_ns" in f else None
        if tm is None or len(tm) != len(c):
            return None, f"{digit}: no timestamps matching {len(c)} contact_est rows"
        tm = tm.astype(np.int64)
        j = np.clip(np.searchsorted(tm, t_q), 1, len(tm) - 1)
        j = np.where(np.abs(tm[j - 1] - t_q) < np.abs(tm[j] - t_q), j - 1, j)
        near = np.abs(tm[j] - t_q) <= 1e9 / 120 / 2
        cj = np.where(near, c[j], -1)
        any_c |= cj == 1
        all_clear &= cj == 0
        n += 1
    if n == 0:
        return None, "no contact_est"
    return np.where(any_c, 1, np.where(all_clear, 0, -1)).astype(np.int8), f"lag {shift / 1e6:.1f} ms"


def _8ch_take(job):
    key, path, out, opt = job
    person = key.split("/")[0]
    recs, stats = [], {"samples_in": 0}
    with h5py.File(path, "r") as f:
        v = as_tc(f["emg/data"][()], 8, f"{key}:emg/data")
        t = f["emg/t_ns"][()].astype(np.int64)
        valid = np.isfinite(v).all(1)
        if "emg/finite" in f:
            fin = np.asarray(f["emg/finite"][()])
            valid &= fin.all(1) if fin.ndim == 2 else fin.reshape(-1).astype(bool)
        stats["samples_in"] = len(v)
        fs_nom = 1e9 / np.median(np.diff(t[: min(len(t), 100_000)]))
        for k, (i0, i1) in enumerate(runs(valid, t, 1.5e9 / fs_nom, int(opt["min_s"] * fs_nom))):
            t0, grid, y = clean_and_resample(v[i0:i1], t[i0:i1], opt["mains"])
            stem = f"{key.replace('/', '_')}_{k:02d}"
            rec = {"emg": f"emg/{stem}.npy", "subject": person, "take": key, "t0_ns": t0,
                   "seconds": len(y) / FS_OUT}
            labels, note = _contact_labels(f, t0 + np.round(grid).astype(np.int64))
            stats["label_note"] = note
            if labels is None:
                labels = np.full(len(y), -1, np.int8)
            np.save(out / rec["emg"], y)
            np.save(out / f"labels/{stem}.npy", labels)
            rec["labels"] = f"labels/{stem}.npy"
            recs.append(rec)
    return recs, stats


def convert_8ch(args):
    src, out = Path(args.src), Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in sorted((src / "takes").glob("*/*/*/take.h5")):
        if args.limit and len(jobs) >= args.limit:
            break
        jobs.append(("/".join(p.parts[-4:-1]), p))
    print(f"{len(jobs)} takes under {src / 'takes'}")
    run_jobs(_8ch_take, jobs, out, args)
    recs = gather(out)
    write_manifest(out / "manifest.json", recs)
    for person in sorted({r["subject"] for r in recs}):
        print(f"  {person}  {sum(r['seconds'] for r in recs if r['subject'] == person) / 3600:6.2f} h")


# ---------------------------------------------------------------------------- driver

def _safe(fn_job):
    fn, job = fn_job
    try:
        return job[0], fn(job), None
    except Exception:
        return job[0], None, traceback.format_exc()


def run_jobs(fn, jobs, out, args):
    (out / "emg").mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    opt = {"mains": args.mains, "min_s": args.min_s}
    part = lambda key: out / "parts" / (key.replace("/", "_") + ".json")
    todo = [(fn, (j[0], *j[1:], out, opt)) for j in jobs if args.overwrite or not part(j[0]).exists()]
    print(f"{len(jobs) - len(todo)} already converted, {len(todo)} to do")
    failed = 0
    with Pool(args.workers) as pool:
        for n, (key, res, err) in enumerate(pool.imap_unordered(_safe, todo), 1):
            if err:
                failed += 1
                print(f"FAILED {key}\n{err}", file=sys.stderr)
                continue
            recs, stats = res
            part(key).write_text(json.dumps({"recordings": recs, "stats": stats}))
            if n % 50 == 0 or n == len(todo) or args.limit:
                print(f"[{n}/{len(todo)}] {key}: {len(recs)} recordings, {stats}")
    if failed:
        print(f"{failed} failed; fix and re-run (finished ones are skipped)", file=sys.stderr)


def gather(out):
    recs = []
    for p in sorted((out / "parts").glob("*.json")):
        recs += json.loads(p.read_text())["recordings"]
    return recs


def write_manifest(path, recs):
    path.write_text(json.dumps({"fs": FS_OUT, "recordings": recs}, indent=1))
    print(f"wrote {path}: {len(recs)} recordings, {sum(r['seconds'] for r in recs) / 3600:.2f} h, "
          f"{len({r['subject'] for r in recs})} subjects")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["3ch", "8ch"])
    ap.add_argument("--src", required=True, help="downloaded release folder (emg1-3ch-v1.0.1 or emg1-8ch-v1.0.1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N clips/takes (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="redo clips/takes that already have parts/*.json")
    ap.add_argument("--mains", type=float, default=50.0, help="mains frequency to notch with harmonics; 0 = off")
    ap.add_argument("--min-s", type=float, default=2.5, help="drop gap-free runs shorter than this")
    ap.add_argument("--tiers", default="GOLD,STANDARD", help="3ch: segment tiers to keep")
    ap.add_argument("--scope", choices=["both", "per-wrist"], default="both",
                    help="3ch: 'both' = both-wrist segments (the 271.8 h); 'per-wrist' = each wrist's own segments")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    convert_3ch(args) if args.dataset == "3ch" else convert_8ch(args)


if __name__ == "__main__":
    main()
