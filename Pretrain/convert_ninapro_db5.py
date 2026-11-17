"""Convert Ninapro DB5 (.mat, as fetched by Setup/Get_Datasets/get_NinaproDB5.sh)
into the manifest format read by Pretrain.data.EMGWindows.

    python -m Pretrain.convert_ninapro_db5 --src NinaproDB5 --out data/db5
    python -m Pretrain.convert_ninapro_db5 --src NinaproDB5 --out data/db5_e2 --exercises 2 --no-rest

Input : <src>/s{1..10}/S{n}_E{1,2,3}_A1.mat with 'emg' (T, 16) at 200 Hz and
        'restimulus' (T, 1), 0 = rest, 1..N = gesture within that exercise.
Output: one recording per subject per exercise, labels merged into one global
        class index over the selected exercises:
          with rest : 0 = rest, gestures 1..K
          --no-rest : gestures 0..K-1, rest samples = -1 (ignored by EMGWindows)

DB5 is two 8-channel Myo armbands, already band-limited by the device; no extra
filtering is applied. Use --target-fs to resample, e.g. to match EMG-1, because
an encoder only transfers to data at the sample rate it was pretrained on.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

FS = 200
N_GESTURES = {1: 12, 2: 17, 3: 23}  # without rest, as in Setup/Utils/utils_NinaproDB5.py


def find_mat(src, subject, exercise):
    for d in (f"s{subject}", f"S{subject}"):
        for name in (f"S{subject}_E{exercise}_A1.mat", f"s{subject}_E{exercise}_A1.mat"):
            p = Path(src) / d / name
            if p.exists():
                return p
    raise FileNotFoundError(f"no file for subject {subject} exercise {exercise} under {src}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="NinaproDB5")
    ap.add_argument("--out", default="data/db5")
    ap.add_argument("--exercises", default="1,2,3", help="comma-separated, from 1,2,3")
    ap.add_argument("--subjects", default="1-10", help="e.g. 1-10 or 1,2,5")
    ap.add_argument("--no-rest", action="store_true", help="drop rest windows instead of making rest a class")
    ap.add_argument("--target-fs", type=int, default=FS)
    args = ap.parse_args()

    exercises = [int(e) for e in args.exercises.split(",")]
    assert all(e in N_GESTURES for e in exercises), f"exercises must be from {sorted(N_GESTURES)}"
    if "-" in args.subjects:
        a, b = map(int, args.subjects.split("-"))
        subjects = list(range(a, b + 1))
    else:
        subjects = [int(s) for s in args.subjects.split(",")]

    # global class offset of each selected exercise, so labels stay contiguous
    offsets, k = {}, 0
    for e in exercises:
        offsets[e] = k
        k += N_GESTURES[e]
    n_classes = k + (0 if args.no_rest else 1)

    ratio = Fraction(args.target_fs, FS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recordings, counts = [], np.zeros(n_classes, dtype=np.int64)

    for s in subjects:
        for e in exercises:
            mat = loadmat(find_mat(args.src, s, e))
            emg = np.asarray(mat["emg"], dtype=np.float32)
            restim = np.asarray(mat["restimulus"]).reshape(-1).astype(np.int64)
            assert len(emg) == len(restim), f"S{s} E{e}: emg {len(emg)} vs restimulus {len(restim)} samples"
            assert restim.min() >= 0 and restim.max() <= N_GESTURES[e], \
                f"S{s} E{e}: restimulus range {restim.min()}..{restim.max()}, expected 0..{N_GESTURES[e]}"

            gesture = restim > 0
            labels = np.full_like(restim, -1)
            labels[gesture] = offsets[e] + restim[gesture] - (1 if args.no_rest else 0)
            if not args.no_rest:
                labels[~gesture] = 0

            if ratio != 1:
                emg = resample_poly(emg, ratio.numerator, ratio.denominator, axis=0).astype(np.float32)
                idx = np.minimum((np.arange(len(emg)) / float(ratio)).astype(np.int64), len(labels) - 1)
                labels = labels[idx]

            name = f"s{s:02d}_e{e}"
            np.save(out / f"{name}_emg.npy", emg)
            np.save(out / f"{name}_labels.npy", labels)
            recordings.append({"emg": f"{name}_emg.npy", "labels": f"{name}_labels.npy",
                               "subject": f"s{s:02d}", "exercise": e})
            counts += np.bincount(labels[labels >= 0], minlength=n_classes)
            print(f"S{s} E{e}: {emg.shape[0]} samples x {emg.shape[1]} channels")

    meta = {"fs": args.target_fs, "dataset": "ninapro-db5", "n_classes": n_classes,
            "rest_class": None if args.no_rest else 0, "exercises": exercises,
            "recordings": recordings}
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out / 'manifest.json'}: {len(recordings)} recordings, {n_classes} classes")
    print("samples per class:", counts.tolist())


if __name__ == "__main__":
    main()
