#!/usr/bin/env bash
# EMG-1 (Encord) for JEPA pretraining + evaluation.
#   3-channel "training" layer (297.6 GB): band sensors + clocks for all 4,916 clips, no video -> pretraining
#   8-channel "benchmark"     ( 32.1 GB): take.h5 + derived/ for all 920 takes, no video  -> evaluation
#
# Usage (run inside tmux/screen on EIDF; re-running resumes an interrupted download):
#   bash get_EMG1.sh starter [DEST]   # ~2.8 GB: inspect the file formats first
#   bash get_EMG1.sh all     [DEST]   # ~330 GB: the two subsets above
# DEST defaults to the current directory. Put it on large storage, not your home quota.
set -euo pipefail

MODE="${1:-starter}"
DEST="$(realpath "${2:-.}")"
WORKERS="${WORKERS:-16}"
mkdir -p "$DEST"
cd "$DEST"

if [ ! -d EMG-1 ]; then
    git clone https://github.com/encord-team/EMG-1.git
fi

dl() {  # dl DATASET SUBSET OUTDIR
    echo "Retrieving EMG-1 $1 ($2) -> $DEST/$3 ..."
    python3 EMG-1/scripts/download.py "$1" "$2" --dest "$DEST/$3" --workers "$WORKERS"
}

case "$MODE" in
    starter)
        dl 3ch-ego   starter emg1-3ch-v1.0.1
        dl 8ch-mocap starter emg1-8ch-v1.0.1
        ;;
    all)
        dl 8ch-mocap benchmark emg1-8ch-v1.0.1
        dl 3ch-ego   training  emg1-3ch-v1.0.1
        ;;
    *)
        echo "unknown mode '$MODE' (use: starter | all)" >&2
        exit 1
        ;;
esac

df -h "$DEST"
echo "Done retrieving EMG-1 ($MODE)."
