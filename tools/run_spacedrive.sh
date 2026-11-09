#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=/home/qh/miniconda3/envs/SpaceDrive/bin/python
export PATH="/home/qh/miniconda3/envs/SpaceDrive/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/outputs/environment_setup/matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$MPLCONFIGDIR"
cd "$PROJECT_DIR"

MODE="${1:-check}"
if (( $# )); then shift; fi
case "$MODE" in
    check)
        exec taskset -c 0-3,6-31 "$PYTHON_BIN" tools/check_environment.py "$@"
        ;;
    train|test)
        exec taskset -c 0-3,6-31 "$PYTHON_BIN" -m torch.distributed.run \
            --standalone --nnodes=1 --nproc-per-node="${SPACEDRIVE_GPUS:-1}" \
            "scripts/$MODE/$MODE.py" "$@" --launcher pytorch
        ;;
    python)
        exec taskset -c 0-3,6-31 "$PYTHON_BIN" "$@"
        ;;
    *)
        echo "Usage: bash tools/run_spacedrive.sh {check|train|test|python} [arguments...]" >&2
        exit 2
        ;;
esac
