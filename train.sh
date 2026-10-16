#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
echo "Project Root: $PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/RL/models"
mkdir -p "$PROJECT_ROOT/logs"

RUN_ID="${WANDB_RUN_ID:-}"
MODEL_PATH="${MODEL_PATH:-}"
DATASET_PATH="${DATASET_PATH:-}"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id|-r)
            RUN_ID="$2"
            shift 2
            ;;
        --run-id=*)
            RUN_ID="${1#*=}"
            shift
            ;;
        --model-path|--checkpoint|--model|-m)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-path=*|--checkpoint=*|--model=*)
            MODEL_PATH="${1#*=}"
            shift
            ;;
        --dataset-path|--dataset|-d)
            DATASET_PATH="$2"
            shift 2
            ;;
        --dataset-path=*|--dataset=*)
            DATASET_PATH="${1#*=}"
            shift
            ;;
        *)
            if [[ "$1" == *.pt ]]; then
                MODEL_PATH="$1"
            elif [ -z "$RUN_ID" ] && [[ "$1" != -* ]]; then
                RUN_ID="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [ -n "$RUN_ID" ]; then
    export WANDB_RUN_ID="$RUN_ID"
    echo ">>> [WandB] RESUME logging into Run ID: $WANDB_RUN_ID"
else
    echo ">>> [WandB] Init new Run"
fi

if [ -n "$MODEL_PATH" ]; then
    export MODEL_PATH="$MODEL_PATH"
    echo ">>> [Model] Load model from: $MODEL_PATH"
else
    echo ">>> [Model] Load model from RL/models/ (if exists)"
fi

if [ -n "$DATASET_PATH" ]; then
    export DATASET_PATH="$DATASET_PATH"
    echo ">>> [Dataset] Dataset path: $DATASET_PATH"
else
    echo ">>> [Dataset] Default dataset: levels/training_dataset"
fi

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/RL:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/logs/train_${TIMESTAMP}.log"

echo "=== Starting Train ==="
python "$PROJECT_ROOT/RL/training.py" \
    ${MODEL_PATH:+--model-path "$MODEL_PATH"} \
    ${RUN_ID:+--run-id "$RUN_ID"} \
    ${DATASET_PATH:+--dataset-path "$DATASET_PATH"} \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
