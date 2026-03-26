#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=lmarena-scoring
#SBATCH --output=/net/scratch2/listar2000/prediction-powered-adaptive-shrinkage/outputs/lmarena/slurm-%j.log
#SBATCH --error=/net/scratch2/listar2000/prediction-powered-adaptive-shrinkage/outputs/lmarena/slurm-%j.err

# ===========================================================================
# Unified LMArena Scoring Pipeline
#
# Starts the SGLang reward model server, runs pairwise scoring, and
# postprocesses results into clean_summary.csv -- all in a single SLURM job.
#
# Usage:
#   sbatch data/lmarena/run_scoring_pipeline.sh
#
# Override defaults via environment variables, e.g.:
#   PREDICTION_MODE=binary TEMPERATURE=2.0 sbatch data/lmarena/run_scoring_pipeline.sh
# ===========================================================================

set -euo pipefail

# ---- Configurable variables (override via env) ----------------------------
SERVER_CONFIG="${SERVER_CONFIG:-data/lmarena/skywork_server_args.yaml}"
PAIRS_CSV="${PAIRS_CSV:-data/lmarena/sample_data/llm_pair_summary_single_turn.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-data/lmarena/results_skywork}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-data/lmarena/clean_data/clean_summary.csv}"
PREDICTION_MODE="${PREDICTION_MODE:-bradley-terry}"
TEMPERATURE="${TEMPERATURE:-1.0}"
PORT="${PORT:-30000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-128}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
EXTRA_ARGS="${EXTRA_ARGS:---single-turn-only}"

# ---- Resolve repo root (parent of data/) ----------------------------------
REPO_ROOT="/net/scratch2/listar2000/prediction-powered-adaptive-shrinkage"
cd "$REPO_ROOT"

echo "=============================================="
echo "LMArena Scoring Pipeline"
echo "=============================================="
echo "Repo root      : $REPO_ROOT"
echo "Pairs CSV      : $PAIRS_CSV"
echo "Output dir     : $OUTPUT_DIR"
echo "Prediction mode: $PREDICTION_MODE"
echo "Temperature    : $TEMPERATURE"
echo "Port           : $PORT"
echo "SLURM Job ID   : ${SLURM_JOB_ID:-N/A}"
echo "Node           : $(hostname)"
echo "GPUs           : ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "=============================================="

# ---- Activate environment -------------------------------------------
echo "Activating environment..."
source /net/scratch2/listar2000/prophet-hindsight/.venv/bin/activate
export HF_HOME=".cache/huggingface"
echo "Environment activated."

# ---- Cleanup trap ---------------------------------------------------------
SERVER_PID=""
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping SGLang server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "Done."
}
trap cleanup EXIT ERR

# ---- Start SGLang server --------------------------------------------------
echo ""
echo "[Step 1/3] Starting SGLang server..."
python -m sglang.launch_server --config "$SERVER_CONFIG" &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# ---- Wait for server to be ready -----------------------------------------
echo "Waiting for server health check at http://localhost:$PORT/health ..."
elapsed=0
backoff=1
while [ "$elapsed" -lt "$HEALTH_TIMEOUT" ]; do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server is ready! (took ${elapsed}s)"
        break
    fi
    # Check that the server process is still alive
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: SGLang server process died unexpectedly."
        exit 1
    fi
    sleep "$backoff"
    elapsed=$((elapsed + backoff))
    # Exponential backoff capped at 30s
    backoff=$((backoff * 2))
    if [ "$backoff" -gt 30 ]; then
        backoff=30
    fi
done

if [ "$elapsed" -ge "$HEALTH_TIMEOUT" ]; then
    echo "ERROR: Server did not become ready within ${HEALTH_TIMEOUT}s."
    exit 1
fi

# ---- Run scoring ----------------------------------------------------------
echo ""
echo "[Step 2/3] Running reward model scoring..."
python data/lmarena/run_judge_skywork.py \
    --pairs-csv "$PAIRS_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --prediction-mode "$PREDICTION_MODE" \
    --temperature "$TEMPERATURE" \
    --port "$PORT" \
    --max-concurrent "$MAX_CONCURRENT" \
    $EXTRA_ARGS

# ---- Run postprocessing ---------------------------------------------------
echo ""
echo "[Step 3/3] Postprocessing results..."
mkdir -p "$(dirname "$CLEAN_OUTPUT")"
python data/lmarena/convert_summary.py \
    --input "$OUTPUT_DIR/summary.csv" \
    --output "$CLEAN_OUTPUT"

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "  Summary  : $OUTPUT_DIR/summary.csv"
echo "  Clean CSV: $CLEAN_OUTPUT"
echo "=============================================="
