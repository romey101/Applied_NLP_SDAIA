#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SAWT_ROOT="$PWD"
export SAWT_LLM="${SAWT_LLM:-qwen2.5:3b}"
export SAWT_ALLOW_DOWNLOAD="${SAWT_ALLOW_DOWNLOAD:-1}"
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false
[[ -x .venv/bin/python && -d frontend/node_modules ]] || { echo 'Run bash setup.sh first'; exit 1; }
for f in masked.csv clusters.csv cluster_labels.csv embeddings.npy feedback.faiss entities.csv sentiment_metrics.csv; do
 [[ -f "data/processed/$f" ]] || { echo "Missing data/processed/$f; restore your private artifacts"; exit 1; }
done
[[ -f models/selected_model.json && -f models/sentiment.joblib && -d models/distilbert ]] || { echo 'Restore models/ from your trained project'; exit 1; }
pids=()
cleanup(){ for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT TERM
if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; then
 ollama serve & pids+=("$!")
 for i in {1..30}; do curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
fi
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null
(cd backend && exec ../.venv/bin/python -X faulthandler -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1) & pids+=("$!")
(cd frontend && exec npm run dev -- --strictPort) & pids+=("$!")
echo 'Sawt: http://127.0.0.1:5173 — Ctrl+C to stop'
while true; do for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || exit 1; done; sleep 2; done
