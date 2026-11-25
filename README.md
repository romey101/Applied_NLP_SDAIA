
## Run locally

Follow these instructions to download and run Sawt on your local machine. No model retraining or notebook execution is required when the precomputed artifacts are available.

### 1. Prerequisites

Make sure the following tools are installed:

| Requirement | Recommended version | Download |
|---|---|---|
| Python | 3.12 | [Download Python](https://www.python.org/downloads/) |
| Node.js | 22 | [Download Node.js](https://nodejs.org/) |
| Git | Latest stable | [Download Git](https://git-scm.com/downloads) |
| Ollama | Latest stable | [Download Ollama](https://ollama.com/download) |

A machine with at least 16 GB of RAM is recommended for local AI inference.

### 2. Download the repository

Clone the repository:

```bash
git clone https://github.com/romey101/Applied_NLP_SDAIA.git
cd Applied_NLP_SDAIA
```

Alternatively, download the repository using GitHub's **Code → Download ZIP** option, extract it, and open a terminal inside the extracted directory.

### 3. Prepare the data and model artifacts

Sawt uses precomputed NLP artifacts and trained models to avoid retraining during application startup.

Make sure the following directories are available:

```text
Applied_NLP_SDAIA/
├── data/
│   └── processed/
│       ├── masked.csv
│       ├── clusters.csv
│       ├── cluster_labels.csv
│       ├── embeddings.npy
│       ├── feedback.faiss
│       ├── entities.csv
│       └── sentiment_metrics.csv
└── models/
    ├── sentiment.joblib
    ├── distilbert/
    └── selected_model.json
```

If these artifacts are not included in the downloaded repository, obtain the complete artifact package from the repository maintainer and extract it into the project root.

Keep all artifacts from the same notebook execution together.

The notebooks are provided for reproducibility and experimentation. They do not need to be executed when the required artifacts are available.

### 4. Start Ollama

Open a terminal and run:

```bash
ollama serve
```

If Ollama is already running, continue to the next step.

Download the default language model:

```bash
ollama pull qwen2.5:3b
```

### 5. Install project dependencies

From the project root, run:

```bash
bash setup.sh
```

The setup script prepares the Python virtual environment and installs the backend and frontend dependencies.

This step is required only during the initial installation or when dependencies change.

### 6. Start Sawt

From the project root, run:

```bash
bash start.sh
```

Open the application in your browser:

| Service | URL |
|---|---|
| Sawt Studio | http://127.0.0.1:5173 |
| Backend API | http://127.0.0.1:8000 |
| Interactive API documentation | http://127.0.0.1:8000/docs |
| System health | http://127.0.0.1:8000/api/health |

The first semantic search may require downloading pretrained Hugging Face models. An internet connection is recommended during the initial setup.

---

### Manual startup

If the automated startup script is unavailable or fails, the services can be started manually using three separate terminals.

#### Terminal 1 — Ollama

```bash
ollama serve
```

#### Terminal 2 — Backend

Navigate to the project root, then run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

export SAWT_ROOT="$(pwd)"
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

cd backend

python -X faulthandler -m uvicorn main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 1
```

The thread limits reduce native-library concurrency and help improve inference stability, particularly on Apple Silicon devices.

#### Terminal 3 — Frontend

Navigate to the project root, then run:

```bash
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173 in your browser.

---

### Optional: Change the language model

Sawt supports local language models through Ollama.

| Model | Ollama identifier |
|---|---|
| Qwen 2.5 | `qwen2.5:3b` |
| Llama 3.1 | `llama3.1:8b` |
| Gemma 3 | `gemma3:4b` |
| Aya Expanse | `aya-expanse:8b` |

For example, to use Llama:

```bash
ollama pull llama3.1:8b
SAWT_LLM=llama3.1:8b bash start.sh
```

Or to use Gemma:

```bash
ollama pull gemma3:4b
SAWT_LLM=gemma3:4b bash start.sh
```

Alternative models may require additional memory and have not been fully evaluated on the Sawt dataset. Arabic generation quality may vary between models.

Changing the generative model does not require retraining the sentiment classifier, regenerating embeddings, or rebuilding the FAISS index.

---

### Verify the installation

Check the backend health:

```bash
curl http://127.0.0.1:8000/api/health
```

Test review retrieval:

```bash
curl "http://127.0.0.1:8000/api/reviews?q=&offset=0&limit=15"
```

Test semantic search:

```bash
curl -X POST http://127.0.0.1:8000/api/search \
    -H "Content-Type: application/json" \
    -d '{"text":"جودة المنتج","rerank":true}'
```

Test retrieval-augmented generation:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"text":"ما هي أبرز شكاوى العملاء؟","rerank":true}'
```

---

### Troubleshooting

| Issue | Recommended solution |
|---|---|
| Missing artifact error | Verify that `data/processed/` and `models/` contain the required files. |
| Ollama connection error | Make sure Ollama is running on port 11434. |
| Model not found | Download the selected model using `ollama pull`. |
| Python dependency conflict | Use Python 3.12 and the provided requirements file. |
| Segmentation fault | Use the thread limits in the manual startup instructions and avoid mixing Conda and Python virtual environments. |
| Frontend fails to start | Run `npm install` inside the `frontend/` directory. |
| API connection error | Confirm that the backend is running on port 8000. |
| Slow first request | Allow time for model downloading and initialization. |

### Stop and restart

Press `Ctrl+C` in the running service terminals to stop the application.

To restart Sawt later, make sure Ollama is running and execute:

```bash
bash start.sh
```

There is no need to reinstall dependencies, rerun notebooks, or retrain models during normal restarts.
