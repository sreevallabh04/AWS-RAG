# Job Search RAG Backend

Backend-only deployment package for the Job Search Retrieval-Augmented Generation (RAG) system. The service exposes a Flask REST API that loads pre-built embeddings, performs semantic search with FAISS, and augments responses with OpenAI models.

## Features

- Semantic search across a pre-computed job corpus
- Retrieval-augmented responses generated with OpenAI GPT models
- FAISS index (flat L2) for fast vector similarity
- Production-ready Gunicorn entry point
- `render.yaml` for hands-free Render provisioning (includes persistent disk mount)

## Project Layout

```
api.py           # Flask app exposing /api routes
rag_system.py    # JobRAGSystem class (embeddings + FAISS + OpenAI)
requirements.txt # Python dependencies (includes gunicorn)
render.yaml      # Render infrastructure definition
embeddings/
├── embeddings.npy
├── faiss_index.bin
└── metadata.pkl
README.md        # This document
```

> `metadata_sample.json` exists only as a reference dump. The API reads the serialized dataframe in `metadata.pkl`.

## Local Setup

1. **Create a virtual environment and install dependencies**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Provide environment variables**
   - Create a `.env` file next to `api.py`, or export variables in your shell:
     ```
     OPENAI_API_KEY=sk-...
     # Optional override when embeddings live elsewhere
     EMBEDDINGS_DIR=embeddings
     PORT=5000
     ```

3. **Start the server locally**
   ```powershell
   python api.py
   ```
   Visit `http://localhost:5000/api/health` and expect `{"status": "healthy"}`.

## Deploying to Render

1. Push this folder (as repo root) to GitHub/GitLab/Bitbucket.
2. In Render, create a **Web Service** and connect the repository.
3. Render auto-detects `render.yaml` and applies:
   - Build: `pip install --upgrade pip && pip install -r requirements.txt`
   - Start: `gunicorn api:app --bind 0.0.0.0:$PORT --timeout 120 --preload`
   - Persistent disk mounted at `/opt/render/project/src/embeddings`
4. Add environment variables via the dashboard:
   - `OPENAI_API_KEY` (required)
   - `EMBEDDINGS_DIR` (optional, only if you mount elsewhere)
5. After the initial deploy, upload `embeddings.npy`, `faiss_index.bin`, and `metadata.pkl` into the mounted `embeddings` directory via Render Shell or a deploy hook (skip if they were checked into git).
6. Hit `https://<service>.onrender.com/api/health` to confirm the deployment.

## API Overview

| Method | Endpoint            | Description                                    |
| ------ | ------------------- | ---------------------------------------------- |
| GET    | `/api/health`       | Liveness probe                                 |
| GET    | `/api/stats`        | Embedding count and dimension                  |
| POST   | `/api/search`       | Full RAG flow; body requires `query`           |
| POST   | `/api/similar-jobs` | Semantic matches without the AI explanation    |

`top_k` defaults to 5 and is capped at 20 to protect the service.

## Operational Notes

- The FAISS index and metadata must exist before the first request; otherwise `JobRAGSystem` raises a descriptive `FileNotFoundError`.
- Gunicorn timeout is 120 seconds to handle cold starts on sleeping instances.
- Logs go to stdout/stderr; monitor them in Render’s log viewer.
- Rotate `OPENAI_API_KEY` directly in Render when needed.

## Security Checklist

- Keep `.env` out of version control.
- Tighten CORS in `api.py` (currently `CORS(app)` allows all origins).
- Add rate limiting or authentication before exposing the API publicly.

## Extending

- Rebuild embeddings and FAISS index when your dataset changes, then redeploy.
- Swap embedding or chat models in `rag_system.py` to control latency/cost.
- For horizontal scaling, enable Render autoscaling and monitor metrics.

Happy shipping! 🚀

