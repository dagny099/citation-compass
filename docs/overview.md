# Overview

The **Academic Citation Platform** is a personal portfolio project that analyzes
academic citation networks and predicts likely citation relationships using a
TransE knowledge-graph embedding model, with an interactive Streamlit interface
and a Neo4j graph backend.

> This project is built and maintained by a single author
> ([dagny099](https://github.com/dagny099)). It is provided as-is for
> demonstration and learning.

## What it does

- **Citation prediction** — a trained TransE model (128-dimensional embeddings,
  ~1.6M parameters) produces ranked citation recommendations for a given paper.
- **Network analysis** — community detection, centrality measures, and temporal
  trends over the citation graph.
- **Interactive exploration** — a multi-page Streamlit app for predictions,
  embedding exploration, visualizations, and results interpretation.
- **Demo mode** — a curated 13-paper offline dataset so you can explore the app
  with zero configuration and no database.
- **Data import** — Semantic Scholar integration with search, paper-ID, and
  file-upload import plus streaming progress.

## How it fits together

| Layer | Where | Responsibility |
|-------|-------|----------------|
| Web interface | `app.py`, `src/streamlit_app/` | Multi-page interactive UI |
| Services | `src/services/` | ML serving (`ml_service.py`), analytics (`analytics_service.py`) |
| Data | `src/data/`, `src/database/` | API client, import pipeline, Neo4j access |
| Models | `models/` | Trained TransE checkpoint and mappings |
| CLI | `src/cli/` (`acp`) | Health checks, DB setup, model info/predictions |

For a deeper description of the design, see the
[Architecture guide](architecture.md).

## Where to go next

- **[Installation](getting-started/installation.md)** — set up the environment.
- **[Quick Start](getting-started/quick-start.md)** — run the app and try demo mode.
- **[User Guide](user-guide/overview.md)** — walk through the interactive features.
- **[Notebooks](notebooks/overview.md)** — the exploration → training → evaluation → presentation pipeline.

## A note on reported metrics

The pipeline includes an evaluation harness for standard ranking/classification
metrics (MRR, Hits@K, AUC). This repository does **not** currently ship a saved,
reproducible evaluation run — the metrics shown on the app's Analysis Pipeline
page are **simulated for demonstration**. Verifiable model facts (dataset size,
parameter count, training convergence) are recorded in
`models/training_metadata.json`.

---

> **Contributing to the docs?** These pages are built with
> [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). Run
> `mkdocs serve` for a live-reloading local preview, or `mkdocs build --strict`
> to catch broken links before publishing.
