# Changelog

All notable changes to the Academic Citation Platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Docker and Google Cloud Run deployment support (`Dockerfile`, `docker-compose.yml`,
  `deploy_cloud_run_container.sh`, `setup_secrets.sh`, `setup_custom_domain.sh`) — see `DEPLOYMENT.md`.
- `docs` optional-dependency extra in `pyproject.toml` for building the documentation.
- `CHANGELOG.md`.

### Changed
- Documentation audit: removed unverified performance metrics (AUC / MRR / Hits@K,
  "25x faster", "100K+ papers", "~200 hours saved") and kept only claims backed by
  `models/training_metadata.json`. The Analysis Pipeline page now clearly labels its
  evaluation metrics as simulated/illustrative.

### Fixed
- Broken MkDocs nav entries (`getting-started/demo-mode.md`,
  `developer-guide/architecture.md`) and dead documentation links.

## [0.1.0]

Initial consolidated release. This project merges three predecessor codebases
(knowledge cartography, citation mapping, and a citation prediction system) into a
single academic research tool.

### Added
- **Streamlit web app** (`app.py`) with multi-page navigation: Home, Data Import,
  Demo Datasets, ML Predictions, Embedding Explorer, Enhanced Visualizations,
  Results Interpretation, and Analysis Pipeline.
- **TransE citation-prediction model** (`src/services/ml_service.py`) with a trained
  checkpoint (128-dim embeddings, 1,612,288 parameters) served for top-K citation
  prediction and embedding exploration.
- **Analytics service** (`src/services/analytics_service.py`) for network analysis,
  community detection, centrality measures, and temporal trends.
- **Demo mode** with a curated 13-paper offline dataset — no database required.
- **Data import pipeline** with Semantic Scholar integration, search/ID/file-upload
  import, streaming pagination, and real-time progress tracking.
- **CLI** (`acp`) with `health`, `db setup`, `db stats`, `ml info`, `ml predict`,
  and `config show` commands.
- **Jupyter notebook pipeline** (exploration → training → evaluation → presentation).
- **MkDocs (Material) documentation** site.
- **Test suite** under `tests/` covering models, services, database, analytics,
  and the import pipeline.

[Unreleased]: https://github.com/dagny099/citation-compass/compare/main...HEAD
[0.1.0]: https://github.com/dagny099/citation-compass/releases/tag/v0.1.0
