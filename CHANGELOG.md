# Changelog

All notable changes to Citation Compass will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Interactive Welcome Graph Snapshot on home page with one-hop knowledge graph visualization
- "My Knowledge Graph at a Glance" section with real-time metrics and schema diagram
- Quick Explore section with prominent center-paper details
- Clickable nodes in citation network visualization for deeper exploration
- Session state management for ego network and selected papers

### Changed
- Refactored home page into focused landing with interactive graph snippet
- Moved status and training information to new "Overview & Training" page
- Repositioned Enhanced Visualizations to third position under Main navigation
- Updated graph legend labels for clarity (Cites center vs references)
- Centered schema diagram with improved layout and spacing
- Improved tone throughout UI to use "my" for personalization
- Set seed pool size to Top 50 by in-degree for freshness

### Fixed
- Documentation index links corrected
- Navigation structure now includes all notebook pages
- Verified all relative paths in documentation

## [0.1.0] - 2025-10-24

### Added
- **Core Platform Features**
  - Complete Streamlit web application with multi-page navigation
  - Interactive citation network visualization with zoom and filtering
  - ML service layer for TransE-based citation predictions
  - Analytics service with community detection and centrality metrics
  - Command-line interface (CLI) with `acp` command

- **Data Import System**
  - Comprehensive data import with multiple sources (search, IDs, file upload)
  - Semantic Scholar API integration with rate limiting
  - Streaming pagination with real-time progress tracking
  - Batch processing with error handling and resumable operations
  - Support for .txt and .csv file uploads for paper ID imports

- **Demo Mode**
  - Curated demo datasets across multiple academic fields
  - Offline operation without Neo4j database requirement
  - Sample ML predictions using included embeddings
  - Complete demo dataset documentation

- **Machine Learning Pipeline**
  - TransE embedding model for citation prediction
  - Four-notebook workflow for end-to-end analysis:
    - 01_comprehensive_exploration.ipynb
    - 02_model_training_pipeline.ipynb
    - 03_prediction_evaluation.ipynb
    - 04_narrative_presentation.ipynb
  - Local model training capabilities (complete project independence)
  - Model compatibility improvements for robustness

- **Network Analysis**
  - Community detection using Louvain and other algorithms
  - Centrality metrics (betweenness, closeness, PageRank)
  - Temporal trend analysis with date range filtering
  - Interactive network visualization with pyvis
  - Export capabilities for LaTeX tables and figures

- **Documentation**
  - Comprehensive MkDocs documentation system
  - Visual improvements with diagrams and screenshots
  - User guides for all major features
  - Developer guides and API documentation
  - Architecture documentation
  - Demo mode documentation
  - Neo4j ping playbook
  - Installation and verification guides

- **Testing Infrastructure**
  - 44 test fixtures for consistent testing
  - Unit, integration, and performance test suites
  - Test markers for slow, integration, analytics, ML, and database tests
  - Coverage reporting configured
  - pytest configuration with strict settings

- **Database & Models**
  - Neo4j graph database integration with connection pooling
  - Pydantic data models for all entities (Paper, Author, Citation, Venue, Field)
  - Database schema with constraints and indices
  - Query efficiency improvements
  - Health check system

### Changed
- **Documentation Restructure**
  - Standardized naming to "Citation Compass" throughout
  - Reduced promotional tone in README
  - Improved navigation structure
  - Added visual diagrams to key pages
  - Enhanced MkDocs interface and user experience
  - Fixed broken links and corrected relative paths

- **Code Quality**
  - Major code quality improvements and standardization
  - Removed external dependencies for better independence
  - Improved documentation structure
  - Enhanced error handling and system reliability
  - Better separation of concerns in service layers

- **User Experience**
  - Simplified Results Interpretation UI
  - Enhanced Home dashboard with clear ML service status guidance
  - Improved home page layout with platform overview
  - Better visualization popup system with local-first data access
  - Enhanced visualization interactivity

- **Performance**
  - Improved Semantic Scholar API performance with streaming
  - Adaptive batching for large imports
  - Query optimization for Neo4j operations
  - Caching improvements for analytics and predictions

### Fixed
- Enhanced Visualization 400 errors resolved
- Results_Interpretation.py indentation syntax error
- API client configuration errors
- Installation verification issues
- MkDocs warnings with comprehensive notebook documentation
- Documentation accuracy issues

### Security
- Removed exposed credentials from codebase
- Cleaned up legacy phase references
- Improved .gitignore for better secret protection

### Deprecated
- Legacy phase-based architecture references removed

## [0.0.1] - 2025-08-22

### Added
- Initial project foundation and architecture
- Basic Neo4j database connection
- Core data models for papers, authors, and citations
- Preliminary Streamlit interface
- Foundation for ML prediction service

---

## Release Notes

### Version 0.1.0 Highlights

This release represents the first public version of Citation Compass, a comprehensive platform for academic citation network analysis and prediction.

**Key Capabilities:**
- Explore citation networks with interactive visualizations
- Predict likely citations using TransE embeddings
- Analyze research communities and collaboration patterns
- Track temporal trends in academic literature
- Export publication-ready figures and tables

**Getting Started:**
- Try the demo mode without any setup: `streamlit run app.py`
- Full documentation: https://docs.barbhs.com/citation-compass/
- Interactive demo: https://cartography.barbhs.com/

**System Requirements:**
- Python 3.8+
- Optional: Neo4j database for full functionality
- All features work in demo mode without external dependencies

### Migration Guide

This is the initial release, so no migration is required. For future updates:
- Version 0.x.x may include breaking changes
- Version 1.0.0 will stabilize the API
- Always check this changelog before upgrading

### Contributing

We welcome contributions! Please see our documentation for:
- Development setup instructions
- Code style guidelines (Black, isort, flake8, mypy)
- Testing requirements
- Pull request process

### Links

- **Homepage**: https://github.com/dagny099/citation-compass
- **Documentation**: https://docs.barbhs.com/citation-compass/
- **Issue Tracker**: https://github.com/dagny099/citation-compass/issues
- **Discussions**: https://github.com/dagny099/citation-compass/discussions

---

[Unreleased]: https://github.com/dagny099/citation-compass/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dagny099/citation-compass/releases/tag/v0.1.0
[0.0.1]: https://github.com/dagny099/citation-compass/releases/tag/v0.0.1
