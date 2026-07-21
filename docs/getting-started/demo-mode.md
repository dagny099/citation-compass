# 🎭 Demo Mode

**The fastest way to see what this platform does — no database, no API keys, no waiting.**

Demo Mode ships a small, curated slice of the academic-citation world and wires it
into every feature of the app: ML citation predictions, interactive network graphs,
community detection, and export. You get a working, clickable product in the time it
takes to run one command.

!!! tip "Zero setup, on purpose"
    Demo Mode needs **no Neo4j database and no Semantic Scholar API key**. It runs
    entirely on bundled data and synthetic embeddings, so you can explore the whole
    interface offline before deciding whether to connect real data.

---

## Try it in 30 seconds

```bash
# From the project root
streamlit run app.py
# Opens at http://localhost:8501/
```

Then, in the app:

1. Open **🎭 Demo Datasets** from the sidebar.
2. Under **Available Demo Datasets**, click **🚀 Load Dataset** on **`complete_demo`**.
3. Jump to **🤖 ML Predictions**, **📊 Enhanced Visualizations**, or **📈 Results
   Interpretation** and start exploring.

That's it. No `.env`, no credentials, no import step.

---

## What's in the box

The `complete_demo` dataset is a hand-picked network of **13 real, high-impact
papers** and the citation links between them — chosen to be recognizable and to span
several research communities so the graph and the clustering have something
interesting to say.

<div class="grid cards" markdown>

-   :material-file-document-multiple: **13 curated papers**

    Real titles, authors, venues, and citation counts — from foundational work like
    *Going Deeper with Convolutions* (Inception) to recent work like *Segment Anything*.

-   :material-shape-outline: **Multiple fields**

    Computer Vision, Machine Learning, Neuroscience, and Medical Informatics — enough
    cross-disciplinary structure to make community detection meaningful.

-   :material-source-branch: **A real citation graph**

    Papers, authors, and venues as nodes, with generated citation relationships that
    form a connected, explorable network.

-   :material-calendar-range: **Spanning years**

    Publications ranging from classic (2009) to current (2024), so temporal analysis
    has a real timeline to work with.

</div>

Prefer something even smaller? The **Generate Fresh Demo Data** action also produces a
**minimal** 5-paper dataset, and a set of **quick test fixtures** are available for
fast, repeatable experimentation.

---

## What you can actually do

Everything in the app works against demo data — nothing is stubbed out with "coming
soon" screens:

- **🤖 ML Predictions** — pick a paper and get ranked citation recommendations with
  confidence scores. In demo mode these come from **synthetic embeddings**, so you can
  exercise the full prediction UI without training a model first.
- **🔗 Interactive Networks** — click nodes to inspect papers, filter in real time, and
  trace citation paths.
- **🏘️ Community Detection & Centrality** — run clustering and centrality measures over
  the demo graph.
- **⏰ Temporal Analysis** — watch how the network evolves across the dataset's year range.
- **📄 Export** — generate reports and LaTeX/CSV/JSON outputs from what you find.

!!! note "About the numbers you'll see"
    Demo Mode is for *exploring the interface*, not for benchmarking model quality.
    Predictions use synthetic embeddings, and the Analysis Pipeline page's evaluation
    metrics are **simulated for demonstration** (clearly labeled in the app). For
    verifiable facts about the trained model, see `models/training_metadata.json`.

---

## Managing demo mode

From the **🎭 Demo Datasets** page you can:

| Action | What it does |
|--------|--------------|
| **🚀 Load Dataset** | Activate a demo dataset (start with `complete_demo`) |
| **🔍 Preview** | Peek at sample papers before loading |
| **🔧 Generate Fresh Demo Data** | Rebuild the `complete_demo` and `minimal` datasets |
| **⚡ Quick Load** | Load a fast fixture for repeatable testing |
| **🔄 Switch to Production Mode** | Point the app at your real Neo4j database |
| **🧹 Clear Demo Mode** | Reset back to a clean state |

---

## When you're ready for real data

Demo Mode is a preview, not a ceiling. When you want to analyze your own research:

1. **[Configure a Neo4j database](configuration.md)** and copy `.env.example` → `.env`.
2. **[Import papers](../user-guide/data-import.md)** by search query, paper-ID list, or
   file upload.
3. Switch to **Production Mode** from the Demo Datasets page and explore your own network.

---

## Next steps

- **[Quick Start](quick-start.md)** — the full first-run walkthrough.
- **[Demo Datasets guide](../user-guide/demo-datasets.md)** — a deeper tour of the sample data.
- **[ML Predictions](../user-guide/ml-predictions.md)** — how citation prediction works.
- **[Architecture](../architecture.md)** — how the pieces fit together.
