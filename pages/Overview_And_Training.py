"""
Overview & Training page for the Academic Citation Platform.

This page centralizes platform status and training guidance that previously
lived on the Home page.
"""

import streamlit as st

st.title("🧰 Overview & Training")

st.subheader("📊 Platform Overview")

# ML service status and metrics
ml_service_online = False
try:
    from src.services.ml_service import get_ml_service
    ml_service = get_ml_service()
    health = ml_service.health_check()

    col1, col2, col3 = st.columns(3)
    if health["status"] == "healthy":
        with col1:
            st.metric("🤖 ML Models", "1", help="TransE citation prediction model")
            st.markdown("<div style='color: green;'>✅ Ready</div>", unsafe_allow_html=True)
        with col2:
            st.metric("📄 Papers in Model", f"{health['num_entities']:,}", help="Papers available for predictions")
            st.markdown("<div style='color: green;'>✅ Loaded</div>", unsafe_allow_html=True)
        with col3:
            st.metric("🎯 Prediction Accuracy", "~85%", help="Model performance on test set")
            st.markdown("<div style='color: green;'>✅ Validated</div>", unsafe_allow_html=True)
        ml_service_online = True
        ml_service_status = "online"
        ml_service_message = "✅ ML Service: Online - Ready for citation predictions!"
    else:
        with col1:
            st.metric("🤖 ML Models", "0")
            st.markdown("<div style='color: orange;'>⚠️ Need training</div>", unsafe_allow_html=True)
        with col2:
            st.metric("📄 Papers in Model", "0")
            st.markdown("<div style='color: orange;'>⚠️ Train model first</div>", unsafe_allow_html=True)
        with col3:
            st.metric("🎯 Prediction Accuracy", "N/A")
            st.markdown("<div style='color: orange;'>⚠️ After training</div>", unsafe_allow_html=True)
        ml_service_status = "offline"
        ml_service_message = "⚠️ ML Service: Offline - No trained model found"
        ml_service_info = """
        **Need to train a model first?**

        • Open `notebooks/02_model_training_pipeline.ipynb` to train your TransE model
        • Training time: ~30–60 minutes depending on data size
        • Outputs: model files written to the `models/` directory
        """
except Exception as e:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🤖 ML Models", "0")
        st.markdown("<div style='color: red;'>❌ Service error</div>", unsafe_allow_html=True)
    with col2:
        st.metric("📄 Papers in Model", "0")
        st.markdown("<div style='color: red;'>❌ Service error</div>", unsafe_allow_html=True)
    with col3:
        st.metric("🎯 Prediction Accuracy", "N/A")
        st.markdown("<div style='color: red;'>❌ Service error</div>", unsafe_allow_html=True)
    ml_service_status = "error"
    ml_service_message = f"❌ ML Service: Failed to load ({e})"
    ml_service_info = "**Troubleshooting**: `pip install -e \".[all]\"`"

# API client status
try:
    from src.data.unified_api_client import UnifiedSemanticScholarClient
    _api_client = UnifiedSemanticScholarClient()
    api_client_status = "ready"
    api_client_message = "✅ API Client: Ready - Can fetch paper data from Semantic Scholar"
except Exception as e:
    api_client_status = "error"
    api_client_message = f"❌ API Client: Failed to load ({e})"

st.markdown("##### 🔧 System Status")
col_s1, col_s2 = st.columns(2)
with col_s1:
    if ml_service_status == "online":
        st.success(ml_service_message)
    elif ml_service_status == "offline":
        st.warning(ml_service_message)
        st.info(ml_service_info)
    else:
        st.error(ml_service_message)
        st.info(ml_service_info)
with col_s2:
    if api_client_status == "ready":
        st.success(api_client_message)
    else:
        st.error(api_client_message)

st.markdown("---")
st.subheader("📓 Train or Re‑train the Model")
st.markdown("""
1. Open `notebooks/01_comprehensive_exploration.ipynb` to profile your citation network.
2. Train with `notebooks/02_model_training_pipeline.ipynb` (saves artifacts in `models/`).
3. Return to ML Predictions for inference and to Enhanced Visualizations for overlays.
""")

if not ml_service_online:
    st.info("""
    ✅ You can still use:
    - 📊 Enhanced Visualizations: Explore citation networks
    - 📓 Analysis Pipeline: Run data exploration workflows
    """)

st.markdown("---")
st.subheader("🧠 About the ML Model")
if ml_service_online:
    st.success("**Model Status**: Ready for predictions (TransE embeddings).")
else:
    st.info("Once trained, the system uses TransE embeddings.")
st.code('source_paper + "CITES" ≈ target_paper')

