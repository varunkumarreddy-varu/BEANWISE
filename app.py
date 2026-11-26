"""
BEANWISE: AI-Powered Dry Bean Classification Platform
Production Web Application

Interface Features:
- Direct opening to the primary prediction experience.
- Manual entry of all 16 numerical morphological measurements with validation.
- Real-time prediction with dynamic display of authentic dry bean photographs.
- Batch CSV prediction with downloadable results.
- Botanical bean variety gallery.
- Session prediction history.
- Model performance insights based on independent test evaluation.
- No decorative emojis, academic phase labels, or AI-generated marketing phrases.
"""

import sys
import io
import time
from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROJECT_NAME,
    PROJECT_TITLE,
    TARGET_CLASSES,
    FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
    REPORTS_DIR,
    PROCESSED_DATASET_DIR,
)
from src.prediction import (
    get_prediction_engine,
    predict_batch_csv,
    get_variety_info,
    get_all_varieties,
    EXPECTED_16_FEATURES,
)

# Page configuration
st.set_page_config(
    page_title=f"{PROJECT_NAME} - Dry Bean Variety Classification",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional CSS Styling (Clean Product Dark Theme)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Brand Bar */
    .brand-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 18px 24px;
        margin-bottom: 20px;
    }

    .brand-title {
        font-size: 22px;
        font-weight: 700;
        color: #F9FAFB;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .brand-subtitle {
        font-size: 13px;
        color: #9CA3AF;
        margin: 4px 0 0 0;
    }

    .brand-badge {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: #1F2937;
        color: #38BDF8;
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid #374151;
    }

    /* Section Containers */
    .content-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .card-title {
        font-size: 15px;
        font-weight: 600;
        color: #E5E7EB;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1F2937;
    }

    /* Result Card */
    .result-banner {
        background: #0B132B;
        border: 1px solid #1E3A8A;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }

    .result-variety {
        font-size: 28px;
        font-weight: 700;
        color: #38BDF8;
        margin: 0 0 4px 0;
        letter-spacing: -0.01em;
    }

    .result-scientific {
        font-size: 13px;
        font-style: italic;
        color: #94A3B8;
        margin-bottom: 12px;
    }

    .result-metric {
        font-size: 13px;
        color: #CBD5E1;
        margin: 4px 0;
    }

    .result-metric strong {
        color: #F8FAFC;
    }

    /* Variety Gallery Card */
    .variety-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        overflow: hidden;
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    .variety-card-body {
        padding: 14px;
        flex-grow: 1;
    }

    .variety-card-title {
        font-size: 16px;
        font-weight: 700;
        color: #F3F4F6;
        margin-bottom: 2px;
    }

    .variety-card-latin {
        font-size: 12px;
        font-style: italic;
        color: #9CA3AF;
        margin-bottom: 8px;
    }

    .variety-card-desc {
        font-size: 12px;
        color: #D1D5DB;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "example_inputs" not in st.session_state:
    st.session_state.example_inputs = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Initialize Prediction Engine
try:
    engine = get_prediction_engine("best_model")
except Exception as e:
    st.error(f"Failed to initialize machine learning engine: {e}")
    st.stop()

# Sidebar Navigation (Clean Product Navigation)
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 18px 0;">
            <div style="font-size: 18px; font-weight: 700; color: #38BDF8; letter-spacing: -0.01em;">BEANWISE</div>
            <div style="font-size: 12px; color: #9CA3AF;">Dry Bean Variety Classifier</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_section = option_menu(
        menu_title=None,
        options=[
            "Predict",
            "Batch Prediction",
            "Bean Varieties",
            "Prediction History",
            "Model Insights",
            "Settings",
        ],
        icons=[
            "cpu",
            "upload",
            "grid",
            "clock-history",
            "bar-chart-line",
            "gear",
        ],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#38BDF8", "font-size": "14px"},
            "nav-link": {
                "font-size": "13px",
                "text-align": "left",
                "margin": "4px 0",
                "color": "#D1D5DB",
                "border-radius": "6px",
            },
            "nav-link-selected": {
                "background-color": "#1F2937",
                "color": "#38BDF8",
                "font-weight": "600",
                "border-left": "3px solid #38BDF8",
            },
        },
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 11px; color: #6B7280; line-height: 1.5;">
            <strong>System:</strong> Support Vector Classifier (RBF)<br>
            <strong>Features:</strong> 16 Geometric Dimensions<br>
            <strong>Supported Classes:</strong> 7 Cultivars<br>
            <strong>Independent Test Accuracy:</strong> 92.22%
        </div>
        """,
        unsafe_allow_html=True,
    )

# Common Header
st.markdown(
    f"""
    <div class="brand-container">
        <div>
            <h1 class="brand-title">BEANWISE: AI-Powered Dry Bean Classification</h1>
            <p class="brand-subtitle">Real-time variety identification using 16 physical morphological measurements.</p>
        </div>
        <div class="brand-badge">{selected_section}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 1. PREDICT (DEFAULT LANDING EXPERIENCE)
# ==============================================================================
if selected_section == "Predict":
    # Optional Load Example Bar
    cleaned_path = PROCESSED_DATASET_DIR / "cleaned_dry_beans.csv"
    example_samples = {}
    if cleaned_path.is_file():
        try:
            df_sample_ref = pd.read_csv(cleaned_path)
            for cls in TARGET_CLASSES:
                match = df_sample_ref[df_sample_ref["Class"] == cls]
                if not match.empty:
                    example_samples[cls] = match.iloc[0].drop("Class").to_dict()
        except Exception:
            pass

    col_hdr, col_actions = st.columns([2, 1.2])
    with col_hdr:
        st.markdown("<p style='font-size: 14px; color: #9CA3AF; margin: 0 0 12px 0;'>Enter the 16 numerical seed measurements below. All fields must be positive numerical values.</p>", unsafe_allow_html=True)
    with col_actions:
        ex_col1, ex_col2 = st.columns([2, 1])
        with ex_col1:
            selected_example = st.selectbox(
                "Load Benchmark Example:",
                ["Custom Measurements"] + list(example_samples.keys()),
                label_visibility="collapsed",
            )
        with ex_col2:
            if st.button("Clear All", use_container_width=True):
                st.session_state.example_inputs = None
                st.session_state.last_result = None
                st.rerun()

    if selected_example != "Custom Measurements" and selected_example in example_samples:
        st.session_state.example_inputs = example_samples[selected_example]

    # Helper function to obtain value
    def get_input_val(feat_name: str, fallback_default: float) -> float:
        if st.session_state.example_inputs and feat_name in st.session_state.example_inputs:
            return float(st.session_state.example_inputs[feat_name])
        return float(fallback_default)

    # 16 Inputs organized into 4 logical physical groups
    inputs = {}

    g1, g2, g3, g4 = st.columns(4)

    with g1:
        st.markdown("<div class='card-title'>Basic Dimensions</div>", unsafe_allow_html=True)
        inputs["Area"] = st.number_input(
            "Area (pixels)",
            value=get_input_val("Area", 42300.0),
            step=100.0,
            help=FEATURE_DESCRIPTIONS.get("Area"),
        )
        inputs["Perimeter"] = st.number_input(
            "Perimeter (pixels)",
            value=get_input_val("Perimeter", 760.0),
            step=5.0,
            help=FEATURE_DESCRIPTIONS.get("Perimeter"),
        )
        inputs["ConvexArea"] = st.number_input(
            "Convex Area (pixels)",
            value=get_input_val("ConvexArea", 42800.0),
            step=100.0,
            help=FEATURE_DESCRIPTIONS.get("ConvexArea"),
        )
        inputs["EquivDiameter"] = st.number_input(
            "Equivalent Diameter",
            value=get_input_val("EquivDiameter", 232.0),
            step=1.0,
            help=FEATURE_DESCRIPTIONS.get("EquivDiameter"),
        )

    with g2:
        st.markdown("<div class='card-title'>Axis Measurements</div>", unsafe_allow_html=True)
        inputs["MajorAxisLength"] = st.number_input(
            "Major Axis Length",
            value=get_input_val("MajorAxisLength", 285.0),
            step=1.0,
            help=FEATURE_DESCRIPTIONS.get("MajorAxisLength"),
        )
        inputs["MinorAxisLength"] = st.number_input(
            "Minor Axis Length",
            value=get_input_val("MinorAxisLength", 190.0),
            step=1.0,
            help=FEATURE_DESCRIPTIONS.get("MinorAxisLength"),
        )
        inputs["AspectRation"] = st.number_input(
            "Aspect Ratio",
            value=get_input_val("AspectRation", 1.50),
            step=0.01,
            format="%.4f",
            help=FEATURE_DESCRIPTIONS.get("AspectRation"),
        )
        inputs["Eccentricity"] = st.number_input(
            "Eccentricity",
            value=get_input_val("Eccentricity", 0.74),
            step=0.01,
            format="%.4f",
            help=FEATURE_DESCRIPTIONS.get("Eccentricity"),
        )

    with g3:
        st.markdown("<div class='card-title'>Shape & Boundary</div>", unsafe_allow_html=True)
        inputs["Extent"] = st.number_input(
            "Extent",
            value=get_input_val("Extent", 0.75),
            step=0.01,
            format="%.4f",
            help=FEATURE_DESCRIPTIONS.get("Extent"),
        )
        inputs["Solidity"] = st.number_input(
            "Solidity",
            value=get_input_val("Solidity", 0.988),
            step=0.001,
            format="%.4f",
            help=FEATURE_DESCRIPTIONS.get("Solidity"),
        )
        inputs["roundness"] = st.number_input(
            "Roundness",
            value=get_input_val("roundness", 0.86),
            step=0.01,
            format="%.4f",
            help=FEATURE_DESCRIPTIONS.get("roundness"),
        )
        inputs["Compactness"] = st.number_input(
            "Compactness",
            value=get_input_val("Compactness", 0.81),
            step=0.01,
            format="%.4f",
            help=FEATURE_DESCRIPTIONS.get("Compactness"),
        )

    with g4:
        st.markdown("<div class='card-title'>Geometric Factors</div>", unsafe_allow_html=True)
        inputs["ShapeFactor1"] = st.number_input(
            "Shape Factor 1",
            value=get_input_val("ShapeFactor1", 0.0067),
            step=0.0001,
            format="%.6f",
            help=FEATURE_DESCRIPTIONS.get("ShapeFactor1"),
        )
        inputs["ShapeFactor2"] = st.number_input(
            "Shape Factor 2",
            value=get_input_val("ShapeFactor2", 0.0021),
            step=0.0001,
            format="%.6f",
            help=FEATURE_DESCRIPTIONS.get("ShapeFactor2"),
        )
        inputs["ShapeFactor3"] = st.number_input(
            "Shape Factor 3",
            value=get_input_val("ShapeFactor3", 0.65),
            step=0.01,
            format="%.4f",
            help=FEATURE_DESCRIPTIONS.get("ShapeFactor3"),
        )
        inputs["ShapeFactor4"] = st.number_input(
            "Shape Factor 4",
            value=get_input_val("ShapeFactor4", 0.995),
            step=0.001,
            format="%.4f",
            help=FEATURE_DESCRIPTIONS.get("ShapeFactor4"),
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Predict Variety", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Executing prediction pipeline..."):
            result = engine.predict_single(inputs)
            st.session_state.last_result = result
            if result["success"]:
                st.session_state.history.append({
                    "timestamp": result["timestamp"],
                    "variety": result["predicted_class"],
                    "probability": result["top_probability"],
                    "confidence": result["confidence_level"],
                    **result["sanitized_inputs"],
                })

    # Render Result if Available
    if st.session_state.last_result:
        res = st.session_state.last_result

        if not res["success"]:
            st.error("Input Validation Failed:")
            for err in res["errors"]:
                st.write(f"- {err}")
        else:
            variety = res["predicted_class"]
            vinfo = res["variety_info"]
            prob = res["top_probability"]
            conf_level = res["confidence_level"]
            conf_color = res["confidence_color"]

            st.markdown("<div class='card-title' style='margin-top: 20px;'>Prediction Result</div>", unsafe_allow_html=True)

            r_col_img, r_col_info = st.columns([1, 1.8])

            with r_col_img:
                img_path = vinfo.get("image_path")
                if img_path and Path(img_path).is_file():
                    st.image(str(img_path), use_container_width=True, caption=f"Specimen photograph: {vinfo.get('scientific_name')}")
                else:
                    st.info(f"Specimen image unavailable for {variety}.")

            with r_col_info:
                st.markdown(
                    f"""
                    <div class="result-banner">
                        <div class="result-variety">{variety}</div>
                        <div class="result-scientific">{vinfo.get('scientific_name', '')} &bull; {vinfo.get('common_name', '')}</div>
                        <p style="color: #E2E8F0; font-size: 13px; line-height: 1.5; margin-bottom: 12px;">
                            {vinfo.get('description', '')}
                        </p>
                        <div class="result-metric">
                            <strong>Model Confidence:</strong> <span style="color: {conf_color}; font-weight: 600;">{conf_level}</span>
                            ({prob * 100:.2f}% model-estimated class probability)
                        </div>
                        <div class="result-metric">
                            <strong>Key Characteristics:</strong> {vinfo.get('key_characteristics', '')}
                        </div>
                        <div class="result-metric">
                            <strong>Culinary Application:</strong> {vinfo.get('culinary_uses', '')}
                        </div>
                        <div class="result-metric" style="color: #64748B; font-size: 11px; margin-top: 8px;">
                            Inference latency: {res['inference_time_ms']} ms &bull; Architecture: Support Vector Machine (RBF)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Out of distribution warnings if present
            if res["warnings"]:
                with st.expander("Distribution Advisories", expanded=False):
                    for w in res["warnings"]:
                        st.warning(w)

            # Class Probability Distribution Chart
            if res["class_probabilities"]:
                prob_df = pd.DataFrame(res["class_probabilities"]).sort_values("probability", ascending=True)

                fig_prob = go.Figure(go.Bar(
                    x=prob_df["probability"],
                    y=prob_df["class"],
                    orientation="h",
                    marker=dict(
                        color=["#38BDF8" if c == variety else "#334155" for c in prob_df["class"]],
                        line=dict(width=0),
                    ),
                    text=[f"{p*100:.1f}%" for p in prob_df["probability"]],
                    textposition="outside",
                ))
                fig_prob.update_layout(
                    title="Model-Estimated Probability Distribution Across Varieties",
                    template="plotly_dark",
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    margin=dict(l=20, r=40, t=40, b=20),
                    xaxis=dict(range=[0, 1.05], title="Probability Score", showgrid=True, gridcolor="#1F2937"),
                    yaxis=dict(title=None),
                    height=260,
                )
                st.plotly_chart(fig_prob, use_container_width=True)

            # Expandable Details & CSV Export
            with st.expander("Review Input Measurements & Export"):
                st.dataframe(pd.DataFrame([res["sanitized_inputs"]]), use_container_width=True, hide_index=True)
                export_csv = pd.DataFrame([{
                    "Predicted_Cultivar": variety,
                    "Confidence_Probability": prob,
                    "Confidence_Level": conf_level,
                    "Inference_Time_ms": res["inference_time_ms"],
                    **res["sanitized_inputs"],
                }]).to_csv(index=False)
                st.download_button(
                    label="Download Result (CSV)",
                    data=export_csv,
                    file_name=f"beanwise_{variety.lower()}_prediction.csv",
                    mime="text/csv",
                )


# ==============================================================================
# 2. BATCH PREDICTION
# ==============================================================================
elif selected_section == "Batch Prediction":
    st.markdown("<div class='card-title'>Batch CSV Prediction Engine</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size: 13px; color: #9CA3AF;">
            Upload a CSV file containing multiple observations of the 16 morphological measurements.
            All input columns must match the required schema.
        </p>
        """,
        unsafe_allow_html=True,
    )

    template_path = PROJECT_ROOT / "assets" / "templates" / "batch_prediction_template.csv"
    if template_path.is_file():
        with open(template_path, "rb") as f:
            st.download_button(
                "Download CSV Schema Template",
                data=f,
                file_name="beanwise_batch_template.csv",
                mime="text/csv",
            )

    uploaded_file = st.file_uploader("Select CSV File for Batch Classification:", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Previewing Uploaded Data ({len(df_upload):,} records):")
            st.dataframe(df_upload.head(5), use_container_width=True)

            if st.button("Execute Batch Prediction", type="primary"):
                with st.spinner("Processing batch records..."):
                    success, result_df, summary, errors = predict_batch_csv(df_upload)

                if not success:
                    st.error("Batch Validation Failed:")
                    for err in errors:
                        st.write(f"- {err}")
                else:
                    st.success(f"Batch prediction completed: {summary['total_records']:,} observations in {summary['inference_duration_seconds']:.3f} seconds ({summary['per_sample_latency_ms']:.3f} ms/sample).")

                    # Class Distribution Chart
                    dist = summary["class_distribution"]
                    dist_df = pd.DataFrame(list(dist.items()), columns=["Cultivar", "Count"]).sort_values("Count", ascending=False)

                    fig_dist = px.bar(
                        dist_df,
                        x="Cultivar",
                        y="Count",
                        title="Predicted Variety Distribution in Uploaded Dataset",
                        color="Cultivar",
                        template="plotly_dark",
                        color_discrete_sequence=px.colors.qualitative.Dark24,
                    )
                    fig_dist.update_layout(
                        paper_bgcolor="#111827",
                        plot_bgcolor="#111827",
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=280,
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                    st.dataframe(result_df, use_container_width=True)

                    out_csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Full Prediction Results (CSV)",
                        data=out_csv,
                        file_name="beanwise_batch_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")


# ==============================================================================
# 3. BEAN VARIETIES GALLERY
# ==============================================================================
elif selected_section == "Bean Varieties":
    st.markdown("<div class='card-title'>Botanical Dry Bean Cultivar Profiles</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size: 13px; color: #9CA3AF; margin-bottom: 20px;">
            Reference photographs and botanical descriptions for the seven Turkish dry bean cultivars
            classified by the BEANWISE platform, sourced from agricultural repositories and scientific literature.
        </p>
        """,
        unsafe_allow_html=True,
    )

    varieties = get_all_varieties()

    # Display in 3 columns
    for row_start in range(0, len(varieties), 3):
        cols = st.columns(3)
        for idx in range(3):
            v_idx = row_start + idx
            if v_idx < len(varieties):
                v = varieties[v_idx]
                with cols[idx]:
                    img_path = v.get("image_path")
                    if img_path and Path(img_path).is_file():
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.info("Specimen image unavailable.")

                    st.markdown(
                        f"""
                        <div class="variety-card">
                            <div class="variety-card-body">
                                <div class="variety-card-title">{v['name']}</div>
                                <div class="variety-card-latin">{v['scientific_name']} &bull; {v['common_name']}</div>
                                <p class="variety-card-desc">{v['description']}</p>
                                <div style="font-size: 11px; color: #94A3B8; margin-top: 8px;">
                                    <strong>Culinary:</strong> {v['culinary_uses']}
                                </div>
                            </div>
                        </div>
                        <br>
                        """,
                        unsafe_allow_html=True,
                    )


# ==============================================================================
# 4. PREDICTION HISTORY
# ==============================================================================
elif selected_section == "Prediction History":
    st.markdown("<div class='card-title'>Session Prediction History</div>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No predictions recorded in current session. Perform an individual or batch prediction to log entries.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.write(f"Total session predictions: {len(history_df)}")

        display_cols = ["timestamp", "variety", "probability", "confidence", "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength"]
        valid_cols = [c for c in display_cols if c in history_df.columns]
        st.dataframe(history_df[valid_cols], use_container_width=True)

        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()
        with c2:
            hist_csv = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export History (CSV)",
                data=hist_csv,
                file_name="beanwise_prediction_history.csv",
                mime="text/csv",
            )


# ==============================================================================
# 5. MODEL INSIGHTS
# ==============================================================================
elif selected_section == "Model Insights":
    st.markdown("<div class='card-title'>Independent Model Evaluation & Performance Metrics</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size: 13px; color: #9CA3AF; margin-bottom: 16px;">
            Actual performance results evaluated on the held-out test partition (2,032 samples, 15.0% of dataset)
            strictly reserved during feature selection and model optimization.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="content-card">
                <div style="font-size: 11px; color: #9CA3AF; text-transform: uppercase;">Test Accuracy</div>
                <div style="font-size: 26px; font-weight: 700; color: #38BDF8; font-family: 'JetBrains Mono';">92.22%</div>
                <div style="font-size: 11px; color: #6B7280;">2,032 independent samples</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="content-card">
                <div style="font-size: 11px; color: #9CA3AF; text-transform: uppercase;">Balanced Accuracy</div>
                <div style="font-size: 26px; font-weight: 700; color: #38BDF8; font-family: 'JetBrains Mono';">93.26%</div>
                <div style="font-size: 11px; color: #6B7280;">Adjusted for class imbalance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="content-card">
                <div style="font-size: 11px; color: #9CA3AF; text-transform: uppercase;">Macro F1-Score</div>
                <div style="font-size: 26px; font-weight: 700; color: #38BDF8; font-family: 'JetBrains Mono';">0.9334</div>
                <div style="font-size: 11px; color: #6B7280;">Equal class weighting</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="content-card">
                <div style="font-size: 11px; color: #9CA3AF; text-transform: uppercase;">Inference Speed</div>
                <div style="font-size: 26px; font-weight: 700; color: #38BDF8; font-family: 'JetBrains Mono';">0.43 ms</div>
                <div style="font-size: 11px; color: #6B7280;">Average per sample latency</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Confusion Matrix Heatmap
    cm_path = REPORTS_DIR / "phase6" / "confusion_matrix.csv"
    if cm_path.is_file():
        cm_df = pd.read_csv(cm_path, index_index=True) if "index_index" in pd.read_csv.__code__.co_varnames else pd.read_csv(cm_path, index_col=0)
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            labels=dict(x="Predicted Cultivar", y="True Cultivar", color="Samples"),
            title="Independent Test Confusion Matrix (Counts)",
            color_continuous_scale="Blues",
            template="plotly_dark",
        )
        fig_cm.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            margin=dict(l=20, r=20, t=40, b=20),
            height=380,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Per-Class Performance Table & Chart
    per_class_path = REPORTS_DIR / "phase6" / "per_class_metrics.csv"
    if per_class_path.is_file():
        pc_df = pd.read_csv(per_class_path)
        col_pc_tbl, col_pc_chart = st.columns([1, 1.2])

        with col_pc_tbl:
            st.write("Per-Cultivar Test Set Performance Breakdown:")
            st.dataframe(pc_df, use_container_width=True, hide_index=True)

        with col_pc_chart:
            fig_pc = go.Figure()
            fig_pc.add_trace(go.Bar(name="Precision", x=pc_df["Cultivar"], y=pc_df["Precision"], marker_color="#2563EB"))
            fig_pc.add_trace(go.Bar(name="Recall", x=pc_df["Cultivar"], y=pc_df["Recall"], marker_color="#10B981"))
            fig_pc.add_trace(go.Bar(name="F1-Score", x=pc_df["Cultivar"], y=pc_df["F1-Score"], marker_color="#F59E0B"))
            fig_pc.update_layout(
                barmode="group",
                title="Precision, Recall, and F1 by Cultivar",
                template="plotly_dark",
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                yaxis=dict(range=[0.75, 1.02]),
                margin=dict(l=20, r=20, t=40, b=20),
                height=320,
            )
            st.plotly_chart(fig_pc, use_container_width=True)


# ==============================================================================
# 6. SETTINGS
# ==============================================================================
elif selected_section == "Settings":
    st.markdown("<div class='card-title'>Application Settings & System Configuration</div>", unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        st.markdown(
            """
            <div class="content-card">
                <div style="font-size: 14px; font-weight: 600; color: #F3F4F6; margin-bottom: 10px;">Platform Information</div>
                <div style="font-size: 13px; color: #9CA3AF; line-height: 1.6;">
                    <strong>Application Name:</strong> BEANWISE<br>
                    <strong>Version:</strong> 1.0.0 (Production Release)<br>
                    <strong>Primary Classifier:</strong> Support Vector Machine (RBF Kernel)<br>
                    <strong>Input Dimensions:</strong> 16 Physical Features (+ 6 Derived Morphological Ratios)<br>
                    <strong>Target Varieties:</strong> 7 Classes (SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON)<br>
                    <strong>Python Runtime:</strong> Python 3.12 (Global Environment)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s2:
        st.markdown(
            """
            <div class="content-card">
                <div style="font-size: 14px; font-weight: 600; color: #F3F4F6; margin-bottom: 10px;">Session Maintenance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Reset Session State & Clear History", type="secondary"):
            st.session_state.history = []
            st.session_state.example_inputs = None
            st.session_state.last_result = None
            st.success("Session state cleared successfully.")
            st.rerun()

# Clean Minimal Product Footer
st.markdown("---")
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: #6B7280; padding: 6px 0;">
        <span><strong>BEANWISE</strong> &bull; AI-Powered Dry Bean Classification Platform</span>
        <span>Production Machine Learning Pipeline &bull; Verified Test Accuracy 92.22%</span>
    </div>
    """,
    unsafe_allow_html=True,
)
