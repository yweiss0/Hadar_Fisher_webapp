import streamlit as st
import pandas as pd
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
import new_app_chatbot

# File Path
DATA_DIR_1 = "data/files_tab_1_2/"
DATA_DIR_4 = "data/files_tab_7/"


def find_file_case_insensitive(directory, pattern):
    """
    Find a file in directory matching pattern (case-insensitive).
    Returns the actual file path if found, None otherwise.
    """
    # Create case-insensitive pattern using glob
    search_pattern = os.path.join(directory, pattern)

    # Try exact match first
    if os.path.exists(search_pattern):
        return search_pattern

    # If exact match fails, try case-insensitive search
    # Get all files in directory
    all_files = glob.glob(os.path.join(directory, "*"))

    # Convert pattern to lowercase for comparison
    pattern_lower = pattern.lower()

    for file_path in all_files:
        filename = os.path.basename(file_path)
        if filename.lower() == pattern_lower:
            return file_path

    return None


st.set_page_config(
    page_title="Compare High vs Low Performance", page_icon="📊", layout="wide"
)

st.title("📊 Compare High vs Low Performance")

# Center the entire layout
st.write(
    "<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True
)

# Create adjusted layout
col_space1, left_col, col_space2, right_col, col_space3 = st.columns([1, 2, 1, 6, 1])

with left_col:
    st.write("**Controls:**")

    # Dropdown Filters
    outcome = st.selectbox("Outcome", ["Negative Affect", "Angry", "Nervous", "Sad"])
    outcome = "na" if outcome.lower() == "negative affect" else outcome.lower()
    ml_model = st.selectbox("ML Model", ["Elastic Net (en)", "Random Forest (rf)"])
    ml_model_short = "en" if ml_model == "Elastic Net (en)" else "rf"

    # Load performance data with case-insensitive search
    perf_pattern = f"comb_{ml_model_short}_{outcome}_idiog.csv"
    perf_file = find_file_case_insensitive(DATA_DIR_1, perf_pattern)

    if not perf_file:
        st.error(f"Performance file not found: {perf_pattern}")
        st.stop()

    df_perf = pd.read_csv(perf_file)
    df_perf.columns = df_perf.columns.str.lower()

    # Ensure 'participant' and 'r2' columns exist
    if "participant" not in df_perf.columns or "r2" not in df_perf.columns:
        st.error("Required columns ('participant' or 'r2') are missing.")
        st.stop()

    # Clean and process data
    df_perf = df_perf.dropna(subset=["r2"])
    df_perf["r2"] = pd.to_numeric(df_perf["r2"], errors="coerce")
    df_perf = df_perf.dropna(subset=["r2"])

    if df_perf.empty:
        st.error("No valid R² data available.")
        st.stop()

    # Threshold Input (Dynamic)
    min_r2 = float(df_perf["r2"].min())
    max_r2 = float(df_perf["r2"].max())
    median_r2 = float(df_perf["r2"].median())

    threshold = st.slider(
        "R² Threshold",
        min_value=min_r2,
        max_value=max_r2,
        value=median_r2,
        step=0.01,
        format="%.2f",
    )

    # Categorize participants
    high_performance = df_perf[df_perf["r2"] >= threshold]["participant"].tolist()
    low_performance = df_perf[df_perf["r2"] < threshold]["participant"].tolist()

    # Display counts
    st.write(f"**High Performance (R² ≥ {threshold:.2f}):** {len(high_performance)}")
    st.write(f"**Low Performance (R² < {threshold:.2f}):** {len(low_performance)}")

    if not high_performance or not low_performance:
        st.warning("At least one group has no participants. Adjust the threshold.")
        st.stop()

with right_col:
    st.write("")  # Spacer

    # Load feature importance data with case-insensitive search
    feat_pattern = f"Featureimportance_{ml_model_short}_comb_{outcome}_abs.csv"
    feat_file = find_file_case_insensitive(DATA_DIR_4, feat_pattern)

    if not feat_file:
        st.error(f"Feature importance file not found: {feat_pattern}")
        st.stop()

    df_feat = pd.read_csv(feat_file, encoding="ISO-8859-1")
    df_feat.columns = df_feat.columns.str.lower()

    # Check required columns
    if not all(
        col in df_feat.columns for col in ["participant", "variable", "importance"]
    ):
        st.error(
            "Required columns ('participant', 'variable', 'importance') are missing from feature importance file."
        )
        st.stop()

    # Normalize participant column
    df_feat["participant"] = df_feat["participant"].astype(str)
    high_performance = [str(p) for p in high_performance]
    low_performance = [str(p) for p in low_performance]

    # Filter data by performance groups
    df_high = df_feat[df_feat["participant"].isin(high_performance)]
    df_low = df_feat[df_feat["participant"].isin(low_performance)]

    if df_high.empty or df_low.empty:
        st.error("No feature data available for one or both performance groups.")
        st.stop()

    # Calculate mean absolute feature importance for each group
    mean_high = df_high.groupby("variable")["importance"].mean().reset_index()
    mean_low = df_low.groupby("variable")["importance"].mean().reset_index()
    mean_high.columns = ["Variable", "High_Performance"]
    mean_low.columns = ["Variable", "Low_Performance"]

    # Merge and calculate difference
    merged = pd.merge(mean_high, mean_low, on="Variable", how="outer").fillna(0)
    merged["Difference"] = merged["High_Performance"] - merged["Low_Performance"]

    # Sort by absolute difference
    merged = merged.reindex(
        merged["Difference"].abs().sort_values(ascending=False).index
    )

    # Take top 20 features
    top_features = merged.head(20)

    if top_features.empty:
        st.error("No feature differences to display.")
        st.stop()

    # Create horizontal bar chart
    fig = go.Figure()

    # Add bars for high performance
    fig.add_trace(
        go.Bar(
            y=top_features["Variable"],
            x=top_features["High_Performance"],
            name=f"High Performance (≥{threshold:.2f})",
            orientation="h",
            marker_color="green",
            opacity=0.7,
        )
    )

    # Add bars for low performance
    fig.add_trace(
        go.Bar(
            y=top_features["Variable"],
            x=top_features["Low_Performance"],
            name=f"Low Performance (<{threshold:.2f})",
            orientation="h",
            marker_color="red",
            opacity=0.7,
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Top 20 Feature Importance Differences: High vs Low R² Performance",
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Features",
        template="plotly_white",
        height=800,
        barmode="group",
        yaxis=dict(categoryorder="total ascending"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display data table
    st.write("### Feature Importance Comparison Table")
    st.dataframe(
        top_features[["Variable", "High_Performance", "Low_Performance", "Difference"]],
        height=400,
    )

st.write("</div>", unsafe_allow_html=True)

# Add the chatbot to the page
new_app_chatbot.show_chatbot_ui(page_name="Compare High vs Low Performance")
