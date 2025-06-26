import streamlit as st
import pandas as pd
import os
import glob
import plotly.graph_objects as go
import new_app_chatbot
import numpy as np  # Needed for dynamic range options

# File Path
DATA_DIR = "data/files_tab_4/"
METRICS_DIR = "data/files_tab_1_2/"  # Directory for R² and RMSE values


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
    page_title="Feature Importance Visualization", page_icon="📊", layout="wide"
)

st.title("📉 Feature importance per participant (SHAP Value)")

# Layout: Right side (1/4) for controls, Left side (3/4) for graph
col_space1, right_col, col_space2, left_col, col_space3 = st.columns([1, 2, 1, 6, 1])

with right_col:
    st.write("### Controls")

    # Dropdowns
    outcome = st.selectbox(
        "Outcome",
        [
            "Negative Affect",
            "Angry",
            "Nervous",
            "Sad",
        ],
    )
    outcome = "na" if outcome.lower() == "negative affect" else outcome.lower()

    # Dropdown for selecting model
    ml_model = st.selectbox(
        "Model", ["Elastic Net (EN)", "Random Forest (RF)"], key="ml_model"
    )
    ml_model_short = "en" if ml_model == "Elastic Net (EN)" else "rf"

    # Check if the model selection has changed and update the default slider accordingly.
    if "prev_ml_model_short" not in st.session_state:
        st.session_state.prev_ml_model_short = ml_model_short
    else:
        if st.session_state.prev_ml_model_short != ml_model_short:
            st.session_state.prev_ml_model_short = ml_model_short
            # Update the symmetric slider default based on the model.
            if ml_model_short == "rf":
                st.session_state.symmetric_val = (-0.0001, 0.0001)
            else:
                st.session_state.symmetric_val = (-0.005, 0.005)
            st.rerun()  # Rerun to apply the new default

    # Checkbox for ABS values
    use_abs = st.checkbox("Use absolute values", value=False)
    # tooltip for ABS values

    st.markdown(
        """
    <style>
        .tooltip-container {
            display: inline-block;
            position: relative;
        }
        .tooltip-container .tooltip-text {
            visibility: hidden;
            width: 350px;
            background-color: #555;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -175px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 14px;
        }
        .tooltip-container:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
    </style>
    <div style="font-size: 12px; margin-top: 5px; padding: 8px; border-radius: 5px; background-color: #f0f2f6;">
        <b>General Rule:</b> Choose absolute values for feature ranking and non-absolute values for understanding directional effects.
        <span class="tooltip-container" style="margin-left: 10px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24">
                <g fill="none">
                    <path d="m12.593 23.258l-.011.002l-.071.035l-.02.004l-.014-.004l-.071-.035q-.016-.005-.024.005l-.004.01l-.017.428l.005.02l.01.013l.104.074l.015.004l.012-.004l.104-.074l.012-.016l.004-.017l-.017-.427q-.004-.016-.017-.018m.265-.113l-.013.002l-.185.093l-.01.01l-.003.011l.018.43l.005.012l.008.007l.201.093q.019.005.029-.008l.004-.014l-.034-.614q-.005-.018-.02-.022m-.715.002a.02.02 0 0 0-.027.006l-.006.014l-.034.614q.001.018.017.024l.015-.002l.201-.093l.01-.008l.004-.011l.017-.43l-.003-.012l-.01-.01z"/>
                    <path fill="currentColor" d="M12 2c5.523 0 10 4.477 10 10s-4.477 10-10 10S2 17.523 2 12S6.477 2 12 2m0 2a8 8 0 1 0 0 16a8 8 0 0 0 0-16m-.01 6c.558 0 1.01.452 1.01 1.01v5.124A1 1 0 0 1 12.5 18h-.49A1.01 1.01 0 0 1 11 16.99V12a1 1 0 1 1 0-2zM12 7a1 1 0 1 1 0 2a1 1 0 0 1 0-2"/>
                </g>
            </svg>
            <span class="tooltip-text">
                <b>Absolute SHAP Values:</b> Show overall importance of each feature in predicting negative affect, regardless of whether the feature's impact increases or decreases negative affect. Use this to understand which features are most influential in the model's predictions, regardless of direction. <br>
                <b>Non-Absolute SHAP Values:</b> Show whether a feature predicts higher or lower negative affect. Use this if you need to interpret how each feature affects the predicted outcome (increasing or decreasing negative affect).
            </span>
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if use_abs:
        DATA_DIR = "data/files_tab_7/"

    # Checkbox for including 'Time' variable (default checked)
    st.write("")
    include_time = st.checkbox("Include the variable 'Time'", value=True)

    # File selection based on user inputs with case-insensitive search
    if use_abs:
        file_name = f"Featureimportance_{ml_model_short}_comb_{outcome}_abs.csv"
    else:
        file_name = f"Featureimportance_{ml_model_short}_comb_{outcome}.csv"
    file_path = find_file_case_insensitive(DATA_DIR, file_name)

    if not file_path:
        st.error(f"File not found: {file_name}")
        st.stop()

    # Load data with case-insensitive column handling
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    df.columns = df.columns.str.lower()

    # Ensure correct column names exist
    required_columns = ["participant", "variable", "importance", "nlp"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()

    # Normalize case for NLP column and handle missing values
    df["nlp"] = df["nlp"].str.lower().fillna("text feature")
    df["participant"] = (
        df["participant"].astype(str).str.lower()
    )  # Case-insensitive participants
    df["variable"] = df["variable"].astype(
        str
    )  # Ensure variable (feature) names are strings
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    df.dropna(subset=["importance"], inplace=True)

    # If not including 'Time', remove rows where variable equals "time" (case-insensitive)
    if not include_time:
        df = df[~df["variable"].str.lower().eq("time")]

    # Get unique participants (keep original case for selection)
    participants = sorted(df["participant"].unique().tolist())
    # Dropdown for selecting a single participant
    selected_participant = st.selectbox("Select a Participant", participants)

    # Load performance metrics file with case-insensitive search
    fixed_outcome = "angry" if outcome == "anger" else outcome
    metrics_file_name = f"comb_{ml_model_short}_{fixed_outcome}_idiog.csv"
    metrics_file_path = find_file_case_insensitive(METRICS_DIR, metrics_file_name)

    r2_value, rmse_value = None, None

    if metrics_file_path and os.path.exists(metrics_file_path):
        df_metrics = pd.read_csv(metrics_file_path, encoding="ISO-8859-1")
        df_metrics.columns = df_metrics.columns.str.lower()
        required_metrics_columns = ["participant", "r2", "rmse"]
        missing_metrics_columns = [
            col for col in required_metrics_columns if col not in df_metrics.columns
        ]
        if not missing_metrics_columns:
            df_metrics["participant"] = (
                df_metrics["participant"].astype(str).str.lower()
            )
            participant_metrics = df_metrics[
                df_metrics["participant"] == selected_participant
            ]
            if not participant_metrics.empty:
                r2_value = participant_metrics["r2"].values[0]
                rmse_value = participant_metrics["rmse"].values[0]

    # --- Dynamic Symmetric Range Selector Component ---
    options = [round(x, 4) for x in np.arange(-0.01, 0.01 + 0.0001, 0.0001)]
    if "symmetric_val" not in st.session_state:
        st.session_state.symmetric_val = (-0.0015, 0.0015)
    if "prev_val" not in st.session_state:
        st.session_state.prev_val = st.session_state.symmetric_val

    def update_symmetric():
        current = st.session_state.symmetric_val
        prev = st.session_state.prev_val
        lower, upper = current
        prev_lower, prev_upper = prev
        if lower != prev_lower:
            st.session_state.symmetric_val = (lower, -lower)
        elif upper != prev_upper:
            st.session_state.symmetric_val = (-upper, upper)
        st.session_state.prev_val = st.session_state.symmetric_val

    slider_value = st.select_slider(
        "Minimum Feature Importance Threshold",
        options=options,
        value=st.session_state.symmetric_val,
        key="symmetric_val",
        on_change=update_symmetric,
    )

    st.markdown(
        f"""
        <div style="font-size: 12px; margin-top: 5px; padding: 8px; border-radius: 5px; background-color: #f0f2f6;">
            <b style="color: green;">✅ Data Included:</b> <br> Importance values &lt; <b>{slider_value[0]:.4f}</b> and &gt; <b>{slider_value[1]:.4f}</b><br>
            <b style="color: red;">❌ Data Filtered Out:</b> <br> Importance values between <b>{slider_value[0]:.4f}</b> and <b>{slider_value[1]:.4f}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    min_importance_threshold = slider_value[1]
    # --- End Dynamic Component ---

with left_col:
    # Filter data for the selected participant
    df_filtered = df[df["participant"] == selected_participant].copy()

    # Apply new importance threshold filter logic
    df_filtered = df_filtered[
        (df_filtered["importance"] <= -abs(min_importance_threshold))
        | (df_filtered["importance"] >= abs(min_importance_threshold))
    ]
    if df_filtered.empty:
        st.warning("No features meet the selected importance threshold.")
        st.stop()

    # Define sorting order for NLP methods
    nlp_order = {
        "text leangth": 1,
        "gpt": 2,
        "vader": 3,
        "liwc": 4,
        "lda": 5,
        "time": 6,
        "text feature": 7,
    }
    df_filtered["nlp_order"] = df_filtered["nlp"].map(nlp_order)
    df_sorted = df_filtered.sort_values(
        by=["nlp_order", "importance"], ascending=[True, False]
    )

    # Ensure Equal Spacing for Features
    feature_list = df_sorted["variable"].unique().tolist()
    y_positions = list(range(len(feature_list)))
    feature_y_map = {feature: y for feature, y in zip(feature_list, y_positions)}
    df_sorted["y_position"] = df_sorted["variable"].map(feature_y_map)

    # Dynamically adjust graph height
    num_features = len(feature_list)
    height_per_feature = 50
    min_height = 350
    max_height = 1800
    dynamic_height = min(max_height, max(min_height, num_features * height_per_feature))

    # Define NLP method colors
    color_map = {
        "liwc": "red",
        "gpt": "blue",
        "vader": "green",
        "text leangth": "black",
        "time": "purple",
        "lda": "orange",
        "text feature": "brown",
    }
    df_sorted["color"] = df_sorted["nlp"].map(color_map).fillna("black")

    # Create SHAP summary scatter plot
    fig = go.Figure()
    for nlp_method, color in color_map.items():
        df_subset = df_sorted[df_sorted["nlp"] == nlp_method]
        for _, row in df_subset.iterrows():
            line_width = 3 if row["nlp"] != "text leangth" else 2
            fig.add_trace(
                go.Scatter(
                    x=[0, row["importance"]],
                    y=[row["y_position"], row["y_position"]],
                    mode="lines",
                    line=dict(color=color, width=line_width),
                    hoverinfo="text",
                    hovertext=f"Feature: {row['variable']}<br>Importance: {row['importance']:.4f}",
                    showlegend=False,
                )
            )
    for nlp_method, color in color_map.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=nlp_method.upper(),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=df_sorted["importance"],
            y=df_sorted["y_position"],
            mode="markers",
            marker=dict(
                size=17,
                color=df_sorted["color"],
            ),
            hoverinfo="text",
            hovertext=[
                f"Feature: {feat}<br>Importance: {imp:.4f}"
                for feat, imp in zip(df_sorted["variable"], df_sorted["importance"])
            ],
            name="Feature Importance",
            showlegend=False,
        )
    )
    fig.update_layout(
        title={
            "text": f"R²: {r2_value:.4f}  |  RMSE: {rmse_value:.4f}",
            "x": 0.5,
            "y": 0.97,
            "xanchor": "center",
            "yanchor": "top",
        },
        margin=dict(l=10, r=20, t=20, b=10),
        xaxis_title="SHAP Value",
        yaxis_title="Features",
        template="plotly_white",
        height=dynamic_height,
        legend=dict(title="NLP Methods", orientation="v", x=1.05, y=0.96),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(tickmode="array", tickvals=y_positions, ticktext=feature_list),
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=-0.01,
                y0=-0.0001,
                x1=1.01,
                y1=0.96,
                line=dict(color="black", width=0.8),
            ),
            dict(
                type="line",
                x0=0,
                x1=0,
                y0=min(y_positions) - 1,
                y1=max(y_positions) + 1,
                xref="x",
                yref="y",
                line=dict(color="black", width=0.3, dash="dash"),
            ),
        ],
    )
    st.plotly_chart(fig, use_container_width=True)
    new_app_chatbot.show_chatbot_ui(
        page_name="Feature Importance per Participant (SHAP Value)"
    )
