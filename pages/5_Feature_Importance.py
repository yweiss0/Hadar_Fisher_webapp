import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import app_with_chatbot

# File Path
DATA_DIR = "data/files_tab_4/"
METRICS_DIR = "data/files_tab_1_2/"  # Directory for R² and RMSE values

st.set_page_config(page_title="Feature Importance Visualization", layout="wide")

st.title("💹 Feature Importance (SHAP Values)")

# Layout: Right side (1/4) for controls, Left side (3/4) for graph
col_space1, right_col, col_space2, left_col, col_space3 = st.columns([1, 2, 1, 6, 1])

with right_col:
    st.write("### Controls")

    # Dropdowns
    outcome = st.selectbox("Outcome", ["Negative Affect", "Angry", "Nervous", "Sad", ])
    outcome = "na" if outcome.lower() == "negative affect" else outcome.lower()

    ml_model = st.selectbox("Model", ["Elastic Net (EN)", "Random Forest (RF)"])
    ml_model_short = "en" if ml_model == "Elastic Net (EN)" else "rf"

    # File selection based on user inputs
    file_name = f"Featureimportance_{ml_model_short}_comb_{outcome}.csv"
    file_path = os.path.join(DATA_DIR, file_name)

    if not os.path.exists(file_path):
        st.error(f"File not found: {file_name}")
        st.stop()

    # Load data with case-insensitive column handling
    df = pd.read_csv(file_path, encoding="ISO-8859-1")

    # Convert all column names to lowercase for case-insensitive handling
    df.columns = df.columns.str.lower()

    # Ensure correct column names exist
    required_columns = ["participant", "variable", "importance", "nlp"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()

    # Normalize case for NLP column and handle missing values
    df["nlp"] = df["nlp"].str.lower().fillna("text feature")
    df["participant"] = df["participant"].astype(str).str.lower()  # Case-insensitive participants
    df["variable"] = df["variable"].astype(str)  # Ensure variable (feature) names are strings

    # Ensure numeric conversion
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")

    # Remove NaN importance values
    df.dropna(subset=["importance"], inplace=True)

    # Get unique participants (keep original case for selection)
    participants = sorted(df["participant"].unique().tolist())

    # Dropdown for selecting a single participant
    selected_participant = st.selectbox("Select a Participant", participants)

    # Load performance metrics file
    fixed_outcome = "angry" if outcome == "anger" else outcome
    metrics_file_name = f"comb_{ml_model_short}_{fixed_outcome}_idiog.csv"
    metrics_file_path = os.path.join(METRICS_DIR, metrics_file_name)

    r2_value, rmse_value = None, None

    if os.path.exists(metrics_file_path):
        df_metrics = pd.read_csv(metrics_file_path, encoding="ISO-8859-1")
        
        # Convert all column names to lowercase
        df_metrics.columns = df_metrics.columns.str.lower()

        # Ensure it has required columns
        required_metrics_columns = ["participant", "r2", "rmse"]
        missing_metrics_columns = [col for col in required_metrics_columns if col not in df_metrics.columns]

        if not missing_metrics_columns:
            # Convert participant column to lowercase for case-insensitive matching
            df_metrics["participant"] = df_metrics["participant"].astype(str).str.lower()
            
            # Find the participant's row
            participant_metrics = df_metrics[df_metrics["participant"] == selected_participant]

            if not participant_metrics.empty:
                r2_value = participant_metrics["r2"].values[0]
                rmse_value = participant_metrics["rmse"].values[0]


    # Slider to filter features based on importance threshold
    min_importance_threshold = st.slider(
        "Minimum Feature Importance Threshold",
        min_value=-0.01, 
        max_value=0.01, 
        value=-0.0020,  # Default at -0.0020
        step=0.0001,
        format="%.4f"
    )

# **Move the graph higher up in the left column**
with left_col:

    # Filter data for the selected participant
    df_filtered = df[df["participant"] == selected_participant].copy()

    # Apply importance threshold
    df_filtered = df_filtered[df_filtered["importance"] >= min_importance_threshold]

    if df_filtered.empty:
        st.warning("No features meet the selected importance threshold.")
        st.stop()

    # Define sorting order for NLP methods
    nlp_order = {"text leangth": 1, "gpt": 2, "vader": 3, "liwc": 4, "lda": 5, "time": 6}
    df_filtered["nlp_order"] = df_filtered["nlp"].map(nlp_order)

    # Sort features by NLP method first, then by importance
    df_sorted = df_filtered.sort_values(by=["nlp_order", "importance"], ascending=[True, False])

    # **Ensure Equal Spacing for Features**
    feature_list = df_sorted["variable"].unique().tolist()
    y_positions = list(range(len(feature_list)))  # Evenly spaced Y values
    feature_y_map = {feature: y for feature, y in zip(feature_list, y_positions)}
    df_sorted["y_position"] = df_sorted["variable"].map(feature_y_map)  # Assign equal spacing

    # **Dynamically adjust graph height**
    num_features = len(feature_list)  # Number of features after filtering
    height_per_feature = 50  # Pixels per feature (adjustable)
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
    }

    # Assign colors dynamically, ensuring no NaN values
    df_sorted["color"] = df_sorted["nlp"].map(color_map).fillna("black")

    # Create SHAP summary scatter plot
    fig = go.Figure()

    # Add vertical lines from 0 to the importance value (with tooltip)
    for nlp_method, color in color_map.items():
        df_subset = df_sorted[df_sorted["nlp"] == nlp_method]
        for _, row in df_subset.iterrows():
            line_width = 3 if row["nlp"] != "text leangth" else 2  # Ensure black lines are always visible
            fig.add_trace(go.Scatter(
                x=[0, row["importance"]],  # Line from 0 to importance value
                y=[row["y_position"], row["y_position"]],  # Keep the same y-value
                mode="lines",
                line=dict(color=color, width=line_width),
                hoverinfo="text",
                hovertext=f"Feature: {row['variable']}<br>Importance: {row['importance']:.4f}",
                showlegend=False
            ))

    # **Fix: Ensure all NLP types appear in legend**
    for nlp_method, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None],  # Dummy invisible point
            y=[None],  # Dummy invisible point
            mode="markers",
            marker=dict(size=10, color=color),
            name=nlp_method.upper()  # Add NLP method name to legend
        ))

    # Add scatter plot dots for feature importance values (with tooltip)
    fig.add_trace(go.Scatter(
        x=df_sorted["importance"],
        y=df_sorted["y_position"],
        mode="markers",
        marker=dict(
            size=17,
            color=df_sorted["color"],  # Coloring by NLP method
        ),
        hoverinfo="text",
        hovertext=[f"Feature: {feat}<br>Importance: {imp:.4f}" for feat, imp in zip(df_sorted["variable"], df_sorted["importance"])],
        name="Feature Importance",
        showlegend=False  # Ensure dots appear in legend but not in legend box
    ))

    # **Fix alignment & adjust border size**
    fig.update_layout(
        title={
        'text': f"R²: {r2_value:.4f}  |  RMSE: {rmse_value:.4f}",
        'x': 0.5,  # x-position of the title (0-1)
        'y': 0.97,  # y-position of the title (0-1)
        'xanchor': 'center',  # horizontal alignment ('left', 'center', 'right')
        'yanchor': 'top'  # vertical alignment ('top', 'middle', 'bottom')
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
        xref="paper", yref="paper",
        x0=-0.01, y0=-0.0001, x1=1.01, y1=0.96,  # **Cover the entire plot area**
        line=dict(color="black", width=0.8)
    ),
    dict(
        type="line",
        x0=0, x1=0,  # Vertical line at x=0
        y0=min(y_positions) - 1, y1=max(y_positions) + 1,  # Extend slightly beyond feature range
        xref="x", yref="y",
        line=dict(color="black", width=0.3, dash="dash")  # Dashed line style
    )
]
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add the chatbot to the page
    app_with_chatbot.show_chatbot_ui()


# WORKING