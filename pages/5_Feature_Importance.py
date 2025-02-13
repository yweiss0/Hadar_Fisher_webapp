import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

# File Path
DATA_DIR = "data/files_tab_4/"

st.set_page_config(page_title="Feature Importance Visualization", layout="wide")

st.title("💹 Feature Importance (SHAP Values)")

# Layout: Right side (1/4) for controls, Left side (3/4) for graph
col_space1, right_col, col_space2, left_col, col_space3 = st.columns([1, 2, 1, 6, 1])

with right_col:
    st.write("### Controls")

    # Dropdowns
    outcome = st.selectbox("Outcome", ["Anger", "Nervous", "Sad", "Negative Affect"])
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

    # **Fix for black dot issue**
    df.dropna(subset=["importance"], inplace=True)  # Remove NaN importance values

    # Get unique participants (keep original case for selection)
    participants = sorted(df["participant"].unique().tolist())

    # Dropdown for selecting a single participant
    selected_participant = st.selectbox("Select a Participant", participants)

    # Slider to filter features based on importance threshold
    min_importance_threshold = st.slider(
        "Minimum Feature Importance Threshold",
        min_value=0.0, 
        max_value=0.01, 
        value=0.0015,  # Default at 0.0015
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
    nlp_order = {"text feature": 1, "gpt": 2, "vader": 3, "liwc": 4}
    df_filtered["nlp_order"] = df_filtered["nlp"].map(nlp_order)

    # Sort features by NLP method first, then by importance
    df_sorted = df_filtered.sort_values(by=["nlp_order", "importance"], ascending=[True, False])

    # Define NLP method colors
    color_map = {
        "liwc": "red",
        "gpt": "blue",
        "vader": "green",
        "text feature": "black"
    }

    # Assign colors dynamically, ensuring no NaN values
    df_sorted["color"] = df_sorted["nlp"].map(color_map).fillna("black")

    # Create SHAP summary scatter plot
    fig = go.Figure()

    # Add vertical lines from 0 to the importance value
    for nlp_method, color in color_map.items():
        df_subset = df_sorted[df_sorted["nlp"] == nlp_method]
        for _, row in df_subset.iterrows():
            fig.add_trace(go.Scatter(
                x=[0, row["importance"]],  # Line from 0 to importance value
                y=[row["variable"], row["variable"]],  # Keep the same y-value
                mode="lines",
                line=dict(color=color, width=1),
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

    # Add scatter plot dots for feature importance values
    fig.add_trace(go.Scatter(
        x=df_sorted["importance"],
        y=df_sorted["variable"],
        mode="markers",
        marker=dict(
            size=10,
            color=df_sorted["color"],  # Coloring by NLP method
        ),
        name="Feature Importance",
        showlegend=False  # Ensure dots appear in legend
    ))

    # **Fix alignment: Adjust y-axis and remove unnecessary spacing**
    fig.update_layout(
        margin=dict(l=10, r=20, t=20, b=10),  # Reduce extra spacing
        xaxis_title="SHAP Value",
        yaxis_title="Features",
        template="plotly_white",
        height=750,  # Increased height to align better
        legend=dict(title="NLP Methods", orientation="v", x=1.05, y=0.96, ),  # Place legend vertically graph
        yaxis=dict(
            tickmode="array",
            tickvals=df_sorted["variable"],
            ticktext=[f"<span style='color:{color}'>{label}</span>" for label, color in zip(df_sorted["variable"], df_sorted["color"])],
        )
    )

    # **Show the plot at the top**
    st.plotly_chart(fig, use_container_width=True)
