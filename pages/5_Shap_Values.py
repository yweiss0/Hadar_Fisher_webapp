import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import app_with_chatbot

# File Path
DATA_DIR = "data/files_tab_4/"

st.set_page_config(page_title="Feature Importance Heatmap", layout="wide")

st.title("📉 Feature Importance Heatmap (SHAP Values)")

# Layout: Right side (1/4) for controls, Left side (3/4) for heatmap
col_space1, right_col, col_space2, left_col, col_space3 = st.columns([1, 2, 1, 6, 1])

with right_col:
    st.write("### Controls")

    # Dropdowns
    outcome = st.selectbox("Outcome", ["Negative Affect", "Angry", "Nervous", "Sad"])
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

    # Convert column names to lowercase for consistency
    df.columns = df.columns.str.lower()

    # Ensure correct column names
    required_columns = ["participant", "variable", "importance", "nlp"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()

    # Ensure numeric conversion
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")

    # Get unique participants
    participants = df["participant"].unique().tolist()

    # Multi-select checkboxes for participants
    st.write("**Select Participants:**")
    select_all = st.checkbox("All", value=False)

    if select_all:
        selected_participants = participants  # Select all by default
    else:
        selected_participants = st.multiselect(
            "Participants", participants, default=participants[:10]
        )

    # Slider to filter features based on importance threshold
    min_importance_threshold = st.slider(
        "Minimum Feature Importance Threshold",
        min_value=-0.01, 
        max_value=0.01, 
        value=0.005,  # Default set to  0.005
        step=0.0001,
        format="%.4f"
    )

with left_col:
    # Filter data based on selected participants
    df_filtered = df[df["participant"].isin(selected_participants)]

    # Aggregate duplicate entries before pivoting
    df_filtered = df_filtered.groupby(["participant", "variable"], as_index=False)["importance"].mean()

    # Apply the importance threshold to filter out values between -threshold and threshold
    df_filtered = df_filtered[(df_filtered["importance"] <= -min_importance_threshold) | (df_filtered["importance"] >= min_importance_threshold)]

    # Pivot for heatmap (Swapped axes: participants → x, features → y)
    heatmap_data = df_filtered.pivot(index="variable", columns="participant", values="importance").fillna(0)

    if heatmap_data.empty:
        st.warning("No features meet the selected importance threshold.")
        st.stop()

    # NLP method coloring (Case-insensitive matching)
    df["nlp"] = df["nlp"].str.lower()  # Normalize NLP column to lowercase
    nlp_methods = df.set_index("variable")["nlp"].to_dict()

    # Define NLP method colors
    color_map = {
        "liwc": "red",
        "gpt": "blue",
        "vader": "green",
        "text length": "black",
        "time": "purple",
        "lda": "orange",
    }

    # Assign colors based on NLP method (default to black if unknown)
    feature_colors = [color_map.get(nlp_methods.get(var, "text length"), "black") for var in heatmap_data.index]

    # Create color scale for positive and negative values
    color_scale = [
    [0.0, "rgb(75, 48, 163)"],  # Deep Blue (Strong Negative)
    [0.25, "rgb(188, 170, 220)"],  # Blue
    [0.5, "rgb(247, 247, 247)"],  # White (Neutral)
    [0.75, "rgb(215, 48, 39)"],  # Red
    [1.0, "rgb(165, 0, 38)"],  # Deep Red (Strong Positive)
]
    

    # Adjust heatmap height dynamically based on number of features
    num_features = len(heatmap_data)
    heatmap_height = min(max(600, num_features * 40), 1000)  # Min 600px, Max 1000px

    # Determine max absolute value for symmetric color scaling
    vmax_value = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
    hover_text = [[f"Participant: {p}<br>Feature: {f}<br>Importance: {heatmap_data.at[f, p]:.4f}" for p in heatmap_data.columns] for f in heatmap_data.index]

    # Create heatmap using Plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,  # Participants on X-axis
            y=heatmap_data.index,  # Features on Y-axis
            colorscale=color_scale,
            colorbar=dict(title="SHAP Value"),
            zmin=-vmax_value,
            zmax=vmax_value,
            text=hover_text,
            hoverinfo="text"
        ),
    )

    # Update y-axis with colored labels
    fig.update_layout(
        height=heatmap_height,
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(heatmap_data.index))),
            ticktext=[f'<span style="color:{color}">{label}</span>' for label, color in zip(heatmap_data.index, feature_colors)],
        )
    )

    # Show Plotly figure
    st.plotly_chart(fig, use_container_width=True)

    # Add the chatbot to the page
app_with_chatbot.show_chatbot_ui()
