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
        min_value=0.0, 
        max_value=0.01, 
        value=0.0065,  # Default at 0.0065
        step=0.0001,
        format="%.4f"
    )

with left_col:
    st.write("")  # Spacer

    # Filter data based on selected participants
    df_filtered = df[df["participant"].isin(selected_participants)]

    # Aggregate duplicate entries before pivoting
    df_filtered = df_filtered.groupby(["participant", "variable"], as_index=False)["importance"].mean()

    # Apply the importance threshold to filter out low-importance features
    filtered_features = df_filtered.groupby("variable")["importance"].max()
    valid_features = filtered_features[filtered_features >= min_importance_threshold].index
    df_filtered = df_filtered[df_filtered["variable"].isin(valid_features)]

    # Pivot for heatmap
    heatmap_data = df_filtered.pivot(index="participant", columns="variable", values="importance").fillna(0)

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
        "text feature": "black"
    }

    # Assign colors based on NLP method (default to black if unknown)
    feature_colors = [color_map.get(nlp_methods.get(var, "text feature"), "black") for var in heatmap_data.columns]

    # Create custom HTML tick labels to color them
    colored_labels = [
        f"<span style='color:{color}'>{label}</span>"
        for label, color in zip(heatmap_data.columns, feature_colors)
    ]

    # Set color scale based on model
    if ml_model_short == "en":
        color_scale = [
            [0.0, "purple"], [0.5, "green"], [0.9, "yellow"], [1.0, "red"]
        ]
        vmax_value = 0.040
    else:  # RF model color scale (Blue → Orange → Red)
        color_scale = [
            [0.0, "blue"], [0.5, "orange"], [0.9, "red"], [1.0, "red"]
        ]
        vmax_value = 0.10

    # Adjust heatmap height dynamically based on number of participants
    num_participants = len(selected_participants)
    heatmap_height = min(max(600, num_participants * 40), 1500)  # Min 600px, Max 1500px

    # Create heatmap using Plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,  # Use original feature names
            y=heatmap_data.index,
            colorscale=color_scale,
            colorbar=dict(title="SHAP Value"),
            zmin=0,
            zmax=vmax_value,
        )
    )

    # Customize layout for better readability
    fig.update_layout(
        
        xaxis=dict(
            title="Feature", 
            tickangle=90, 
            tickmode="array", 
            tickvals=list(range(len(heatmap_data.columns))),
            ticktext=colored_labels  # Use colored HTML labels
        ),
        yaxis=dict(title="Participant", tickfont=dict(size=10)),
        autosize=False,
        height=heatmap_height  # Dynamically adjusted height
    )

    # Show Plotly figure
    st.plotly_chart(fig, use_container_width=True)

    # Add centered and smaller legend title below the heatmap
    st.write("")
    st.markdown(
        "<h5 style='text-align: center;'>Feature Reference (NLP Method Coloring)</h5>", 
        unsafe_allow_html=True
    )

    # Add legend **below** the heatmap, centered
    legend_html = """
    <div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-top: 5px;">
        <div style="width: 20px; height: 10px; background-color: red;"></div> <span>LIWC</span>
        <div style="width: 20px; height: 10px; background-color: blue;"></div> <span>GPT</span>
        <div style="width: 20px; height: 10px; background-color: green;"></div> <span>VADER</span>
        <div style="width: 20px; height: 10px; background-color: black;"></div> <span>Text Feature</span>
    </div>
    """
    st.write(legend_html, unsafe_allow_html=True)


# Add the chatbot to the page
app_with_chatbot.show_chatbot_ui()