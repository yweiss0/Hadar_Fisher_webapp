import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import app_with_chatbot
import numpy as np

# File Path
DATA_DIR = "data/files_tab_4/"

st.set_page_config(page_title="Feature Importance Heatmap", page_icon="📊", layout="wide")

st.title("📉 feature importance Heatmap (SHAP Values)")

# Layout: Right side (1/4) for controls, Left side (3/4) for heatmap
col_space1, right_col, col_space2, left_col, col_space3 = st.columns([1, 2, 1, 6, 1])

with right_col:
    st.write("### Controls")

    # Dropdowns
    outcome = st.selectbox("Outcome", ["Negative Affect", "Angry", "Nervous", "Sad"])
    outcome = "na" if outcome.lower() == "negative affect" else outcome.lower()

    # Dropdown for selecting model
    ml_model = st.selectbox("Model", ["Elastic Net (EN)", "Random Forest (RF)"], key="ml_model")
    ml_model_short = "en" if ml_model == "Elastic Net (EN)" else "rf"

    # Check if the model selection has changed and update the default slider accordingly.
    if "prev_ml_model_short" not in st.session_state:
        st.session_state.prev_ml_model_short = ml_model_short
    else:
        if st.session_state.prev_ml_model_short != ml_model_short:
            st.session_state.prev_ml_model_short = ml_model_short
            # Update the symmetric slider default based on the model.
            if ml_model_short == "rf":
                st.session_state.symmetric_val = (-0.0005, 0.0005)
            else:
                st.session_state.symmetric_val = (-0.005, 0.005)
            st.rerun()  # Rerun to apply the new default

    # Checkbox for ABS values
    use_abs = st.checkbox("Use ABS values only", value=False)
    if use_abs:
        DATA_DIR = "data/files_tab_7/"
        st.markdown(
            '<div style="font-size: 12px; margin-top: 5px; padding: 8px; border-radius: 5px; background-color: #f0f2f6;">All values are in ABS.</div>',
            unsafe_allow_html=True
        )

    # File selection based on user inputs
    if use_abs:
        file_name = f"Featureimportance_{ml_model_short}_comb_{outcome}_abs.csv"
    else:
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

    # --- Dynamic Symmetric Range Selector Component ---
    # Create a list of options with 0.0001 increments.
    options = [round(x, 4) for x in np.arange(-0.01, 0.01 + 0.0001, 0.0001)]

    # Initialize symmetric slider state if not already set.
    if 'symmetric_val' not in st.session_state:
        st.session_state.symmetric_val = (-0.005, 0.005)  # Default fallback

    if 'prev_val' not in st.session_state:
        st.session_state.prev_val = st.session_state.symmetric_val

    def update_symmetric():
        # Compare current and previous values to decide which handle was moved.
        current = st.session_state.symmetric_val
        prev = st.session_state.prev_val
        lower, upper = current
        prev_lower, prev_upper = prev

        # If the lower handle changed, update the upper handle to its negative.
        if lower != prev_lower:
            st.session_state.symmetric_val = (lower, -lower)
        # If the upper handle changed, update the lower handle to its negative.
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

    # Display messages with clear formatting and smaller font
    st.markdown(
        f"""
        <div style="font-size: 12px; margin-top: 5px; padding: 8px; border-radius: 5px; background-color: #f0f2f6;">
            <b style="color: green;">✅ Data Included:</b> <br> Importance values < <b>{slider_value[0]:.4f}</b> and > <b>{slider_value[1]:.4f}</b><br>
            <b style="color: red;">❌ Data Filtered Out:</b> <br> Importance values between <b>{slider_value[0]:.4f}</b> and <b>{slider_value[1]:.4f}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Use the positive threshold from the symmetric range as the filtering threshold
    min_importance_threshold = slider_value[1]
    # --- End Dynamic Component ---

with left_col:
    # Filter data based on selected participants
    df_filtered = df[df["participant"].isin(selected_participants)]

    # Aggregate duplicate entries before pivoting
    df_filtered = df_filtered.groupby(["participant", "variable"], as_index=False)["importance"].mean()

    # Apply the importance threshold to filter out values between -threshold and threshold
    df_filtered = df_filtered[
        (df_filtered["importance"] <= -min_importance_threshold) | (df_filtered["importance"] >= min_importance_threshold)
    ]

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

    # --- Group & Sort y-axis by NLP Method ---
    # Define the desired order: text length, LIWC, time, gpt, vader, lda.
    nlp_order = {"text length": 1, "liwc": 2, "time": 3, "gpt": 4, "vader": 5, "lda": 6}
    sorted_features = sorted(
        heatmap_data.index,
        key=lambda var: nlp_order.get(nlp_methods.get(var, "text length"), 999)
    )
    heatmap_data = heatmap_data.loc[sorted_features]
    # --- End Sorting ---

    # Assign colors based on NLP method (default to black if unknown)
    feature_colors = [color_map.get(nlp_methods.get(var, "text length"), "black") for var in heatmap_data.index]

    # Create color scale for positive and negative values
    color_scale = [
        [0.0, "rgb(75, 48, 163)"],     # Deep Blue (Strong Negative)
        [0.25, "rgb(188, 170, 220)"],   # Blue
        [0.5, "rgb(247, 247, 247)"],    # White (Neutral)
        [0.75, "rgb(215, 48, 39)"],     # Red
        [1.0, "rgb(165, 0, 38)"],       # Deep Red (Strong Positive)
    ]

    # Adjust heatmap height and width dynamically
    num_features = len(heatmap_data)
    num_participants = len(heatmap_data.columns)
    heatmap_height = min(max(600, num_features * 40), 1000)  # Min 600px, Max 1000px
    heatmap_width = max(800, num_participants * 75)  # Keep dynamic width for consistency

    # Determine max absolute value for symmetric color scaling
    vmax_value = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
    hover_text = [
        [f"Participant: {p}<br>Feature: {f}<br>Importance: {heatmap_data.at[f, p]:.4f}" for p in heatmap_data.columns]
        for f in heatmap_data.index
    ]

    # Create heatmap using Plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,  # Participants on X-axis
            y=heatmap_data.index,    # Features on Y-axis
            colorscale=color_scale,
            colorbar=dict(title="SHAP Value"),
            zmin=-vmax_value,
            zmax=vmax_value,
            text=hover_text,
            hoverinfo="text"
        ),
    )

    # Update layout with conditional x-axis ticks based on "All" selection
    if select_all:
        fig.update_layout(
            height=heatmap_height,
            width=heatmap_width,
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(heatmap_data.index))),
                ticktext=[
                    f'<span style="color:{color}">{label}</span>'
                    for label, color in zip(heatmap_data.index, feature_colors)
                ],
            ),
            xaxis=dict(
                tickmode="array",
                tickvals=[len(heatmap_data.columns) // 2],  # Single tick in the middle
                ticktext=["All Participants"],  # Single label
                tickfont=dict(size=12),
            )
        )
    else:
        fig.update_layout(
            height=heatmap_height,
            width=heatmap_width,
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(heatmap_data.index))),
                ticktext=[
                    f'<span style="color:{color}">{label}</span>'
                    for label, color in zip(heatmap_data.index, feature_colors)
                ],
            ),
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(heatmap_data.columns))),  # One tick per participant
                ticktext=heatmap_data.columns,  # Individual participant names
                tickangle=45,  # Rotate labels for readability
                tickfont=dict(size=10),
            )
        )

    # Show Plotly figure
    st.plotly_chart(fig, use_container_width=False)  # Use fixed width

    # Add a legend under the graph for the NLP method colors
    legend_items = []
    for method, color in color_map.items():
        legend_items.append(f'<span style="color: {color}; font-weight: bold;">■</span> {method.upper()}')
    legend_html = "    ".join(legend_items)
    st.markdown(
        f"""
        <div style="font-size: 12px; margin-top: 10px;">
            <strong>NLP Methods:</strong><br>
            {legend_html}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add the chatbot to the page
    app_with_chatbot.show_chatbot_ui()

    # WORKING
