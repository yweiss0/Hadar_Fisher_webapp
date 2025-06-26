import streamlit as st
import pandas as pd
import os
import glob
import plotly.graph_objects as go
import new_app_chatbot
import numpy as np

# File Path
DATA_DIR = "data/files_tab_4/"


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
    page_title="Feature Importance Heatmap", page_icon="📊", layout="wide"
)

st.title("📉 feature importance Heatmap (SHAP Values)")

# Layout: Right side (1/4) for controls, Left side (3/4) for heatmap
col_space1, right_col, col_space2, left_col, col_space3 = st.columns([1, 2, 1, 6, 1])

with right_col:
    st.write("### Controls")

    # Dropdowns
    outcome = st.selectbox("Outcome", ["Negative Affect", "Angry", "Nervous", "Sad"])
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
                st.session_state.symmetric_val = (-0.0005, 0.0005)
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
        # st.markdown(
        #     '<div style="font-size: 12px; margin-top: 5px; padding: 8px; border-radius: 5px; background-color: #f0f2f6;">All values are in ABS.</div>',
        #     unsafe_allow_html=True
        # )

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
    st.write("")
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
    if "symmetric_val" not in st.session_state:
        st.session_state.symmetric_val = (-0.005, 0.005)  # Default fallback

    if "prev_val" not in st.session_state:
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
        unsafe_allow_html=True,
    )

    # Use the positive threshold from the symmetric range as the filtering threshold
    min_importance_threshold = slider_value[1]
    # --- End Dynamic Component ---

with left_col:
    # Filter data based on selected participants
    df_filtered = df[df["participant"].isin(selected_participants)]

    # Aggregate duplicate entries before pivoting
    df_filtered = df_filtered.groupby(["participant", "variable"], as_index=False)[
        "importance"
    ].mean()

    # Apply the importance threshold to filter out values between -threshold and threshold
    df_filtered = df_filtered[
        (df_filtered["importance"] <= -min_importance_threshold)
        | (df_filtered["importance"] >= min_importance_threshold)
    ]

    # Pivot for heatmap (Swapped axes: participants → x, features → y)
    heatmap_data = df_filtered.pivot(
        index="variable", columns="participant", values="importance"
    ).fillna(0)

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
        "text feature": "brown",
    }

    # --- Group & Sort y-axis by NLP Method ---
    # Define the desired order: text length, LIWC, time, gpt, vader, lda.
    nlp_order = {
        "text length": 1,
        "liwc": 2,
        "time": 3,
        "gpt": 4,
        "vader": 5,
        "lda": 6,
        "text feature": 7,
    }
    sorted_features = sorted(
        heatmap_data.index,
        key=lambda var: nlp_order.get(nlp_methods.get(var, "text length"), 999),
    )
    heatmap_data = heatmap_data.loc[sorted_features]
    # --- End Sorting ---

    # Assign colors based on NLP method (default to black if unknown)
    feature_colors = [
        color_map.get(nlp_methods.get(var, "text length"), "black")
        for var in heatmap_data.index
    ]

    # Create color scale for positive and negative values
    color_scale = [
        [0.0, "rgb(75, 48, 163)"],  # Deep Blue (Strong Negative)
        [0.25, "rgb(188, 170, 220)"],  # Blue
        [0.5, "rgb(247, 247, 247)"],  # White (Neutral)
        [0.75, "rgb(215, 48, 39)"],  # Red
        [1.0, "rgb(165, 0, 38)"],  # Deep Red (Strong Positive)
    ]

    # Adjust heatmap height and width dynamically
    num_features = len(heatmap_data)
    num_participants = len(heatmap_data.columns)
    heatmap_height = min(max(600, num_features * 40), 1000)  # Min 600px, Max 1000px
    heatmap_width = max(
        800, num_participants * 75
    )  # Keep dynamic width for consistency

    # Determine max absolute value for symmetric color scaling
    vmax_value = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
    hover_text = [
        [
            f"Participant: {p}<br>Feature: {f}<br>Importance: {heatmap_data.at[f, p]:.4f}"
            for p in heatmap_data.columns
        ]
        for f in heatmap_data.index
    ]

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
            hoverinfo="text",
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
            ),
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
                tickvals=list(
                    range(len(heatmap_data.columns))
                ),  # One tick per participant
                ticktext=heatmap_data.columns,  # Individual participant names
                tickangle=45,  # Rotate labels for readability
                tickfont=dict(size=10),
            ),
        )

    # Show Plotly figure
    st.plotly_chart(fig, use_container_width=False)  # Use fixed width

    # Add a legend under the graph for the NLP method colors
    legend_items = []
    for method, color in color_map.items():
        legend_items.append(
            f'<span style="color: {color}; font-weight: bold;">■</span> {method.upper()}'
        )
    legend_html = "    ".join(legend_items)
    st.markdown(
        f"""
        <div style="font-size: 12px; margin-top: 10px;">
            <strong>NLP Methods:</strong><br>
            {legend_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add the chatbot to the page
    new_app_chatbot.show_chatbot_ui(
        page_name="feature importance Heatmap (SHAP Values)"
    )

    # WORKING
