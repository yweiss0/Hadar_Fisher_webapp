import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import app_with_chatbot

# File Paths
DATA_DIR_1 = "data/files_tab_1_2/"
DATA_DIR_4 = "data/files_tab_4/"

st.set_page_config(page_title="Feature Importance Analysis", layout="wide")
st.title("🔬 Feature Importance Analysis")

col_space1, left_col, col_space2, right_col, col_space3 = st.columns([1, 2, 1, 6, 1])

with left_col:
    st.write("**Controls:**")
    
    # Dropdowns
    ml_model = st.selectbox("ML Model", ["Elastic Net (en)", "Random Forest (rf)"])
    outcome = st.selectbox("Outcome", ["Angry", "Nervous", "Sad", "Negative Affect"]).lower()
    outcome = "na" if outcome == "negative affect" else outcome
    ml_model_short = "en" if ml_model == "Elastic Net (en)" else "rf"

    # Load performance data
    perf_file = os.path.join(DATA_DIR_1, f"comb_{ml_model_short}_{outcome}_idiog.csv")
    if os.path.exists(perf_file):
        perf_df = pd.read_csv(perf_file)
        perf_df.columns = perf_df.columns.str.lower().str.strip()
    else:
        st.error(f"Performance file {perf_file} not found.")
        st.stop()

    # Ensure required columns exist
    if "participant" not in perf_df.columns or "r2" not in perf_df.columns:
        st.error("Required columns ('participant' or 'r2') are missing in the data.")
        st.stop()

    # Drop NaN values and convert to numeric
    perf_df = perf_df.dropna(subset=["participant", "r2"])
    perf_df["participant"] = perf_df["participant"].astype(str)
    perf_df["r2"] = pd.to_numeric(perf_df["r2"], errors="coerce")

    # **Slider for selecting percentage of participants (10% - 50% in 5% steps)**
    total_participants = len(perf_df)
    percentage_options = list(range(10, 51, 5))  # [10, 15, 20, ..., 50]

    # **Styled label with Tooltip**
    

    # **Slider with dynamic participant calculation**
    percentage = st.select_slider("Percentage of Participants in Each Group", options=percentage_options, value=25)

    # Calculate number of participants based on selected percentage
    num_of_participants = max(1, int((percentage / 100) * total_participants))  # Ensure at least 1 participant

    tooltip_html = f"""
    <style>
        .tooltip-container {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 14px;
            
        }}
        .tooltip {{
            position: relative;
            display: inline-block;
            cursor: pointer;
        }}
        .tooltip .tooltip-text {{
            visibility: hidden;
            width: 220px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -110px;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .tooltip-container:hover .tooltip-text {{
            visibility: visible;
            opacity: 1;
        }}
        .tooltip svg {{
            fill: #007BFF; /* Blue color */
            width: 18px;
            height: 18px;
        }}
    </style>
    <div class="tooltip-container">
        <span>Selected Participants</span>
        <div class="tooltip">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24">
                <g fill="none">
                    <path d="m12.593 23.258l-.011.002l-.071.035l-.02.004l-.014-.004l-.071-.035q-.016-.005-.024.005l-.004.01l-.017.428l.005.02l.01.013l.104.074l.015.004l.012-.004l.104-.074l.012-.016l.004-.017l-.017-.427q-.004-.016-.017-.018m.265-.113l-.013.002l-.185.093l-.01.01l-.003.011l.018.43l.005.012l.008.007l.201.093q.019.005.029-.008l.004-.014l-.034-.614q-.005-.018-.02-.022m-.715.002a.02.02 0 0 0-.027.006l-.006.014l-.034.614q.001.018.017.024l.015-.002l.201-.093l.01-.008l.004-.011l.017-.43l-.003-.012l-.01-.01z"/>
                    <path fill="currentColor" d="M12 2c5.523 0 10 4.477 10 10s-4.477 10-10 10S2 17.523 2 12S6.477 2 12 2m0 2a8 8 0 1 0 0 16a8 8 0 0 0 0-16m-.01 6c.558 0 1.01.452 1.01 1.01v5.124A1 1 0 0 1 12.5 18h-.49A1.01 1.01 0 0 1 11 16.99V12a1 1 0 1 1 0-2zM12 7a1 1 0 1 1 0 2a1 1 0 0 1 0-2"/>
                </g>
            </svg>
            <span class="tooltip-text">🔢 Selected: {percentage}% → {num_of_participants} participants per group</span>
        </div>
    </div>
    """
    
    # Render styled label above the slider
    st.markdown(tooltip_html, unsafe_allow_html=True)

    # **Slider dynamically adjusts based on `num_of_participants`**
    st.write("") # Add space for better alignment
    min_part_var = st.slider(
        "Minimum Number of Participants per Variable", 
        0, num_of_participants, min(2, num_of_participants), step=1
    )

    # Sort participants based on R²
    highest_r2 = perf_df.nlargest(num_of_participants, "r2")["participant"].tolist()
    lowest_r2 = perf_df.nsmallest(num_of_participants, "r2")["participant"].tolist()

    # Load feature importance data
    feature_file = os.path.join(DATA_DIR_4, f"Featureimportance_{ml_model_short}_comb_{outcome}.csv")
    if os.path.exists(feature_file):
        feature_df = pd.read_csv(feature_file)
        feature_df.columns = feature_df.columns.str.lower().str.strip()
    else:
        st.error(f"Feature importance file {feature_file} not found.")
        st.stop()

    # Ensure required columns exist
    if "participant" not in feature_df.columns or "variable" not in feature_df.columns:
        st.error("Required columns ('participant' or 'variable') are missing in the data.")
        st.stop()

    # Keep only selected participants
    feature_df = feature_df[feature_df["participant"].astype(str).isin(highest_r2 + lowest_r2)]

    # Count occurrences of each variable
    var_counts = feature_df["variable"].value_counts()
    valid_vars = var_counts[var_counts >= min_part_var].index.tolist()
    
    # Filter feature dataframe
    feature_df = feature_df[feature_df["variable"].isin(valid_vars)]

    # Convert numeric columns to absolute values
    numeric_cols = feature_df.select_dtypes(include=['number']).columns
    feature_df[numeric_cols] = feature_df[numeric_cols].abs()

    # Compute absolute mean for each group
    high_r2_df = feature_df[feature_df["participant"].astype(str).isin(highest_r2)].groupby("variable")[numeric_cols].mean().mean(axis=1)
    low_r2_df = feature_df[feature_df["participant"].astype(str).isin(lowest_r2)].groupby("variable")[numeric_cols].mean().mean(axis=1)

    # Merge and sort by absolute mean difference
    abs_mean_df = pd.DataFrame({
        "High R²": high_r2_df,
        "Low R²": low_r2_df
    }).fillna(0)

    abs_mean_df["Abs Mean Diff"] = abs(abs_mean_df["High R²"] - abs_mean_df["Low R²"])
    abs_mean_df = abs_mean_df.nlargest(20, "Abs Mean Diff")

with right_col:
    # st.write("**Top 20 Features by Absolute Mean Difference**")
    
    if not abs_mean_df.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=abs_mean_df.index,
            x=abs_mean_df["High R²"],
            orientation='h',
            name='High R²',
            marker=dict(color='red')
        ))
        
        fig.add_trace(go.Bar(
            y=abs_mean_df.index,
            x=abs_mean_df["Low R²"],
            orientation='h',
            name='Low R²',
            marker=dict(color='turquoise')
        ))
        
        fig.update_layout(
            title="Feature Importance (Top 20): High vs. Low R² Groups",
            xaxis_title="Absolute Mean Value",
            yaxis_title="Features",
            barmode='group',
            height=600,
            template='plotly_white',
            legend_title_text="R² Groups",
            legend=dict(
                orientation="v",  # **Horizontal legend**
                yanchor="top",    # **Align legend to the top of its box**           # **Move legend below the chart**
                xanchor="center",  # **Center legend horizontally**
                x=1             # **Position at center**
            )
        )
        
        st.plotly_chart(fig)
        # **Second Graph: Difference between Absolute Means**
        # st.write("**Difference in Absolute Mean Between High and Low R² Groups**")

        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            y=abs_mean_df.index,
            x=abs_mean_df["Abs Mean Diff"],
            orientation='h',
            name="Abs Mean Difference",
            marker=dict(color='gray')  # **Choose a distinct color for clarity**
        ))

        fig2.update_layout(
            title="Difference (Top 20): High vs. Low R² Groups",
            xaxis_title="Absolute Mean Difference",
            yaxis_title="Features",
            height=600,
            template='plotly_white'
        )

        st.plotly_chart(fig2)
    else:
        st.warning("No variables met the filtering criteria.") 
# Add the chatbot to the page
app_with_chatbot.show_chatbot_ui()