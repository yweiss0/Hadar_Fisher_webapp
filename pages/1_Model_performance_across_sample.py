import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import new_app_chatbot

# File Path
DATA_DIR = "data/files_tab_1_2/"

st.set_page_config(
    page_title="Model Performance Analysis", page_icon="📊", layout="wide"
)

st.title("📊 Model Performance Analysis")

# Center the entire layout
st.write(
    "<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True
)

# Create adjusted layout (Left: 1/4, Right: 3/4)
col_space1, left_col, col_space2, right_col, col_space3 = st.columns([1, 2, 1, 6, 1])

with left_col:
    st.write("**Model Selection:**")

    # Dropdowns
    nlp_approach = st.selectbox("NLP Approach", ["COMBINED", "GPT", "LDA", "LIWC"])
    # For GPT, default to Nomothetic so that "na" (displayed as Negative Affect) has data
    if nlp_approach == "GPT":
        nomothetic_idiographic = st.selectbox(
            "Idiographic/Nomothetic", ["Nomothetic", "Idiographic"]
        )
    else:
        nomothetic_idiographic = st.selectbox(
            "Idiographic/Nomothetic", ["Idiographic", "Nomothetic"]
        )

    # Decide Nomothetic/Idiographic value
    nom_idio_value = "nomot" if nomothetic_idiographic == "Nomothetic" else "idiog"

    # -- GPT CASE: Load GPT Model File --
    if nlp_approach == "GPT":
        gpt_file_path = os.path.join(DATA_DIR, "modelfit_gpt_all.csv")
        if not os.path.exists(gpt_file_path):
            st.error("File not found: modelfit_gpt_all.csv")
            st.stop()

        df_gpt_all = pd.read_csv(gpt_file_path)
        df_gpt_all.columns = df_gpt_all.columns.str.lower()

        if "emotion_affect" not in df_gpt_all.columns:
            st.error("Column 'emotion_affect' not found in GPT file.")
            st.stop()

        # Convert emotion_affect to lowercase and standardize values
        df_gpt_all["emotion_affect"] = df_gpt_all["emotion_affect"].str.lower()
        df_gpt_all["emotion_affect"] = df_gpt_all["emotion_affect"].replace(
            {"negative affect": "na"}
        )
        # Filter out rows where emotion_affect is not one of the allowed outcomes
        allowed = ["angry", "na", "nervous", "sad"]
        df_gpt_all = df_gpt_all[df_gpt_all["emotion_affect"].isin(allowed)]

        # Define the fixed order with "na" first
        ordered_emotions = ["na", "angry", "nervous", "sad"]
        # Format function to display "na" as "Negative Affect" and others in title case
        # def format_emotion(x):
        #     return "Negative Affect" if x == "na" else x.title()

        outcome = st.selectbox("Outcome", ordered_emotions)
        # outcome = st.selectbox("Outcome", ordered_emotions, format_func=format_emotion)

        # Dynamically filter based on user selection
        if "nom_idio" in df_gpt_all.columns:
            df_gpt_filtered = df_gpt_all[
                (df_gpt_all["emotion_affect"] == outcome)
                & (df_gpt_all["nom_idio"] == nom_idio_value)
            ]
        else:
            df_gpt_filtered = df_gpt_all[df_gpt_all["emotion_affect"] == outcome]

        if df_gpt_filtered.empty:
            st.warning(
                "No data for the selected outcome and Nomothetic/Idiographic combination."
            )
            st.stop()

    else:
        outcome = st.selectbox(
            "Outcome", ["Negative Affect", "Angry", "Nervous", "Sad"]
        ).lower()
        outcome = "na" if outcome == "negative affect" else outcome
        nlp_approach_value = "comb" if nlp_approach == "COMBINED" else nlp_approach

        # File names for EN and RF models
        file_name_en = f"{nlp_approach_value}_en_{outcome}_{nom_idio_value}.csv"
        file_name_rf = f"{nlp_approach_value}_rf_{outcome}_{nom_idio_value}.csv"

        file_path_en = os.path.join(DATA_DIR, file_name_en)
        file_path_rf = os.path.join(DATA_DIR, file_name_rf)

        # Load CSVs dynamically based on selection
        df_en, df_rf = None, None

        if os.path.exists(file_path_en):
            df_en = pd.read_csv(file_path_en)
            df_en.columns = df_en.columns.str.lower()
        else:
            st.error(f"File not found: {file_name_en}")

        if os.path.exists(file_path_rf):
            df_rf = pd.read_csv(file_path_rf)
            df_rf.columns = df_rf.columns.str.lower()
        else:
            st.error(f"File not found: {file_name_rf}")

        # Stop if either model is missing
        if df_en is None or df_rf is None:
            st.stop()

        # Dynamically filter data to ensure changes take effect
        df_en["ml model"] = "Elastic Net (en)"
        df_rf["ml model"] = "Random Forest (rf)"
        df_combined = pd.concat([df_en, df_rf], ignore_index=True)
        df_combined = df_combined[df_combined["r2"].notna()]

# Display Graph in Right Column
with right_col:
    st.write("")  # Spacer
    st.write("### Model Performance Graph")

    if nlp_approach == "GPT":
        # --- Plot GPT Model with Plotly ---
        fig = px.violin(
            df_gpt_filtered.assign(**{"ml model": "GPT"}),
            x="ml model",
            y="r2",
            box=True,
            points="all",
            color_discrete_sequence=["blue"],  # Set color to blue
        )

        fig.update_layout(
            title="R² Values for GPT Model",
            xaxis_title="ML Model",
            yaxis_title="R²",
            template="plotly_white",
        )

    else:
        # --- Plot EN & RF Models with Plotly ---
        fig = px.violin(
            df_combined,
            x="ml model",
            y="r2",
            box=True,
            points="all",
            color="ml model",
            color_discrete_map={
                "Elastic Net (en)": "red",
                "Random Forest (rf)": "blue",
            },
        )

        fig.update_layout(
            title="R² Values for Machine Learning Models",
            xaxis_title="ML Model",
            yaxis_title="R²",
            template="plotly_white",
        )

    st.plotly_chart(fig)

    # --- Summary Table ---
    summary_data = []
    if nlp_approach == "GPT":
        M_SD = f"{df_gpt_filtered['r2'].mean():.2f} ({df_gpt_filtered['r2'].std():.2f})"
        N = df_gpt_filtered.shape[0]
        Range = (
            f"({df_gpt_filtered['r2'].min():.2f}, {df_gpt_filtered['r2'].max():.2f})"
        )
        P_below_0_05 = df_gpt_filtered[df_gpt_filtered["p_value"] < 0.05].shape[0]
        summary_data.append(["GPT", M_SD, N, Range, P_below_0_05])
    else:
        for model, df_temp in zip(
            ["Elastic Net (en)", "Random Forest (rf)"], [df_en, df_rf]
        ):
            M_SD = f"{df_temp['r2'].mean():.2f} ({df_temp['r2'].std():.2f})"
            N = df_temp.shape[0]
            Range = f"({df_temp['r2'].min():.2f}, {df_temp['r2'].max():.2f})"
            P_below_0_05 = df_temp[df_temp["p_value"] < 0.05].shape[0]
            summary_data.append([model, M_SD, N, Range, P_below_0_05])

    summary_df = pd.DataFrame(
        summary_data, columns=["ML Model", "M (SD)", "N", "Range", "P < 0.05 Count"]
    )

    st.write("### Model Summary")
    st.table(summary_df)

# Add the chatbot to the page
new_app_chatbot.show_chatbot_ui(page_name="Model Performance Analysis")
