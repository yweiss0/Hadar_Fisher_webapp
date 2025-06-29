import streamlit as st
import pandas as pd
import os
import glob
import new_app_chatbot

# File Path
DATA_DIR = "data/files_tab_1_2/"
# DATA_DIR = "C:/Projects/personal_projects2/Hadar_Fisher_Website/data/files_tab_1_2/"

st.set_page_config(page_title="Data Table View", page_icon="📊", layout="wide")

st.title("📋 Model performance per participant")


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


# Center the entire layout
st.write(
    "<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True
)

# Create adjusted layout
col_space1, left_col, col_space2, right_col, col_space3 = st.columns([1, 2, 1, 6, 1])

with left_col:
    st.write("**File Selection:**")

    # Dropdowns
    nlp_approach = st.selectbox("NLP Approach", ["COMBINED", "LDA", "GPT", "LIWC"])
    nomothetic_idiographic = st.selectbox(
        "Idiographic/Nomothetic", ["Idiographic", "Nomothetic"]
    )
    ml_model = st.selectbox("ML Model", ["Elastic Net (en)", "Random Forest (rf)"])

    nom_idio_value = "nomot" if nomothetic_idiographic == "Nomothetic" else "idiog"
    ml_model_short = "en" if ml_model == "Elastic Net (en)" else "rf"
    nlp_approach_value = "comb" if nlp_approach == "COMBINED" else nlp_approach

    # -- GPT CASE: If user selects GPT, load modelfit_gpt_all.csv and show unique outcomes --
    if nlp_approach == "GPT":
        # Load GPT file
        gpt_file_path = os.path.join(DATA_DIR, "modelfit_gpt_all.csv")
        if not os.path.exists(gpt_file_path):
            st.error("File not found: modelfit_gpt_all.csv")
            st.stop()

        df_gpt_all = pd.read_csv(gpt_file_path)
        df_gpt_all.columns = df_gpt_all.columns.str.lower()

        if "emotion_affect" not in df_gpt_all.columns:
            st.error("Column 'emotion_affect' not found in GPT file.")
            st.stop()

        unique_emotions = df_gpt_all["emotion_affect"].unique().tolist()
        outcome = st.selectbox("Outcome", unique_emotions)

        # Filter GPT data by the selected outcome + nom_idio_value if possible
        if "nom_idio" not in df_gpt_all.columns:
            st.warning("Nomothetic/Idiographic filter not working for GPT NLP.")
            df = df_gpt_all[df_gpt_all["emotion_affect"] == outcome]
        else:
            df = df_gpt_all[
                (df_gpt_all["emotion_affect"] == outcome)
                & (df_gpt_all["nom_idio"] == nom_idio_value)
            ]

        if df.empty:
            st.warning(
                "No data for the selected outcome and Nomothetic/Idiographic combination."
            )
            st.stop()

        # st.success("Loaded data: **modelfit_gpt_all.csv** (GPT mode)")
        # Clean up columns
        if "unnamed: 0" in df.columns:
            df = df.drop(columns=["unnamed: 0"])
        required_columns = ["id", "participant", "r", "r2", "rmse", "p_value"]
        df = df[[col for col in required_columns if col in df.columns]]

    else:
        # -- Non-GPT case (original logic) --
        outcome = st.selectbox(
            "Outcome", ["Negative Affect", "Angry", "Nervous", "Sad"]
        ).lower()
        outcome = "na" if outcome == "negative affect" else outcome

        # Try multiple case variations for the NLP approach
        possible_approaches = [
            nlp_approach_value.lower(),  # lowercase
            (
                nlp_approach if nlp_approach != "COMBINED" else "COMB"
            ),  # original/uppercase
            (
                nlp_approach.capitalize() if nlp_approach != "COMBINED" else "Comb"
            ),  # capitalized
        ]

        file_path = None
        file_name = None

        # Try each case variation until we find the file
        for approach_variant in possible_approaches:
            test_file_name = (
                f"{approach_variant}_{ml_model_short}_{outcome}_{nom_idio_value}.csv"
            )
            test_file_path = find_file_case_insensitive(DATA_DIR, test_file_name)

            if test_file_path:
                file_path = test_file_path
                file_name = os.path.basename(test_file_path)
                break

        # If no variation worked, use the original for error message
        if not file_path:
            file_name = (
                f"{nlp_approach_value}_{ml_model_short}_{outcome}_{nom_idio_value}.csv"
            )

        df = None
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()

            if "unnamed: 0" in df.columns:
                df = df.drop(columns=["unnamed: 0"])

            required_columns = ["id", "participant", "r", "r2", "rmse", "p_value"]
            df = df[[col for col in required_columns if col in df.columns]]

            # st.success(f"Loaded data: **{file_name}**")
        else:
            st.error(f"File not found: {file_name}")
            st.stop()


# Display on the right
with right_col:
    st.write("")  # Spacer
    if nlp_approach == "GPT":
        st.write(f"### Data Table: modelfit_gpt_all.csv (GPT mode)")
    else:
        st.write(f"### Data Table:")

    st.dataframe(df, height=600, width=1200)
    st.write("")  # Spacer

st.write("</div>", unsafe_allow_html=True)


# Add the chatbot to the page
new_app_chatbot.show_chatbot_ui(page_name="Data Table View")
# WORKING
