import streamlit as st
import pandas as pd
import os
import app_with_chatbot

# File Path
DATA_DIR = "data/files_tab_1_2/"

st.set_page_config(page_title="Best Model Performance per Participant", layout="wide")

st.title("🥇 Best Model Performance")

# Center the entire layout
st.write("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)

# Create adjusted layout
col_space1, left_col, col_space2, right_col, col_space3 = st.columns([1, 2, 1, 6, 1])

with left_col:
    st.write("**Controls:**")

    # Dropdowns
    ml_model = st.selectbox("ML Model", ["Elastic Net (en)", "Random Forest (rf)"])
    outcome = st.selectbox("Outcome", ["Angry", "Nervous", "Sad", "Negative Affect"]).lower()
    outcome = "na" if outcome == "negative affect" else outcome
    ml_model_short = "en" if ml_model == "Elastic Net (en)" else "rf"

    # Define file names
    file_patterns = [
        f"comb_{ml_model_short}_{outcome}_idiog.csv",
        f"comb_{ml_model_short}_{outcome}_nomot.csv",
        f"LDA_{ml_model_short}_{outcome}_idiog.csv",
        f"LDA_{ml_model_short}_{outcome}_nomot.csv",
        f"LIWC_{ml_model_short}_{outcome}_idiog.csv",
        f"LIWC_{ml_model_short}_{outcome}_nomot.csv",
        "modelfit_gpt_all.csv"
    ]

    # Load all relevant data
    all_data = []
    
    for file_name in file_patterns:
        file_path = os.path.join(DATA_DIR, file_name)
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()

            # Drop first unnamed column if it exists
            if df.columns[0].startswith("unnamed"):
                df = df.iloc[:, 1:]

            # Standardize columns and track source
            if file_name == "modelfit_gpt_all.csv":
                if "emotion_affect" not in df.columns:
                    st.error("Column 'emotion_affect' not found in GPT file.")
                    st.stop()
                
                df = df[df["emotion_affect"] == outcome]
                df["nomot_idiog"] = ""  # GPT has no nomot/idiog column
                nlp_approach = "GPT"
            else:
                nlp_approach = "comb" if "comb" in file_name else "LDA" if "LDA" in file_name else "LIWC"
                nomot_idiog = "Nomothetic" if "nomot" in file_name else "Idiographic"
                df["nomot_idiog"] = nomot_idiog

            df["nlp_approach"] = nlp_approach
            df["source_file"] = file_name  # Track source file

            required_columns = ["id", "participant", "r2", "rmse", "p_value", "nomot_idiog", "nlp_approach"]
            df = df[[col for col in required_columns if col in df.columns]]

            all_data.append(df)

    if not all_data:
        st.error("No relevant files found for the selected model and outcome.")
        st.stop()

    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Find the row with the highest R2 for each participant
    best_performance_df = combined_df.loc[combined_df.groupby("participant")["r2"].idxmax()].reset_index(drop=True)

    # Remove 'id' column if all values are None
    if "id" in best_performance_df.columns and best_performance_df["id"].isna().all():
        best_performance_df = best_performance_df.drop(columns=["id"])

    # Make index start from 1 instead of 0
    best_performance_df.index = best_performance_df.index + 1

    # Rename columns
    column_renames = {
        "participant": "Participant",
        "nomot_idiog": "Nomothetic/Idiographic",
        "r2": "R²",
        "rmse": "RMSE",
        "p_value": "P Value"
    }
    best_performance_df = best_performance_df.rename(columns=column_renames)

    # Reorder columns
    ordered_columns = ["Participant", "Nomothetic/Idiographic", "R²", "RMSE", "P Value"]
    best_performance_df = best_performance_df[ordered_columns]

    st.success("Loaded and processed all relevant data.")

# Display on the right
with right_col:
    st.write("")  # Spacer
    if not best_performance_df.empty:
        st.dataframe(best_performance_df, height=600, width=1200)
    else:
        st.warning("No data available for the selected criteria.")

    st.write("")  # Spacer

st.write("</div>", unsafe_allow_html=True)

# Add the chatbot to the page
app_with_chatbot.show_chatbot_ui()
