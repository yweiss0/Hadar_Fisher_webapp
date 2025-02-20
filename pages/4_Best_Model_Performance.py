import streamlit as st
import pandas as pd
import os
import app_with_chatbot

# File Path
DATA_DIR = "data/files_tab_1_2/"

st.set_page_config(page_title="Best Model Performance per Participant", layout="wide")
st.title("🥇 Best Model Performance")

st.write("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)

col_space1, left_col, col_space2, right_col, col_space3 = st.columns([1, 2, 1, 6, 1])

with left_col:
    st.write("**Controls:**")

    # Dropdowns
    ml_model = st.selectbox("ML Model", ["Elastic Net (en)", "Random Forest (rf)"])
    outcome = st.selectbox("Outcome", ["Negative Affect", "Angry", "Nervous", "Sad"]).lower()
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
            df.columns = df.columns.str.lower()  # Convert all column names to lowercase

            # Drop first unnamed column if it exists
            if df.columns[0].startswith("unnamed"):
                df = df.iloc[:, 1:]

            # Handle the GPT file separately
            if file_name == "modelfit_gpt_all.csv":
                if "emotion_affect" not in df.columns:
                    st.error("Column 'emotion_affect' not found in GPT file.")
                    st.stop()
                
                # Convert emotion_affect to lowercase before filtering
                df["emotion_affect"] = df["emotion_affect"].str.lower()
                df = df[df["emotion_affect"] == outcome]

                # Rename "id" to "participant" to match other files
                if "id" in df.columns:
                    df = df.rename(columns={"id": "participant"})
                
                df["nomot_idiog"] = "N/A (LLM Ratings)"  # GPT has no nomot/idiog column
                nlp_approach = "GPT"
            else:
                nlp_approach = "comb" if "comb" in file_name else "LDA" if "LDA" in file_name else "LIWC"
                nomot_idiog = "Nomothetic" if "nomot" in file_name else "Idiographic"
                df["nomot_idiog"] = nomot_idiog

            df["nlp_approach"] = nlp_approach  # Track NLP Approach
            df["source_file"] = file_name  # Track source file

            # Standardize column names for consistency
            required_columns = ["participant", "r2", "rmse", "p_value", "nomot_idiog", "nlp_approach"]
            df = df[[col for col in required_columns if col in df.columns]]

            all_data.append(df)

    if not all_data:
        st.error("No relevant files found for the selected model and outcome.")
        st.stop()

    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Ensure required columns exist
    if "participant" not in combined_df.columns or "r2" not in combined_df.columns:
        st.error("Required columns ('participant' or 'r2') are missing in the data.")
        st.stop()

    # Drop rows where 'participant' or 'r2' is NaN
    combined_df = combined_df.dropna(subset=["participant", "r2"])

    # Convert 'participant' to string type to avoid indexing issues
    combined_df["participant"] = combined_df["participant"].astype(str)

    # Ensure no empty DataFrame before proceeding
    if combined_df.empty:
        st.warning("No valid data after removing NaN values.")
        st.stop()

    # Ensure 'r2' is numeric
    combined_df["r2"] = pd.to_numeric(combined_df["r2"], errors="coerce")

    # Drop any remaining NaN values in 'r2' to avoid idxmax() issues
    combined_df = combined_df.dropna(subset=["r2"])

    # If the dataframe is still empty, stop execution
    if combined_df.empty:
        st.warning("No valid data after filtering R² values.")
        st.stop()

    # Group by participant and find row with highest R²
    best_performance_df = combined_df.loc[combined_df.groupby("participant")["r2"].idxmax()].reset_index(drop=True)

    # Ensure column renaming is correctly applied
    rename_map = {
        "participant": "Participant",
        "r2": "R²",
        "rmse": "RMSE",
        "p_value": "P Value",
        "nomot_idiog": "Nomothetic/Idiographic",
        "nlp_approach": "NLP Approach"
    }
    best_performance_df = best_performance_df.rename(columns=rename_map)

    # Load the "Counts" file
    counts_file = f"comb_{ml_model_short}_{outcome}_nomot.csv"
    counts_path = os.path.join(DATA_DIR, counts_file)

    if os.path.exists(counts_path):
        counts_df = pd.read_csv(counts_path)
        counts_df.columns = counts_df.columns.str.lower()  # Standardize column names to lowercase

        # Find participant and count column names case-insensitively
        participant_col = next((col for col in counts_df.columns if "participant" in col), None)
        count_col = next((col for col in counts_df.columns if "count" in col), None)

        if participant_col and count_col:
            counts_df = counts_df[[participant_col, count_col]].rename(columns={participant_col: "Participant", count_col: "Counts"})
            counts_df["Participant"] = counts_df["Participant"].astype(str)

            # Merge counts into best_performance_df
            best_performance_df = best_performance_df.merge(counts_df, on="Participant", how="left")
        else:
            st.warning(f"Could not find required columns in {counts_file}. 'Counts' column will be missing.")

    else:
        st.warning(f"File {counts_file} not found. 'Counts' column will be missing.")

    # Ensure "Counts" column exists, fill missing values, and move to the last position
    best_performance_df["Counts"] = best_performance_df["Counts"].fillna(0).astype(int)

    # Sort table by "Participant" (A-Z) by default
    best_performance_df = best_performance_df.sort_values("Participant", ascending=True)

    # Ensure index starts from 1
    best_performance_df.index = range(1, len(best_performance_df) + 1)

    # Reorder columns
    best_performance_df = best_performance_df[["Participant", "Nomothetic/Idiographic", "NLP Approach", "R²", "RMSE", "P Value", "Counts"]]

# Display on the right
with right_col:
    st.write("")
    if not best_performance_df.empty:
        st.dataframe(best_performance_df, height=600, width=1200)
    else:
        st.warning("No data available for the selected criteria.")

st.write("</div>", unsafe_allow_html=True)

# Add the chatbot to the page
app_with_chatbot.show_chatbot_ui()
