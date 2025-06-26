import streamlit as st
import pandas as pd
import os
import glob
import plotly.express as px
import new_app_chatbot

# File Path
DATA_DIR = "data/files_tab_1_2/"


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
    page_title="Best Model Performance per Participant", page_icon="📊", layout="wide"
)
st.title("🥇 Best Model Performance")

st.write(
    "<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True
)

col_space1, left_col, col_space2, right_col, col_space3 = st.columns([1, 2, 1, 6, 1])

with left_col:
    st.write("**Controls:**")

    # Checkbox for including both models
    include_both = st.checkbox("Include both Elastic Net and Random Forest")

    if not include_both:
        # Dropdowns (only shown when checkbox is not selected)
        ml_model = st.selectbox("ML Model", ["Elastic Net (en)", "Random Forest (rf)"])
        ml_model_short = "en" if ml_model == "Elastic Net (en)" else "rf"
        ml_model_list = [ml_model_short]
    else:
        ml_model_list = ["en", "rf"]

    outcome = st.selectbox(
        "Outcome", ["Negative Affect", "Angry", "Nervous", "Sad"]
    ).lower()
    outcome = "na" if outcome == "negative affect" else outcome

    # Load all relevant data
    all_data = []

    for ml_model_short in ml_model_list:
        # Define file patterns with case variations
        base_patterns = [
            f"comb_{ml_model_short}_{outcome}_idiog.csv",
            f"comb_{ml_model_short}_{outcome}_nomot.csv",
            f"LDA_{ml_model_short}_{outcome}_idiog.csv",
            f"LDA_{ml_model_short}_{outcome}_nomot.csv",
            f"LIWC_{ml_model_short}_{outcome}_idiog.csv",
            f"LIWC_{ml_model_short}_{outcome}_nomot.csv",
            "modelfit_gpt_all.csv",
        ]

        for pattern in base_patterns:
            file_path = find_file_case_insensitive(DATA_DIR, pattern)

            if file_path:
                file_name = os.path.basename(file_path)
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.lower()

                if df.columns[0].startswith("unnamed"):
                    df = df.iloc[:, 1:]

                if file_name.lower() == "modelfit_gpt_all.csv":
                    if "emotion_affect" not in df.columns:
                        st.error("Column 'emotion_affect' not found in GPT file.")
                        st.stop()

                    df["emotion_affect"] = df["emotion_affect"].str.lower()
                    df = df[df["emotion_affect"] == outcome]

                    if "id" in df.columns:
                        df = df.rename(columns={"id": "participant"})

                    df["nomot_idiog"] = "N/A (LLM Ratings)"
                    nlp_approach = "GPT"
                else:
                    # Extract NLP approach from actual filename (case-insensitive)
                    filename_lower = file_name.lower()
                    if "comb" in filename_lower:
                        nlp_approach = "comb"
                    elif "lda" in filename_lower:
                        nlp_approach = "LDA"
                    elif "liwc" in filename_lower:
                        nlp_approach = "LIWC"
                    else:
                        nlp_approach = "unknown"

                    nomot_idiog = (
                        "Nomothetic" if "nomot" in filename_lower else "Idiographic"
                    )
                    df["nomot_idiog"] = nomot_idiog

                df["nlp_approach"] = nlp_approach
                df["source_file"] = file_name

                # Add ML model column only if including both models
                if include_both and file_name.lower() != "modelfit_gpt_all.csv":
                    df["ml_model"] = (
                        "Elastic Net" if ml_model_short == "en" else "Random Forest"
                    )
                elif include_both and file_name.lower() == "modelfit_gpt_all.csv":
                    df["ml_model"] = "N/A (GPT)"

                required_columns = [
                    "participant",
                    "r2",
                    "rmse",
                    "p_value",
                    "nomot_idiog",
                    "nlp_approach",
                ]
                if include_both:
                    required_columns.append("ml_model")
                df = df[[col for col in required_columns if col in df.columns]]

                all_data.append(df)

    if not all_data:
        st.error("No relevant files found for the selected model and outcome.")
        st.stop()

    combined_df = pd.concat(all_data, ignore_index=True)

    if "participant" not in combined_df.columns or "r2" not in combined_df.columns:
        st.error("Required columns ('participant' or 'r2') are missing in the data.")
        st.stop()

    combined_df = combined_df.dropna(subset=["participant", "r2"])
    combined_df["participant"] = combined_df["participant"].astype(str)

    if combined_df.empty:
        st.warning("No valid data after removing NaN values.")
        st.stop()

    combined_df["r2"] = pd.to_numeric(combined_df["r2"], errors="coerce")
    combined_df = combined_df.dropna(subset=["r2"])

    if combined_df.empty:
        st.warning("No valid data after filtering R² values.")
        st.stop()

    best_performance_df = combined_df.loc[
        combined_df.groupby("participant")["r2"].idxmax()
    ].reset_index(drop=True)

    rename_map = {
        "participant": "Participant",
        "r2": "R²",
        "rmse": "RMSE",
        "p_value": "P Value",
        "nomot_idiog": "Nomothetic/Idiographic",
        "nlp_approach": "NLP Approach",
    }
    if include_both:
        rename_map["ml_model"] = "ML Model"
    best_performance_df = best_performance_df.rename(columns=rename_map)

    # Use case-insensitive search for counts file too
    counts_pattern = f"comb_{ml_model_list[0]}_{outcome}_nomot.csv"
    counts_path = find_file_case_insensitive(DATA_DIR, counts_pattern)

    if counts_path and os.path.exists(counts_path):
        counts_df = pd.read_csv(counts_path)
        counts_df.columns = counts_df.columns.str.lower()

        participant_col = next(
            (col for col in counts_df.columns if "participant" in col), None
        )
        count_col = next((col for col in counts_df.columns if "count" in col), None)

        if participant_col and count_col:
            counts_df = counts_df[[participant_col, count_col]].rename(
                columns={participant_col: "Participant", count_col: "Counts"}
            )
            counts_df["Participant"] = counts_df["Participant"].astype(str)
            best_performance_df = best_performance_df.merge(
                counts_df, on="Participant", how="left"
            )
        else:
            counts_filename = (
                os.path.basename(counts_path) if counts_path else counts_pattern
            )
            st.warning(
                f"Could not find required columns in {counts_filename}. 'Counts' column will be missing."
            )
    else:
        st.warning(f"File {counts_pattern} not found. 'Counts' column will be missing.")

    best_performance_df["Counts"] = best_performance_df["Counts"].fillna(0).astype(int)
    best_performance_df = best_performance_df.sort_values("Participant", ascending=True)
    best_performance_df.index = range(1, len(best_performance_df) + 1)

    # Reorder columns conditionally including ML Model only when checkbox is selected
    if include_both:
        best_performance_df = best_performance_df[
            [
                "Participant",
                "ML Model",
                "Nomothetic/Idiographic",
                "NLP Approach",
                "R²",
                "RMSE",
                "P Value",
                "Counts",
            ]
        ]
    else:
        best_performance_df = best_performance_df[
            [
                "Participant",
                "Nomothetic/Idiographic",
                "NLP Approach",
                "R²",
                "RMSE",
                "P Value",
                "Counts",
            ]
        ]

with right_col:
    st.write("")
    if not best_performance_df.empty:
        st.dataframe(best_performance_df, height=600, width=1200)

        # Create columns for pie charts (2 or 3 based on include_both)
        if include_both:
            col1, col2, col3 = st.columns([1, 1, 1])  # Equal column widths
        else:
            col1, col2 = st.columns([1, 1])  # Equal column widths

        # Define consistent chart dimensions
        chart_width = 300  # Adjust as needed
        chart_height = 300  # Ensure consistent height

        # Pie Chart 1: Nomothetic vs Idiographic with specific colors
        nomot_idiog_counts = best_performance_df[
            "Nomothetic/Idiographic"
        ].value_counts()
        fig1 = px.pie(
            values=nomot_idiog_counts.values,
            names=nomot_idiog_counts.index,
            title="Nomothetic vs Idiographic",
            color=nomot_idiog_counts.index,
            color_discrete_map={
                "Nomothetic": "lightblue",
                "Idiographic": "darkblue",
                "N/A (LLM Ratings)": "red",
            },
        )
        fig1.update_traces(domain=dict(x=[0.15, 0.85], y=[0.15, 0.85]))
        fig1.update_layout(
            margin=dict(t=40, b=0, l=0, r=0),
            title_x=0.3,
            title_font_size=14,
            legend=dict(orientation="v", y=-0.1, x=0.5, xanchor="center"),
        )
        with col1:
            st.plotly_chart(
                fig1, use_container_width=False, width=chart_width, height=chart_height
            )

        # Pie Chart 2: NLP Approach with specific colors
        nlp_approach_counts = best_performance_df["NLP Approach"].value_counts()
        nlp_approach_labels = [
            label.replace("comb", "All text features combined")
            for label in nlp_approach_counts.index
        ]
        fig2 = px.pie(
            values=nlp_approach_counts.values,
            names=nlp_approach_labels,
            title="NLP Approaches",
            color=nlp_approach_labels,
            color_discrete_map={
                "All text features combined": "darkblue",
                "LDA": "lightblue",
                "GPT": "red",
                "LIWC": "purple",
            },
        )
        fig2.update_traces(domain=dict(x=[0.15, 0.85], y=[0.15, 0.85]))
        fig2.update_layout(
            margin=dict(t=40, b=0, l=0, r=0),
            title_x=0.3,
            title_font_size=14,
            legend=dict(orientation="v", y=-0.1, x=0.5, xanchor="center"),
        )
        with col2:
            st.plotly_chart(
                fig2, use_container_width=False, width=chart_width, height=chart_height
            )

        # Pie Chart 3: ML Models with specific colors (only if include_both is selected)
        if include_both:
            ml_model_counts = best_performance_df["ML Model"].value_counts()
            fig3 = px.pie(
                values=ml_model_counts.values,
                names=ml_model_counts.index,
                title="ML Models",
                color=ml_model_counts.index,
                color_discrete_map={
                    "Elastic Net": "darkblue",
                    "Random Forest": "lightblue",
                    "N/A (GPT)": "red",
                },
            )
            fig3.update_traces(domain=dict(x=[0.15, 0.85], y=[0.15, 0.85]))
            fig3.update_layout(
                margin=dict(t=40, b=0, l=0, r=0),
                title_x=0.3,
                title_font_size=14,
                legend=dict(orientation="v", y=-0.1, x=0.5, xanchor="center"),
            )
            with col3:
                st.plotly_chart(
                    fig3,
                    use_container_width=False,
                    width=chart_width,
                    height=chart_height,
                )
    else:
        st.warning("No data available for the selected criteria.")

st.write("</div>", unsafe_allow_html=True)

new_app_chatbot.show_chatbot_ui(page_name="Best Model Performance per Participant")
