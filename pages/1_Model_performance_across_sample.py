import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import new_app_chatbot  # Assuming this module exists and is correct

# File Path
DATA_DIR = "data/files_tab_1_2/"

st.set_page_config(
    page_title="Model Performance Analysis", page_icon="📊", layout="wide"
)

st.title("📊 Model Performance Analysis")

# Create adjusted layout
col_space1, left_col, col_space2, right_col, col_space3 = st.columns(
    [0.5, 2, 0.5, 6, 0.5]
)

with left_col:
    st.write("**Model Selection:**")

    # Dropdowns
    nlp_approach = st.selectbox("NLP Approach", ["COMBINED", "GPT", "LDA", "LIWC"])
    if nlp_approach == "GPT":
        nomothetic_idiographic = st.selectbox(
            "Idiographic/Nomothetic", ["Nomothetic", "Idiographic"]
        )
    else:
        nomothetic_idiographic = st.selectbox(
            "Idiographic/Nomothetic", ["Idiographic", "Nomothetic"]
        )

    nom_idio_value = "nomot" if nomothetic_idiographic == "Nomothetic" else "idiog"

    # Initialize dataframes for final filtered data
    df_gpt_filtered_final = pd.DataFrame()
    df_en_final = pd.DataFrame()
    df_rf_final = pd.DataFrame()
    df_combined_final = pd.DataFrame()  # For plotting non-GPT

    # --- GPT CASE: Load GPT Model File ---
    if nlp_approach == "GPT":
        gpt_file_path = os.path.join(DATA_DIR, "modelfit_gpt_all.csv")
        if not os.path.exists(gpt_file_path):
            st.error(f"File not found: {gpt_file_path}")
            st.stop()

        try:
            df_gpt_all = pd.read_csv(gpt_file_path)
            df_gpt_all.columns = df_gpt_all.columns.str.lower()
        except Exception as e:
            st.error(f"Error loading GPT file: {e}")
            st.stop()

        required_cols_gpt = [
            "emotion_affect",
            "r2",
            "p_value",
        ]  # Add nom_idio if needed
        if "nom_idio" in df_gpt_all.columns:
            required_cols_gpt.append("nom_idio")
        else:
            st.warning(
                "Column 'nom_idio' not found in GPT file. Filtering by outcome only."
            )

        if not all(
            col in df_gpt_all.columns for col in required_cols_gpt if col != "nom_idio"
        ):  # Check core columns
            missing_cols = [
                col
                for col in required_cols_gpt
                if col not in df_gpt_all.columns and col != "nom_idio"
            ]
            st.error(f"Required columns missing in GPT file: {', '.join(missing_cols)}")
            st.stop()

        df_gpt_all["emotion_affect"] = (
            df_gpt_all["emotion_affect"].str.lower().fillna("unknown")
        )
        df_gpt_all["emotion_affect"] = df_gpt_all["emotion_affect"].replace(
            {"negative affect": "na"}
        )
        allowed = ["angry", "na", "nervous", "sad"]
        df_gpt_all = df_gpt_all[df_gpt_all["emotion_affect"].isin(allowed)]

        ordered_emotions = ["na", "angry", "nervous", "sad"]
        outcome = st.selectbox("Outcome", ordered_emotions)

        # --- Filter GPT Data ---
        if "nom_idio" in df_gpt_all.columns:
            df_gpt_filtered_initial = df_gpt_all[
                (df_gpt_all["emotion_affect"] == outcome)
                & (df_gpt_all["nom_idio"] == nom_idio_value)
            ].copy()
        else:
            df_gpt_filtered_initial = df_gpt_all[
                (df_gpt_all["emotion_affect"] == outcome)
            ].copy()

        # Filter out rows with NA in 'r2' for plotting, stats, AND NOW FOR N COUNT
        # Ensure 'r2' column exists before trying to drop NA based on it
        if "r2" in df_gpt_filtered_initial.columns:
            df_gpt_filtered_final = df_gpt_filtered_initial.dropna(subset=["r2"]).copy()
        else:
            st.error("Column 'r2' not found in the filtered GPT data. Cannot proceed.")
            df_gpt_filtered_final = pd.DataFrame()  # Ensure it's an empty DF

        if df_gpt_filtered_initial.empty:
            st.warning(
                f"No initial data found for Outcome '{outcome}' and '{nomothetic_idiographic}' in the GPT file."
            )
        elif df_gpt_filtered_final.empty and not df_gpt_filtered_initial.empty:
            st.warning(
                f"Data found for Outcome '{outcome}' and '{nomothetic_idiographic}', but all entries have missing R² values."
            )

    else:
        # --- Non-GPT Case ---
        outcome = st.selectbox(
            "Outcome", ["Negative Affect", "Angry", "Nervous", "Sad"]
        ).lower()
        outcome = "na" if outcome == "negative affect" else outcome
        nlp_approach_value = (
            "comb" if nlp_approach == "COMBINED" else nlp_approach.lower()
        )

        file_name_en = f"{nlp_approach_value}_en_{outcome}_{nom_idio_value}.csv"
        file_name_rf = f"{nlp_approach_value}_rf_{outcome}_{nom_idio_value}.csv"
        file_path_en = os.path.join(DATA_DIR, file_name_en)
        file_path_rf = os.path.join(DATA_DIR, file_name_rf)

        df_en, df_rf = None, None  # Initialize raw loaded dataframes

        # Load EN Data
        if os.path.exists(file_path_en):
            try:
                df_en = pd.read_csv(file_path_en)
                df_en.columns = df_en.columns.str.lower()
                if "r2" in df_en.columns:
                    df_en["ml model"] = "Elastic Net (en)"
                    # Filter NA R2 immediately for the final DF
                    df_en_final = df_en.dropna(subset=["r2"]).copy()
                else:
                    st.error(f"Column 'r2' not found in {file_name_en}.")
                    df_en = None  # Mark as not loaded successfully
            except Exception as e:
                st.error(f"Error loading file {file_name_en}: {e}")
                df_en = None
        else:
            st.warning(
                f"File not found: {file_name_en}. Elastic Net data will be missing."
            )

        # Load RF Data
        if os.path.exists(file_path_rf):
            try:
                df_rf = pd.read_csv(file_path_rf)
                df_rf.columns = df_rf.columns.str.lower()
                if "r2" in df_rf.columns:
                    df_rf["ml model"] = "Random Forest (rf)"
                    # Filter NA R2 immediately for the final DF
                    df_rf_final = df_rf.dropna(subset=["r2"]).copy()
                else:
                    st.error(f"Column 'r2' not found in {file_name_rf}.")
                    df_rf = None  # Mark as not loaded successfully
            except Exception as e:
                st.error(f"Error loading file {file_name_rf}: {e}")
                df_rf = None
        else:
            st.warning(
                f"File not found: {file_name_rf}. Random Forest data will be missing."
            )

        # Combine the *final* (R2-filtered) dataframes for plotting
        df_combined_final = pd.concat([df_en_final, df_rf_final], ignore_index=True)

        # Warnings based on final dataframes
        if df_en is None and df_rf is None:
            st.error("No data loaded for either Elastic Net or Random Forest model.")
        elif df_combined_final.empty and (df_en is not None or df_rf is not None):
            # Check if files were loaded but resulted in empty final DFs
            if (df_en is not None and df_en_final.empty) or (
                df_rf is not None and df_rf_final.empty
            ):
                st.warning(
                    "Data loaded, but all valid R² values are missing after filtering. Cannot plot or calculate stats."
                )


# Display Graph in Right Column
with right_col:
    st.write("")  # Spacer
    st.write("### Model Performance Graph")

    fig = None  # Initialize fig

    if nlp_approach == "GPT":
        # --- Plot GPT Model with Plotly ---
        if not df_gpt_filtered_final.empty:
            fig = px.violin(
                df_gpt_filtered_final,  # Use final R2-filtered data
                x="ml model",
                y="r2",
                box=True,
                points="all",
                color_discrete_sequence=["blue"],
            )
            fig.update_layout(
                title=f"R² Values for GPT Model ({outcome.upper()}, {nomothetic_idiographic})",
                xaxis_title="ML Model",
                yaxis_title="R²",
                template="plotly_white",
            )
        else:
            # Message already shown in left column if data was missing R2
            if not df_gpt_filtered_initial.empty and df_gpt_filtered_final.empty:
                pass  # Warning was already displayed
            elif df_gpt_filtered_initial.empty:
                st.write("No initial data was found to plot for GPT.")
            else:  # Should not happen if R2 check passed, but just in case
                st.write("No data with valid R² values to plot for GPT.")

    else:
        # --- Plot EN & RF Models with Plotly ---
        if not df_combined_final.empty:
            fig = px.violin(
                df_combined_final,  # Use final combined R2-filtered data
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
                title=f"R² Values for Machine Learning Models ({outcome.upper()}, {nomothetic_idiographic})",
                xaxis_title="ML Model",
                yaxis_title="R²",
                template="plotly_white",
            )
        else:
            # Message already potentially shown in left column
            if df_en is None and df_rf is None:
                st.write("No data files were loaded to plot.")
            elif (
                df_en is not None or df_rf is not None
            ):  # Files loaded but final DF is empty
                st.write("Data loaded, but no valid R² values available to plot.")

    # Display the plot if it was created
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # --- Summary Table ---
    st.write("### Model Summary")
    summary_data = []

    if nlp_approach == "GPT":
        # N is now the count from the R2-filtered dataframe
        n_gpt = df_gpt_filtered_final.shape[0]
        if n_gpt > 0:  # Calculate stats only if there's data *after* NA R2 filtering
            # Ensure p_value column exists before using it
            if "p_value" in df_gpt_filtered_final.columns:
                M_SD = f"{df_gpt_filtered_final['r2'].mean():.2f} ({df_gpt_filtered_final['r2'].std():.2f})"
                Range = f"({df_gpt_filtered_final['r2'].min():.2f}, {df_gpt_filtered_final['r2'].max():.2f})"
                P_below_0_05 = df_gpt_filtered_final[
                    df_gpt_filtered_final["p_value"] < 0.05
                ].shape[0]
            else:
                st.warning(
                    "Column 'p_value' missing in GPT data, cannot calculate P < 0.05 Count."
                )
                M_SD = f"{df_gpt_filtered_final['r2'].mean():.2f} ({df_gpt_filtered_final['r2'].std():.2f})"  # Can still calc R2 stats
                Range = f"({df_gpt_filtered_final['r2'].min():.2f}, {df_gpt_filtered_final['r2'].max():.2f})"
                P_below_0_05 = "N/A"  # Mark p-value count as not available
        else:  # Handle case where R2 filtering resulted in zero rows
            M_SD = "N/A"
            Range = "N/A"
            P_below_0_05 = 0
        # Use the new N count (based on valid R2)
        summary_data.append(["GPT", M_SD, n_gpt, Range, P_below_0_05])

    else:
        # Process Elastic Net
        n_en = df_en_final.shape[0]  # N based on final R2-filtered EN data
        if n_en > 0:
            # Ensure p_value column exists
            if "p_value" in df_en_final.columns:
                M_SD_en = (
                    f"{df_en_final['r2'].mean():.2f} ({df_en_final['r2'].std():.2f})"
                )
                Range_en = (
                    f"({df_en_final['r2'].min():.2f}, {df_en_final['r2'].max():.2f})"
                )
                P_below_0_05_en = df_en_final[df_en_final["p_value"] < 0.05].shape[0]
            else:
                st.warning(
                    f"Column 'p_value' missing in {file_name_en}, cannot calculate P < 0.05 Count for EN."
                )
                M_SD_en = (
                    f"{df_en_final['r2'].mean():.2f} ({df_en_final['r2'].std():.2f})"
                )
                Range_en = (
                    f"({df_en_final['r2'].min():.2f}, {df_en_final['r2'].max():.2f})"
                )
                P_below_0_05_en = "N/A"
        else:
            M_SD_en = "N/A"
            Range_en = "N/A"
            P_below_0_05_en = 0
        # Use the N count from df_en_final
        summary_data.append(
            ["Elastic Net (en)", M_SD_en, n_en, Range_en, P_below_0_05_en]
        )

        # Process Random Forest
        n_rf = df_rf_final.shape[0]  # N based on final R2-filtered RF data
        if n_rf > 0:
            # Ensure p_value column exists
            if "p_value" in df_rf_final.columns:
                M_SD_rf = (
                    f"{df_rf_final['r2'].mean():.2f} ({df_rf_final['r2'].std():.2f})"
                )
                Range_rf = (
                    f"({df_rf_final['r2'].min():.2f}, {df_rf_final['r2'].max():.2f})"
                )
                P_below_0_05_rf = df_rf_final[df_rf_final["p_value"] < 0.05].shape[0]
            else:
                st.warning(
                    f"Column 'p_value' missing in {file_name_rf}, cannot calculate P < 0.05 Count for RF."
                )
                M_SD_rf = (
                    f"{df_rf_final['r2'].mean():.2f} ({df_rf_final['r2'].std():.2f})"
                )
                Range_rf = (
                    f"({df_rf_final['r2'].min():.2f}, {df_rf_final['r2'].max():.2f})"
                )
                P_below_0_05_rf = "N/A"
        else:
            M_SD_rf = "N/A"
            Range_rf = "N/A"
            P_below_0_05_rf = 0
        # Use the N count from df_rf_final
        summary_data.append(
            ["Random Forest (rf)", M_SD_rf, n_rf, Range_rf, P_below_0_05_rf]
        )

    # Create and display the summary DataFrame
    if summary_data:  # Check if there's anything to display
        summary_df = pd.DataFrame(
            summary_data, columns=["ML Model", "M (SD)", "N", "Range", "P < 0.05 Count"]
        )
        st.table(summary_df)
    else:
        st.write("No summary data to display.")


# Add the chatbot to the page
try:
    new_app_chatbot.show_chatbot_ui(page_name="Model Performance Analysis")
except AttributeError:
    st.warning("Chatbot UI function not found or module 'new_app_chatbot' is missing.")
except Exception as e:
    st.error(f"Error displaying chatbot: {e}")
