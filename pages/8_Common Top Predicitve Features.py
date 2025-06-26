import streamlit as st
import pandas as pd
import os
import glob
import new_app_chatbot

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
    page_title="Common Top Predictive Features", page_icon="📊", layout="wide"
)

st.title("🔝 Common Top Predictive Features Across Participants")

# Center the entire layout
st.write(
    "<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True
)

# Create adjusted layout
col_space1, left_col, col_space2, right_col, col_space3 = st.columns([1, 2, 1, 6, 1])

with left_col:
    st.write("**Controls:**")

    # Dropdown Filters
    selected_emotion = st.selectbox(
        "Outcome", ["Negative Affect", "Angry", "Nervous", "Sad"]
    )
    selected_emotion = (
        "na"
        if selected_emotion.lower() == "negative affect"
        else selected_emotion.lower()
    )

    ml_model = st.selectbox("ML Model", ["Elastic Net (en)", "Random Forest (rf)"])
    ml_model_short = "en" if ml_model == "Elastic Net (en)" else "rf"

    # Checkbox for absolute values
    use_abs = st.checkbox("Use absolute values", value=True)

    # Threshold input for feature importance
    threshold = st.number_input(
        "Feature Importance Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.001,
    )

    # Top N features input
    top_n = st.number_input(
        "Top N Features per Participant", min_value=1, max_value=50, value=5, step=1
    )

    # Minimum participant count input
    min_participants = st.number_input(
        "Minimum Participants (for a feature to be considered common)",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
    )

with right_col:
    st.write("")  # Spacer

    # Load data with case-insensitive search
    if use_abs:
        file_pattern = (
            f"Featureimportance_{ml_model_short}_comb_{selected_emotion}_abs.csv"
        )
    else:
        file_pattern = f"Featureimportance_{ml_model_short}_comb_{selected_emotion}.csv"

    file_path = find_file_case_insensitive(DATA_DIR, file_pattern)

    if not file_path:
        st.error(f"File not found: {file_pattern}")
        st.stop()

    # Load the data
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    df.columns = df.columns.str.lower()

    # Check for required columns
    required_columns = ["participant", "variable", "importance"]
    if not all(col in df.columns for col in required_columns):
        st.error(f"Required columns {required_columns} not found in the data.")
        st.stop()

    # Clean the data
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    df = df.dropna(subset=["importance"])

    # Apply threshold filter
    if use_abs:
        df_filtered = df[df["importance"].abs() >= threshold]
    else:
        df_filtered = df[df["importance"] >= threshold]

    if df_filtered.empty:
        st.warning("No data meets the specified threshold.")
        st.stop()

    # Get top N features per participant
    top_features_per_participant = []
    for participant in df_filtered["participant"].unique():
        participant_data = df_filtered[df_filtered["participant"] == participant]

        if use_abs:
            # Sort by absolute value of importance
            top_features = participant_data.nlargest(top_n, "importance")
        else:
            # Sort by importance value
            top_features = participant_data.nlargest(top_n, "importance")

        top_features_per_participant.append(top_features)

    # Combine all top features
    if top_features_per_participant:
        all_top_features = pd.concat(top_features_per_participant, ignore_index=True)
    else:
        st.warning("No top features found.")
        st.stop()

    # Count how many participants use each feature
    feature_counts = all_top_features["variable"].value_counts()

    # Filter features that appear in at least min_participants
    common_features = feature_counts[feature_counts >= min_participants]

    if common_features.empty:
        st.warning(f"No features appear in at least {min_participants} participants.")
        st.stop()

    # Create a detailed analysis
    common_features_df = pd.DataFrame(
        {
            "Feature": common_features.index,
            "Participant_Count": common_features.values,
            "Percentage_of_Participants": (
                common_features.values / len(df_filtered["participant"].unique()) * 100
            ).round(2),
        }
    )

    # Calculate average importance for each common feature
    avg_importance = []
    std_importance = []
    for feature in common_features_df["Feature"]:
        feature_data = all_top_features[all_top_features["variable"] == feature][
            "importance"
        ]
        avg_importance.append(feature_data.mean())
        std_importance.append(feature_data.std())

    common_features_df["Average_Importance"] = avg_importance
    common_features_df["Std_Importance"] = std_importance

    # Sort by participant count (descending)
    common_features_df = common_features_df.sort_values(
        "Participant_Count", ascending=False
    )

    st.write("### Common Top Predictive Features")
    st.write(f"**Total Participants:** {len(df_filtered['participant'].unique())}")
    st.write(f"**Features shown:** Top {top_n} per participant")
    st.write(f"**Minimum participants:** {min_participants}")
    st.write(f"**Threshold:** {threshold}")

    # Display the results table
    st.dataframe(common_features_df.round(4), height=600, use_container_width=True)

    # Create a bar chart
    import plotly.express as px

    fig = px.bar(
        common_features_df.head(20),  # Show top 20 most common features
        x="Participant_Count",
        y="Feature",
        title="Top 20 Most Common Predictive Features Across Participants",
        labels={"Participant_Count": "Number of Participants", "Feature": "Features"},
        orientation="h",
    )

    fig.update_layout(
        height=600, yaxis={"categoryorder": "total ascending"}, template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    st.write("### Summary Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Common Features", len(common_features_df))

    with col2:
        st.metric(
            "Most Common Feature Count", common_features_df["Participant_Count"].max()
        )

    with col3:
        st.metric(
            "Average Participants per Feature",
            round(common_features_df["Participant_Count"].mean(), 1),
        )

st.write("</div>", unsafe_allow_html=True)

# Add the chatbot to the page
new_app_chatbot.show_chatbot_ui(page_name="Common Top Predictive Features")
