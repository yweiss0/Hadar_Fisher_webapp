import streamlit as st
import pandas as pd
import os
import plotly.express as px
import app_with_chatbot

# File Path
DATA_DIR = "data/files_tab_4/"

st.set_page_config(page_title="Feature Importance Visualization", layout="wide")
st.title("💡 Common Top Predictive Features")

# Layout: Sidebar (1/4 width) for controls, Main area (3.5/4 width) for graphs
left_col, spacer_col, right_col = st.columns([1, 0.5, 3])

with left_col:
    st.write("### Controls")
    
    # Model selection dropdown
    ml_model = st.selectbox("Model", ["Elastic Net (EN)", "Random Forest (RF)"])
    ml_model_short = "en" if ml_model == "Elastic Net (EN)" else "rf"
    
    # Sliders for user input
    num_features = st.slider("Number of Features per Participant", 5, 20, 10)
    num_variables = st.slider("Number of Variables in Figure", 5, 20, 10)
    shap_threshold = st.slider("SHAP Value Threshold", 0.001, 0.05, 0.01, step=0.001)
    
    # Load and filter data
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and "en" in f]
    
    # List of predefined emotions
    emotions = ["na", "sad", "angry", "nervous"]

# Create plots for each emotion
figs = []
for selected_emotion in emotions:
    file_name = f"Featureimportance_{ml_model_short}_comb_{selected_emotion}.csv"
    file_path = os.path.join(DATA_DIR, file_name)
    
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_name}")
        continue
    
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    df.columns = df.columns.str.lower()
    
    if "participant" not in df.columns or "variable" not in df.columns or "importance" not in df.columns:
        st.error("Missing required columns in the CSV file")
        continue
    
    df = df[df["importance"].abs() > shap_threshold]
    
    # Compute percentage of participants for each variable
    df_count = df.groupby(["variable", df["importance"] > 0])["participant"].nunique().reset_index()
    df_count.columns = ["variable", "positive", "count"]
    df_count["shap_sign"] = df_count["positive"].replace({True: "Positive", False: "Negative"})
    
    # Filter for top variables
    top_variables = df_count.groupby("variable")["count"].sum().nlargest(num_variables).index
    df_filtered = df_count[df_count["variable"].isin(top_variables)]
    if selected_emotion == "na":
        selected_emotion = "Negative Affect"

    # Plot with Plotly
    fig = px.bar(
        df_filtered, 
        x="count", 
        y="variable", 
        color="shap_sign", 
        orientation="h", 
        title=f"Top {num_variables} Variables for {selected_emotion.capitalize()} (Model: {ml_model})",
        labels={"count": "Percent of Participants", "variable": "Feature", "shap_sign": "SHAP Sign"},
        barmode="stack",
        color_discrete_map={"Positive": "rgb(0,182,185)", "Negative": "rgb(255,79,82)"}
    )
    
    figs.append(fig)

# Display figures in 2x2 grid
with right_col:
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    if len(figs) > 0:
        row1_col1.plotly_chart(figs[0], use_container_width=True)
    if len(figs) > 1:
        row1_col2.plotly_chart(figs[1], use_container_width=True)
    if len(figs) > 2:
        row2_col1.plotly_chart(figs[2], use_container_width=True)
    if len(figs) > 3:
        row2_col2.plotly_chart(figs[3], use_container_width=True)


# Add the chatbot to the page
app_with_chatbot.show_chatbot_ui()