import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import app_with_chatbot




# File Path
DATA_DIR = "data/files_tab_3/"


st.set_page_config(page_title="True vs Predicted", layout="wide")
# **Page Title with Refresh Button**
col1, col2 = st.columns([8, 1])  # Title takes 8/9th, Refresh button takes 1/9th
with col1:
    st.title("📈 True vs Predicted NA Levels")
with col2:
    if st.button("🔄 Clear All", key="hard_refresh"):
        # **Clear all session state variables before refreshing**
        for key in list(st.session_state.keys()):
            del st.session_state[key]  # Completely reset session state
        st.rerun()  # **Force a full reset of the app**


# st.title("📈 True vs Predicted NA Levels")

# Store the number of graphs dynamically
if "graph_count" not in st.session_state:
    st.session_state.graph_count = 1  # Start with 1 graph

# Store used file names to track duplicates
if "used_files" not in st.session_state:
    st.session_state.used_files = {}

# Center the entire layout
st.write("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)

# Function to load and plot data
def load_and_plot_graph(graph_index):
    st.subheader(f"Graph {graph_index} / {st.session_state.graph_count}")

    # **Updated Layout: Left 1/4 (Dropdowns) | Right 3/4 (Graph)**
    col_space1, left_col, col_space2, right_col, col_space3 = st.columns([1, 2, 1, 6, 1])  # 1/4 - 3/4 layout

    with left_col:
        st.write("**Graph Settings:**")

        # Dropdown Filters (Stacked in Left Column)
        outcome = st.selectbox(f"Outcome ({graph_index})", ["Negative Affect", "Angry", "Nervous", "Sad"], key=f"outcome_{graph_index}").lower()
        outcome = "na" if outcome == "negative affect" else outcome
        nomothetic_idiographic = st.selectbox("Idiographic/Nomothetic", ["Idiographic", "Nomothetic"], key=f"nom_idio_{graph_index}")
        ml_model = st.selectbox(f"ML Model ({graph_index})", ["Elastic Net (en)", "Random Forest (rf)"], key=f"ml_model_{graph_index}")

        # Determine correct values
        nom_idio_value = "nomot" if nomothetic_idiographic == "Nomothetic" else "idiog"
        ml_model_short = "en" if ml_model == "Elastic Net (en)" else "rf"

        # Construct File Name
        file_name = f"True_vs_predicted_comb_{ml_model_short}_{outcome}_{nom_idio_value}.csv"
        file_path = os.path.join(DATA_DIR, file_name)

        # Check if the file has already been used in another graph
        duplicate_graph_number = st.session_state.used_files.get(file_name)

        # If the file is already used, show a warning and **skip the graph**
        # if duplicate_graph_number and duplicate_graph_number != graph_index:
        #     st.warning("⚠️ The selected properties have already been used.")
        #     # return

        # Load CSV File
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()  # Add lowercase conversion
            
            # Get unique participants from column D ("Participant")
            participants = df['participant'].unique()
            selected_participant = st.selectbox(f"Participant ({graph_index})", participants, key=f"participant_{graph_index}")

            # **Now show success message AFTER participant selection**
            # st.success(f"Loaded data: **{file_name}**")

        else:
            st.warning("⚠️ The selected properties have already been used.")  # Custom warning message
            return  # Stop execution for this graph but keep others

        # Move "Add Another Graph" button here to align with dropdowns
        if st.button("➕ Add Another Graph to Compare", key=f"add_graph_{graph_index}"):
            st.session_state.graph_count += 1
            st.rerun()  # Refresh to display the new section

    # **Updated Graph Placement: Right Column (3/4 Width)**
    with right_col:
        st.write("")  # Add space at the top to center the graph vertically

        # Filter data for the selected participant
        participant_data = df[df['participant'] == selected_participant]

        # Process Data
        time = participant_data['time']
        na_levels = participant_data['na_ratings']
        na_predicted = participant_data['predicted_estimates']

        # Compute correlation coefficient
        correlation = np.corrcoef(na_levels, na_predicted)[0, 1] if len(na_levels) > 1 else np.nan

        # Create Plotly Figure (Fixed Width & Height)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=na_levels, mode='lines', name='Actual', line=dict(color='teal', width=2)))
        fig.add_trace(go.Scatter(x=time, y=na_predicted, mode='lines', name='Predicted', line=dict(color='red', width=2, dash='dash')))

        # Customize Layout (Fixed Width & Height, Centered)
        fig.update_layout(
            title=f"Participant: {selected_participant} | R = {correlation:.2f}",
            xaxis_title="Time",
            yaxis_title="NA Levels",
            template="plotly_white",
            width=900,  # **Fixed Width for 3/4 of screen**
            height=400,  # Adjusted height for better visibility
            margin=dict(l=10, r=10, t=30, b=20),  # Adjust margins for centering
        )

        # Display the plot (Center it inside the column)
        st.write("")  # Spacer
        st.plotly_chart(fig, key=f"plot_{graph_index}")
        st.write("")  # Spacer to further help with centering

    # Store this file as used for this graph index
    st.session_state.used_files[file_name] = graph_index

# **Loop through all graphs the user has added**
for i in range(1, st.session_state.graph_count + 1):
    load_and_plot_graph(i)

# Close center alignment div
st.write("</div>", unsafe_allow_html=True)

# **Add the chatbot to the page**
app_with_chatbot.show_chatbot_ui()


# WORKING