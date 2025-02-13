
import streamlit as st
import streamlit.components.v1 as components



st.set_page_config(page_title="ML Analysis Dashboard", layout="wide")
components.html(
    open("threed.html").read(),
    height=800, # Full screen height width=None,  # Full width
    width=None,  # Full width
    scrolling=False
)
# st.title("📊 Can Natural Language Processing Track Negative Emotions in the Daily Lives of Adolescents?")

# st.write("Welcome to the Machine Learning Model Analysis Dashboard. Navigate using the sidebar to explore model performance and predictions.")

st.sidebar.success("Select a page above.")





# Embed Three.js visualization
