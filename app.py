
import streamlit as st
from utils import load_data

st.set_page_config(page_title="Cloud Kitchen Dashboard", layout="wide")

st.title("Cloud Kitchen Consumer Insights Dashboard :fork_and_knife:")

with st.sidebar:
    st.markdown("### Data Source")
    default_url = st.secrets.get("csv_url", "https://raw.githubusercontent.com/<your_username>/<your_repo>/main/data/cloud_kitchen_survey_descriptive.csv")
    csv_url = st.text_input("Raw CSV URL (GitHub or web)", value=default_url)
    if st.button("Reload data"):
        st.session_state['df'] = load_data(csv_url)

if 'df' not in st.session_state:
    st.session_state['df'] = load_data(csv_url)

st.success("Use the left navigation pane (or > menu) to explore visualisations, models, and insights.")
