# app.py – main entry point
from pathlib import Path
import streamlit as st
from utils import load_data   # <- same helper you already have

st.set_page_config(page_title="Cloud Kitchen Dashboard",
                   page_icon="🍲",
                   layout="wide")

st.title("Cloud-Kitchen Consumer Insights Dashboard")

# ------------------------------------------------------------------
# 1️⃣  Default LOCAL dataset path (relative to this file)
# ------------------------------------------------------------------
DEFAULT_LOCAL = Path(__file__).parent / "data" / "cloud_kitchen_survey_descriptive.csv"

# ------------------------------------------------------------------
# 2️⃣  Sidebar – allow user to override path/URL if they wish
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Data source")
    csv_path = st.text_input(
        label="CSV path or URL",
        value=str(DEFAULT_LOCAL)  # 👈 pre-filled with the local file
    )

    if st.button("Reload data"):
        # Clear any cached dataframe so the new path reloads
        st.session_state.pop("df", None)

# ------------------------------------------------------------------
# 3️⃣  Lazy-load the dataframe (cached by utils.load_data)
# ------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state["df"] = load_data(csv_path)

df = st.session_state["df"]

# ------------------------------------------------------------------
# 4️⃣  Basic confirmation / housekeeping
# ------------------------------------------------------------------
st.success(f"Loaded **{len(df):,} rows** from **{csv_path}**")
st.write(
    "Use the navigation menu (≡ or ‘Pages’ pane) to explore visualisations, "
    "classification, clustering, association rules, and regression insights."
)
