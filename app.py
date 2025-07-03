# app.py  ── main entry point
from pathlib import Path
import requests
import pandas as pd
import streamlit as st
from io import StringIO

st.set_page_config(page_title="Cloud Kitchen Dashboard", page_icon="🍲", layout="wide")
st.title("Cloud-Kitchen Consumer Insights Dashboard")

# -----------------------------------------------------------------------
# Helper: robust CSV loader (works with local paths *and* URLs)
# -----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def robust_read_csv(src: str | Path) -> pd.DataFrame:
    src = str(src)                      # normalise type
    try:
        if src.startswith(("http://", "https://")):
            r = requests.get(src, timeout=15)
            r.raise_for_status()
            return pd.read_csv(StringIO(r.text))
        else:
            return pd.read_csv(Path(src).expanduser())
    except Exception as e:
        st.error(f"❌ Failed to load **{src}** – {e}")
        raise

# -----------------------------------------------------------------------
# Default LOCAL file that lives inside the repo
#   ☞  make sure you committed this file!
# -----------------------------------------------------------------------
DEFAULT_LOCAL = Path(__file__).parent / "data" / "cloud_kitchen_survey_descriptive.csv"

# Sidebar – let power-users override the source
with st.sidebar:
    st.markdown("### Data source")
    csv_input = st.text_input(
        "Local path or URL",
        value=str(DEFAULT_LOCAL)        # pre-filled with the local file
    )
    if st.button("Reload data"):
        st.session_state.pop("df", None)  # clear cache

# First load (cached thereafter)
if "df" not in st.session_state:
    st.session_state["df"] = robust_read_csv(csv_input)

df = st.session_state["df"]

st.success(f"Loaded **{len(df):,} rows** from **{csv_input}**")
st.write(
    "Use the left-hand Pages menu to explore visualisations, "
    "classification, clustering, association rules, and regression tabs."
)
