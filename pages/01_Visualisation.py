# pages/01_Visualisation.py
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.header("Data Visualisation :bar_chart:")

# ------------------------------------------------------------------
# Retrieve the dataframe loaded in app.py
# ------------------------------------------------------------------
df = st.session_state.get("df")
if df is None:
    st.error("Dataset not loaded – open the Home page first.")
    st.stop()

# ------------------------------------------------------------------
# Interactive filters
# ------------------------------------------------------------------
with st.expander("Filters"):
    age_filter = st.multiselect(
        "Age Brackets",
        options=sorted(df["Q1_AgeBracket"].unique()),
        default=sorted(df["Q1_AgeBracket"].unique()),
    )
    diet_filter = st.multiselect(
        "Diet Style",
        options=sorted(df["Q3_DietStyle"].unique()),
        default=sorted(df["Q3_DietStyle"].unique()),
    )

filt_df = df[
    df["Q1_AgeBracket"].isin(age_filter) & df["Q3_DietStyle"].isin(diet_filter)
]

# ------------------------------------------------------------------
# 1. Age-Bracket distribution
# ------------------------------------------------------------------
st.subheader("1. Age-Bracket Distribution")
fig1 = px.histogram(
    filt_df,
    x="Q1_AgeBracket",
    nbins=6,
    title="Age-Bracket Distribution",
    labels={"Q1_AgeBracket": "Age Bracket"},
)
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------------------------
# 2. Diet-Style counts  🛠  (robust)
# ------------------------------------------------------------------
st.subheader("2. Diet-Style Counts")

style_counts = (
    filt_df["Q3_DietStyle"]
    .value_counts()
    .sort_index()              # keeps codes 1-6 in order
    .reset_index(name="Count") # FORCE second column to be 'Count'
    .rename(columns={"index": "DietStyle"})
)

if style_counts.empty:
    st.info("No rows match the current filters.")
else:
    fig2 = px.bar(
        style_counts,
        x="DietStyle",
        y="Count",
        labels={"DietStyle": "Diet Style", "Count": "Count"},
    )
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------
# 3. Weekly Orders distribution
# ------------------------------------------------------------------
st.subheader("3. Weekly Orders Distribution")
fig3 = px.box(filt_df, y="Q6_OrdersPerWeek")
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------------
# 4. Average Spend by Age Bracket
# ------------------------------------------------------------------
st.subheader("4. Average Spend by Age Bracket")
spend_age = (
    filt_df.groupby("Q1_AgeBracket")["Q8_AvgSpendAED"]
    .mean()
    .reset_index()
)
fig4 = px.line(
    spend_age,
    x="Q1_AgeBracket",
    y="Q8_AvgSpendAED",
    markers=True,
    labels={"Q1_AgeBracket": "Age Bracket", "Q8_AvgSpendAED": "Avg Spend (AED)"},
)
st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------------------------------
# 5. Tip % vs Eco-Priority
# ------------------------------------------------------------------
st.subheader("5. Tip % vs Eco Priority")
fig5 = px.scatter(
    filt_df,
    x="Q5_EcoPriority",
    y="Q9_TipPct",
    trendline="ols",
    labels={"Q5_EcoPriority": "Eco Priority (1-5)", "Q9_TipPct": "Tip %"},
)
st.plotly_chart(fig5, use_container_width=True)

# ------------------------------------------------------------------
# 6. Eco-Priority vs Zero-Waste adoption
# ------------------------------------------------------------------
st.subheader("6. Eco-Priority vs Zero-Waste Adoption")
eco_zws = (
    filt_df.groupby(["Q5_EcoPriority", "Q18_ZeroWasteSubscription"])
    .size()
    .reset_index(name="Count")
)
fig6 = px.bar(
    eco_zws,
    x="Q5_EcoPriority",
    y="Count",
    color="Q18_ZeroWasteSubscription",
    barmode="group",
    labels={"Q5_EcoPriority": "Eco Priority", "Q18_ZeroWasteSubscription": "Subscription"},
)
st.plotly_chart(fig6, use_container_width=True)

# ------------------------------------------------------------------
# 7. Group-Cart usage by Age
# ------------------------------------------------------------------
st.subheader("7. Group-Cart Usage by Age")
fig7 = px.box(
    filt_df,
    x="Q1_AgeBracket",
    y="Q21_GroupCartUsage",
    points="all",
    labels={"Q1_AgeBracket": "Age Bracket", "Q21_GroupCartUsage": "Usage (1-5)"},
)
st.plotly_chart(fig7, use_container_width=True)

# ------------------------------------------------------------------
# 8. Correlation heatmap (numeric)
# ------------------------------------------------------------------
st.subheader("8. Correlation Heatmap (numeric features)")
num_df = filt_df.select_dtypes(include=["int", "float"])
if num_df.empty or num_df.shape[0] < 2:
    st.info("Not enough rows for a correlation heatmap after filters.")
else:
    corr = num_df.corr()
    fig8, ax8 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, ax=ax8, center=0, cmap="coolwarm")
    st.pyplot(fig8)

# ------------------------------------------------------------------
# 9. Spice-Level distribution
# ------------------------------------------------------------------
st.subheader("9. Spice-Level Distribution")
fig9 = px.pie(
    f
