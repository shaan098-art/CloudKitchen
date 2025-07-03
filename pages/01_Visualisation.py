
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from utils import load_data

st.header("Data Visualisation :bar_chart:")

df = st.session_state.get('df')
if df is None:
    st.error("Dataset not loaded!")
    st.stop()

# Filters
with st.expander("Filters"):
    age_filter = st.multiselect("Age Brackets", options=sorted(df['Q1_AgeBracket'].unique()), default=sorted(df['Q1_AgeBracket'].unique()))
    diet_filter = st.multiselect("Diet Style", options=sorted(df['Q3_DietStyle'].unique()), default=sorted(df['Q3_DietStyle'].unique()))

filt_df = df[df['Q1_AgeBracket'].isin(age_filter) & df['Q3_DietStyle'].isin(diet_filter)]

st.subheader("1. Age Distribution")
fig1 = px.histogram(filt_df, x="Q1_AgeBracket", nbins=6, title="Age Bracket Distribution")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("2. Diet Style Counts")
fig2 = px.bar(filt_df['Q3_DietStyle'].value_counts().reset_index(),
              x='index', y='Q3_DietStyle', labels={'index':'Diet Style','Q3_DietStyle':'Count'})
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. Weekly Orders Distribution")
fig3 = px.box(filt_df, y="Q6_OrdersPerWeek")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("4. Average Spend by Age Bracket")
spend_age = filt_df.groupby('Q1_AgeBracket')['Q8_AvgSpendAED'].mean().reset_index()
fig4 = px.line(spend_age, x='Q1_AgeBracket', y='Q8_AvgSpendAED', markers=True)
st.plotly_chart(fig4, use_container_width=True)

st.subheader("5. Tip % vs Eco Priority")
fig5 = px.scatter(filt_df, x="Q5_EcoPriority", y="Q9_TipPct", trendline="ols")
st.plotly_chart(fig5, use_container_width=True)

st.subheader("6. Eco Priority vs Zero‑Waste Adoption")
eco_zws = filt_df.groupby(['Q5_EcoPriority','Q18_ZeroWasteSubscription']).size().reset_index(name='count')
fig6 = px.bar(eco_zws, x='Q5_EcoPriority', y='count', color='Q18_ZeroWasteSubscription', barmode='group')
st.plotly_chart(fig6, use_container_width=True)

st.subheader("7. Group Cart Usage by Age")
fig7 = px.box(filt_df, x="Q1_AgeBracket", y="Q21_GroupCartUsage", points="all")
st.plotly_chart(fig7, use_container_width=True)

st.subheader("8. Correlation Heatmap (numeric)")
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
num_df = filt_df.select_dtypes(include=['int','float'])
corr = num_df.corr()
fig8, ax8 = plt.subplots(figsize=(10,6))
sns.heatmap(corr, ax=ax8, center=0, cmap='coolwarm')
st.pyplot(fig8)

st.subheader("9. Spice Level Distribution")
fig9 = px.pie(filt_df, names='Q12_SpiceLevel')
st.plotly_chart(fig9, use_container_width=True)

st.subheader("10. Loyalty Perk Preference")
fig10 = px.bar(filt_df['Q19_LoyaltyPerk'].value_counts().reset_index(),
               x='index', y='Q19_LoyaltyPerk', labels={'index':'Perk','Q19_LoyaltyPerk':'Count'})
st.plotly_chart(fig10, use_container_width=True)
