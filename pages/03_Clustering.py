
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px

st.header("Customer Segmentation (K‑Means)")

df = st.session_state.get('df')
if df is None:
    st.error("Data missing"); st.stop()

numeric_cols = df.select_dtypes(include=['int','float']).columns.tolist()
feature_opts = st.multiselect("Features to cluster on", numeric_cols, default=["Q8_AvgSpendAED","Q6_OrdersPerWeek","Q9_TipPct","Q5_EcoPriority"])

if len(feature_opts) < 2:
    st.warning("Select at least 2 features."); st.stop()

X = df[feature_opts]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow plot
st.subheader("Elbow Method")
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(X_scaled)
    inertias.append(km.inertia_)
fig, ax = plt.subplots()
ax.plot(list(K_range), inertias, marker='o')
ax.set_xlabel("k")
ax.set_ylabel("Inertia")
st.pyplot(fig)

# Slider for clusters
k = st.slider("Select number of clusters", 2, 10, 4)
km_final = KMeans(n_clusters=k, random_state=42, n_init='auto')
df['cluster'] = km_final.fit_predict(X_scaled)

# Persona table
st.subheader("Cluster Personas")
persona = df.groupby('cluster')[feature_opts].mean().round(2)
st.dataframe(persona)

# Download button
csv = df.to_csv(index=False).encode()
st.download_button("Download data with cluster labels", csv, "clustered_data.csv", mime="text/csv")
