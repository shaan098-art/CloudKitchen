
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.header("Association Rule Mining")

df = st.session_state.get('df')
if df is None:
    st.error("Data missing"); st.stop()

# Candidate columns (object columns with comma separated strings)
candidate_cols = [c for c in df.select_dtypes(include='object').columns if df[c].str.contains(',').any()]
if not candidate_cols:
    st.info("No suitable columns with comma‑separated items found."); st.stop()

cols = st.multiselect("Select columns (at least 2)", candidate_cols, default=candidate_cols[:2])
if len(cols) < 2:
    st.warning("Select at least two columns."); st.stop()

min_support = st.slider("Minimum support", 0.01, 0.5, 0.05, 0.01)
min_conf = st.slider("Minimum confidence", 0.1, 1.0, 0.5, 0.05)

# Prepare basket encoding
baskets = []
for col in cols:
    baskets.append(df[col].str.get_dummies(sep=','))

basket_df = pd.concat(baskets, axis=1)
basket_df = basket_df.groupby(basket_df.columns, axis=1).max()  # merge same item names if duplicates

frequent = apriori(basket_df, min_support=min_support, use_colnames=True)
rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
rules = rules.sort_values("confidence", ascending=False).head(10)

st.subheader("Top-10 Rules")
st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])
