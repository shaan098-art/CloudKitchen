
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from utils import plot_conf_matrix

st.header("Classification Suite")

df = st.session_state.get('df')
if df is None:
    st.error("Dataset missing."); st.stop()

target = st.selectbox("Target label", ["Q18_ZeroWasteSubscription", "Q23_ChurnIntent"])
algo_choices = ["KNN", "DecisionTree", "RandomForest", "GradientBoosting"]
test_size = st.slider("Test size fraction", 0.1, 0.4, 0.2, 0.05)

# Prepare X, y
X_full = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
y_full = df[target]

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=42, stratify=y_full)

models = {
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=250, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

metrics_rows = []
roc_curves = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics_rows.append({"Algorithm": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})

    if len(y_full.unique()) == 2:   # binary ROC
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr)

    # store confusion matrix
    models[name] = (model, y_pred)

st.subheader("Performance Comparison")
st.dataframe(pd.DataFrame(metrics_rows).set_index("Algorithm").style.format("{:.2%}"))

# Toggle confusion matrix
with st.expander("Show Confusion Matrix"):
    algo = st.selectbox("Select algorithm", algo_choices)
    cm = confusion_matrix(y_test, models[algo][1])
    fig_cm = plot_conf_matrix(cm, class_labels=sorted(y_full.unique()))
    st.pyplot(fig_cm)

# ROC Curve
if roc_curves:
    st.subheader("ROC Curves (binary classification only)")
    fig, ax = plt.subplots()
    for name, (fpr, tpr) in roc_curves.items():
        ax.plot(fpr, tpr, label=name)
    ax.plot([0,1], [0,1], linestyle='--', color='grey')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# Upload new data
st.markdown("---")
st.subheader("Predict on New Data")
uploaded = st.file_uploader("Upload CSV (same schema, no target column)", type="csv")
if uploaded:
    new_df = pd.read_csv(uploaded)
    new_X = pd.get_dummies(new_df, drop_first=True)
    new_X = new_X.reindex(columns=X_full.columns, fill_value=0)
    chosen_algo = st.selectbox("Choose model for inference", algo_choices, key="infer_algo")
    pred = models[chosen_algo][0].predict(new_X)
    new_df[target + "_pred"] = pred
    st.write(new_df.head())
    csv = new_df.to_csv(index=False).encode()
    st.download_button("Download predictions", data=csv, file_name="predictions.csv", mime="text/csv")
