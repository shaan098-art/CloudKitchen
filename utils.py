
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load survey data from local path or URL.
    Uses streamlit caching to avoid re‑download / re‑parsing.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Could not load CSV: {e}")
        raise
    return df

def preprocessing_pipeline(df: pd.DataFrame, target: str):
    """
    Build a sklearn Pipeline that one‑hot‑encodes categoricals and leaves numeric untouched.
    Returns fitted pipeline, X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target])
    y = df[target]

    numeric_cols = X.select_dtypes(include=['int','float']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # OneHotEncoder for categoricals, ignore unknown to accept new values on uploaded data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    model_input = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(model_input, y, test_size=0.2, random_state=42, stratify=y)

    return preprocessor, X_train, X_test, y_train, y_test

def get_classification_metrics(y_true, y_pred, average="weighted"):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0)
    }

def plot_conf_matrix(cm, class_labels):
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig
