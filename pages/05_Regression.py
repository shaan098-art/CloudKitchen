
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

st.header("Regression Insights")

df = st.session_state.get('df')
if df is None:
    st.error("Dataset missing"); st.stop()

target = st.selectbox("Choose numeric target", ["Q8_AvgSpendAED","Q9_TipPct","Q6_OrdersPerWeek","Q10_DeliveryRadiusKm"])
feature_cols = st.multiselect("Predictor features", [c for c in df.columns if c != target], default=[c for c in df.select_dtypes(include=['int','float']).columns if c!=target])

X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scal = StandardScaler()
X_train_scaled = scal.fit_transform(X_train)
X_test_scaled = scal.transform(X_test)

models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "DecisionTree": DecisionTreeRegressor(random_state=42)
}

results = []
for name, model in models.items():
    if name in ["Ridge","Lasso"]:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)
    results.append({"Model": name, "R2": r2, "MAE": mae, "RMSE": rmse})

st.subheader("Performance Metrics")
st.dataframe(pd.DataFrame(results).set_index("Model").style.format("{:.3f}"))

# Scatter plot of actual vs predicted for best model
best_model = max(results, key=lambda x: x["R2"])["Model"]
st.subheader(f"Actual vs Predicted ({best_model})")
if best_model in ["Ridge","Lasso"]:
    pred_best = models[best_model].predict(X_test_scaled)
else:
    pred_best = models[best_model].predict(X_test)
fig, ax = plt.subplots()
ax.scatter(y_test, pred_best)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
st.pyplot(fig)
