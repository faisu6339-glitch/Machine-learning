import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📉 Bagging Regressor Visualization")

# -------------------------------------------------
# SIDEBAR - DATASET
# -------------------------------------------------
st.sidebar.header("📊 Dataset Configuration")

n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 500)
noise = st.sidebar.slider("Noise", 0.0, 50.0, 15.0)

# -------------------------------------------------
# DATASET CREATION
# -------------------------------------------------
X, y = make_regression(
    n_samples=n_samples,
    n_features=1,
    noise=noise,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------------------------
# SIDEBAR - MODEL CONFIG
# -------------------------------------------------
st.sidebar.header("⚙️ Bagging Configuration")

base_estimator_name = st.sidebar.selectbox(
    "Base Estimator",
    ["Decision Tree", "SVR", "KNN"]
)

n_estimators = st.sidebar.slider("Number of Estimators", 1, 100, 25)
max_samples = st.sidebar.slider("Max Samples", 0.1, 1.0, 0.8)
max_features = st.sidebar.slider("Max Features", 0.1, 1.0, 1.0)

bootstrap_samples = st.sidebar.radio("Bootstrap Samples", [True, False])
bootstrap_features = st.sidebar.radio("Bootstrap Features", [False, True])

# -------------------------------------------------
# BASE ESTIMATOR SELECTION
# -------------------------------------------------
if base_estimator_name == "Decision Tree":
    base_estimator = DecisionTreeRegressor(max_depth=5, random_state=42)
    use_scaling = False

elif base_estimator_name == "SVR":
    base_estimator = SVR(kernel="rbf")
    use_scaling = True

else:  # KNN
    base_estimator = KNeighborsRegressor(n_neighbors=5)
    use_scaling = True

# -------------------------------------------------
# BUILD MODEL
# -------------------------------------------------
bagging = BaggingRegressor(
    estimator=base_estimator,
    n_estimators=n_estimators,
    max_samples=max_samples,
    max_features=max_features,
    bootstrap=bootstrap_samples,
    bootstrap_features=bootstrap_features,
    random_state=42
)

if use_scaling:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("bagging", bagging)
    ])
else:
    model = bagging

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

st.sidebar.success(f"R² Score: {r2:.3f}")

# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------
st.subheader("📈 Actual vs Predicted")

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(X_test, y_test, color="blue", label="Actual", alpha=0.6)
ax.scatter(X_test, y_pred, color="red", label="Predicted", alpha=0.6)

ax.set_title(
    f"Bagging Regressor with {base_estimator_name}\nR² = {r2:.3f}"
)
ax.set_xlabel("Feature X")
ax.set_ylabel("Target y")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -------------------------------------------------
# MODEL SUMMARY
# -------------------------------------------------
st.subheader("📌 Model Summary")

st.write(f"""
- **Base Estimator**: {base_estimator_name}
- **Number of Estimators**: {n_estimators}
- **Max Samples**: {max_samples}
- **Max Features**: {max_features}
- **Bootstrap Samples**: {bootstrap_samples}
- **Bootstrap Features**: {bootstrap_features}
- **R² Score**: {r2:.3f}
""")
