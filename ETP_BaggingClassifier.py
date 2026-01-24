import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("🌲 Bagging Classifier Visualization")

# -------------------------------------------------
# SIDEBAR - DATASET
# -------------------------------------------------
st.sidebar.header("📊 Dataset Configuration")

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ["Moons", "Circles", "Blobs"]
)

n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 500)

# -------------------------------------------------
# DATASET CREATION
# -------------------------------------------------
def create_dataset(name, n):
    if name == "Moons":
        X, y = make_moons(n_samples=n, noise=0.25, random_state=42)
    elif name == "Circles":
        X, y = make_circles(n_samples=n, noise=0.15, factor=0.5, random_state=42)
    else:
        X, y = make_blobs(n_samples=n, centers=2, cluster_std=1.2, random_state=42)
    return X, y

X, y = create_dataset(dataset_name, n_samples)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------------------------
# SIDEBAR - MODEL CONFIG
# -------------------------------------------------
st.sidebar.header("⚙️ Bagging Configuration")

base_estimator_name = st.sidebar.selectbox(
    "Base Estimator",
    ["Decision Tree", "SVM", "KNN"]
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
    base_estimator = DecisionTreeClassifier(max_depth=5, random_state=42)
    use_scaling = False

elif base_estimator_name == "SVM":
    base_estimator = SVC(kernel="rbf", probability=True)
    use_scaling = True

else:  # KNN
    base_estimator = KNeighborsClassifier(n_neighbors=5)
    use_scaling = True

# -------------------------------------------------
# BUILD MODEL PIPELINE
# -------------------------------------------------
bagging = BaggingClassifier(
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
accuracy = accuracy_score(y_test, y_pred)

st.sidebar.success(f"Accuracy: {accuracy:.3f}")

# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------
st.subheader("📈 Data Visualization")

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(
    X_test[y_test == 0, 0],
    X_test[y_test == 0, 1],
    c="red",
    label="Class 0",
    alpha=0.7
)

ax.scatter(
    X_test[y_test == 1, 0],
    X_test[y_test == 1, 1],
    c="purple",
    label="Class 1",
    alpha=0.7
)

ax.set_title(
    f"Bagging with {base_estimator_name}\nAccuracy = {accuracy:.3f}"
)
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -------------------------------------------------
# PREDICTION SUMMARY
# -------------------------------------------------
st.subheader("📌 Model Summary")

st.write(f"""
- **Dataset**: {dataset_name}
- **Base Estimator**: {base_estimator_name}
- **Estimators**: {n_estimators}
- **Max Samples**: {max_samples}
- **Max Features**: {max_features}
- **Bootstrap Samples**: {bootstrap_samples}
- **Bootstrap Features**: {bootstrap_features}
""")
