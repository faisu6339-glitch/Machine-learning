import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(layout="wide")
st.title("🗳️ Voting Classifier Visualization")

# ---------------------------
# CUSTOM DATASETS
# ---------------------------
def make_u_shape(n=500, noise=0.2):
    x = np.linspace(-3, 3, n)
    y = x**2 + np.random.normal(0, noise, n)
    X = np.c_[x, y]
    y_cls = (x > 0).astype(int)
    return X, y_cls


def make_outliers(n=500):
    X, y = make_blobs(n_samples=n, centers=2, random_state=42)
    outliers = np.random.uniform(-8, 8, (40, 2))
    out_y = np.random.randint(0, 2, 40)
    return np.vstack((X, outliers)), np.hstack((y, out_y))


def make_two_spirals(n=500, noise=0.5):
    n = n // 2
    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi
    r1, r2 = 2 * theta, -2 * theta

    X1 = np.c_[np.cos(theta) * r1, np.sin(theta) * r1]
    X2 = np.c_[np.cos(theta) * r2, np.sin(theta) * r2]
    X = np.vstack((X1, X2)) + np.random.randn(2*n, 2) * noise
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


def make_xor(n=500):
    X = np.random.randn(n, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    return X, y


def create_dataset(name):
    if name == "Concentric Circles":
        return make_circles(n_samples=500, noise=0.2, factor=0.4, random_state=42)
    elif name == "Moons":
        return make_moons(n_samples=500, noise=0.25, random_state=42)
    elif name == "Blobs":
        return make_blobs(n_samples=500, centers=2, random_state=42)
    elif name == "U-Shaped":
        return make_u_shape()
    elif name == "Outliers":
        return make_outliers()
    elif name == "Two Spirals":
        return make_two_spirals()
    elif name == "XOR":
        return make_xor()

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("⚙️ Configuration")

dataset_name = st.sidebar.selectbox(
    "Dataset",
    [
        "Concentric Circles",
        "Moons",
        "Blobs",
        "U-Shaped",
        "Outliers",
        "Two Spirals",
        "XOR"
    ]
)

estimators_selected = st.sidebar.multiselect(
    "Estimators",
    ["Logistic Regression", "KNN", "Gaussian Naive Bayes", "SVM"],
    default=["Logistic Regression", "Gaussian Naive Bayes"]
)

voting_type = st.sidebar.radio("Voting Type", ["hard", "soft"])
run = st.sidebar.button("Run Algorithm")

# ---------------------------
# MAIN
# ---------------------------
if run:

    if len(estimators_selected) != 2:
        st.error("⚠️ Select EXACTLY TWO estimators")
        st.stop()

    X, y = create_dataset(dataset_name)

    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    models = {}
    estimators = []

    for est in estimators_selected:
        if est == "Logistic Regression":
            m = LogisticRegression()
            models["Logistic Regression"] = m
            estimators.append(("lr", m))

        elif est == "KNN":
            m = KNeighborsClassifier(n_neighbors=7)
            models["KNN"] = m
            estimators.append(("knn", m))

        elif est == "Gaussian Naive Bayes":
            m = GaussianNB()
            models["Gaussian Naive Bayes"] = m
            estimators.append(("gnb", m))

        elif est == "SVM":
            m = SVC(probability=True)
            models["SVM"] = m
            estimators.append(("svm", m))

    # Train individual models
    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracies[name] = accuracy_score(y_test, model.predict(X_test))

    # Voting model
    voting = VotingClassifier(estimators=estimators, voting=voting_type)
    voting.fit(X_train, y_train)
    voting_acc = accuracy_score(y_test, voting.predict(X_test))

    # Sidebar metrics
    st.sidebar.markdown("### 📊 Accuracies")
    st.sidebar.write(f"🗳️ Voting : **{voting_acc:.3f}**")
    for k, v in accuracies.items():
        st.sidebar.write(f"{k} : **{v:.3f}**")

    # Mesh
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    def plot(model, title):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(6,6))
        ax.contourf(xx, yy, Z, alpha=0.3)
        ax.scatter(X[:,0], X[:,1], c=y, edgecolor="k")
        ax.set_title(title)
        return fig

    # Layout
    top = st.container()
    col1, col2 = st.columns(2)

    with top:
        st.subheader("🗳️ Voting Classifier")
        st.pyplot(plot(voting, f"Voting ({voting_type.upper()}) | Acc={voting_acc:.3f}"))

    names = list(models.keys())

    with col1:
        st.pyplot(plot(models[names[0]], f"{names[0]} | Acc={accuracies[names[0]]:.3f}"))

    with col2:
        st.pyplot(plot(models[names[1]], f"{names[1]} | Acc={accuracies[names[1]]:.3f}"))
