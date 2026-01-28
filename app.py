# =================================================
# EarthScape – Surficial Geology Classifier
# Cloud-Safe Full Pipeline (RF + LightGBM)
# =================================================

import os, glob, zipfile, tempfile, time
import numpy as np
import pandas as pd
import streamlit as st
import rasterio as rio
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# =================================================
# Page config
# =================================================
st.set_page_config(page_title="EarthScape – Surficial Geology Classifier", layout="wide")

# =================================================
# Session state
# =================================================
for k in ["rf_results", "lgbm_results", "patch_dir"]:
    if k not in st.session_state:
        st.session_state[k] = None

# =================================================
# File handling
# =================================================
def extract_zip(uploaded_zip):
    tmpdir = tempfile.mkdtemp()
    zip_path = os.path.join(tmpdir, uploaded_zip.name)

    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmpdir)

    return tmpdir

# =================================================
# Data functions
# =================================================
def load_patches(patch_dir):
    patch_dirs = sorted([p for p in glob.glob(os.path.join(patch_dir, "256_50_*")) if os.path.isdir(p)])
    all_patches = {}

    for folder in patch_dirs:
        patch_name = os.path.basename(folder)
        modalities = {}

        for tif in glob.glob(os.path.join(folder, "*.tif")):
            key = os.path.splitext(os.path.basename(tif))[0]
            if "geology" in key.lower():
                continue
            with rio.open(tif) as src:
                modalities[key] = src.read(1)

        all_patches[patch_name] = modalities

    return all_patches

def extract_labels(csv_file, n):
    df = pd.read_csv(csv_file)
    df["dominant"] = df.iloc[:, 1:].idxmax(axis=1)
    return df["dominant"][:n]

def extract_features(patch_data):
    rows = []

    for patch, modalities in patch_data.items():
        for name, arr in modalities.items():
            flat = arr.flatten()
            rows.append({
                "patch": patch, "modality": name,
                "min": np.min(flat), "max": np.max(flat),
                "mean": np.mean(flat), "median": np.median(flat),
                "std": np.std(flat)
            })

    df = pd.DataFrame(rows)
    wide = df.pivot_table(index="patch", columns="modality",
                           values=["min", "max", "mean", "median", "std"])
    wide.columns = [f"{s}_{m}" for s, m in wide.columns]
    return wide.reset_index(drop=True)

# =================================================
# Models
# =================================================
def load_model(choice):
    if choice == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=10,
            min_samples_split=50, min_samples_leaf=20,
            n_jobs=-1, random_state=42
        )
    else:
        return LGBMClassifier(
            num_leaves=200, learning_rate=0.05,
            n_estimators=500, objective="multiclass",
            random_state=42
        )

# =================================================
# Pipeline
# =================================================
def run_pipeline(model_choice, base_dir, csv_file, bar, status):

    timings = {}
    start = time.time()

    # 1 Load patches
    t0 = time.time()
    bar.progress(10); status.info("Loading patches...")
    patch_data = load_patches(base_dir)
    timings["Loading patches"] = time.time() - t0

    # 2 Labels
    t0 = time.time()
    bar.progress(25); status.info("Extracting labels...")
    y = extract_labels(csv_file, len(patch_data))
    timings["Extracting labels"] = time.time() - t0

    # 3 Features
    t0 = time.time()
    bar.progress(40); status.info("Extracting features...")
    X = extract_features(patch_data)
    timings["Feature extraction"] = time.time() - t0

    # 4 Split
    t0 = time.time()
    bar.progress(55); status.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    timings["Train-test split"] = time.time() - t0

    # 5 Train
    t0 = time.time()
    bar.progress(70); status.info("Training model...")
    model = load_model(model_choice)
    model.fit(X_train, y_train)
    timings["Model training"] = time.time() - t0

    # 6 Evaluate
    t0 = time.time()
    bar.progress(85); status.info("Evaluating model...")
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1": f1_score(y_test, preds, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_test, preds),
        "confusion": confusion_matrix(y_test, preds)
    }
    timings["Evaluation"] = time.time() - t0

    # 7 SHAP
    t0 = time.time()
    bar.progress(95); status.info("Computing SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test[:100])
    timings["SHAP"] = time.time() - t0

    bar.progress(100); status.success("Pipeline completed successfully.")
    timings["Total runtime"] = time.time() - start

    return metrics, preds, y_test, shap_values, timings

# =================================================
# UI
# =================================================
st.markdown("<h1 style='text-align:center;'>🌍 EarthScape – Surficial Geology Classifier</h1>", unsafe_allow_html=True)
st.divider()

st.sidebar.header("Upload Inputs")

zip_file = st.sidebar.file_uploader("Upload patches ZIP", type=["zip"])
csv_file = st.sidebar.file_uploader("Upload areas CSV", type=["csv"])

if zip_file:
    st.session_state.patch_dir = extract_zip(zip_file)
    st.sidebar.success("ZIP extracted successfully")

tab1, tab2 = st.tabs(["Random Forest", "LightGBM"])

def render(results):
    metrics, preds, y_test, shap_values, timings = results

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Performance")
        st.metric("Accuracy", round(metrics["accuracy"],3))
        st.metric("Precision", round(metrics["precision"],3))
        st.metric("Recall", round(metrics["recall"],3))
        st.metric("F1-score", round(metrics["f1"],3))
        st.write("**Cohen's Kappa:**", round(metrics["kappa"],3))

    with c2:
        st.subheader("Execution Time (seconds)")
        st.dataframe(pd.DataFrame(timings.items(), columns=["Stage","Seconds"]))

    st.subheader("Confusion Matrix")
    st.dataframe(pd.DataFrame(metrics["confusion"]))

# ---------------- RF ----------------
with tab1:
    if st.button("Run Random Forest"):
        if not zip_file or not csv_file:
            st.error("Please upload both ZIP and CSV first.")
        else:
            bar = st.progress(0)
            status = st.empty()
            st.session_state.rf_results = run_pipeline("Random Forest", st.session_state.patch_dir, csv_file, bar, status)

    if st.session_state.rf_results:
        render(st.session_state.rf_results)

# ---------------- LGBM ----------------
with tab2:
    if st.button("Run LightGBM"):
        if not zip_file or not csv_file:
            st.error("Please upload both ZIP and CSV first.")
        else:
            bar = st.progress(0)
            status = st.empty()
            st.session_state.lgbm_results = run_pipeline("LightGBM", st.session_state.patch_dir, csv_file, bar, status)

    if st.session_state.lgbm_results:
        render(st.session_state.lgbm_results)

st.caption("EarthScape – Automating Insight. Empowering Geoscience.")





