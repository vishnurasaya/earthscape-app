#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, glob, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import rasterio as rio
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(page_title="EarthScape – Surficial Geology Classifier", layout="wide")

# -------------------------------------------------
# Session state initialization (for persistence)
# -------------------------------------------------
if "rf_results" not in st.session_state:
    st.session_state.rf_results = None

if "lgbm_results" not in st.session_state:
    st.session_state.lgbm_results = None

# -------------------------------------------------
# Data loading and feature extraction
# -------------------------------------------------
def load_patches(patch_dir):
    modalities = {}
    for tif_path in glob.glob(os.path.join(patch_dir, "*.tif")):
        key = os.path.splitext(os.path.basename(tif_path))[0]
        if "geology" in key.lower():
            continue
        with rio.open(tif_path) as src:
            modalities[key] = {"array": src.read(1)}
    return modalities

def load_all_patches(base_dir):
    patch_dirs = sorted([p for p in glob.glob(os.path.join(base_dir, "256_50_*")) if os.path.isdir(p)])
    return {os.path.basename(p): load_patches(p) for p in patch_dirs}

def extract_labels(areas_csv, num_rows):
    areas = pd.read_csv(areas_csv)
    areas["dominant"] = areas.iloc[:, 1:].idxmax(axis=1)
    return areas[["patch_id", "dominant"]][:num_rows]

def extract_statistical_features(patch_data):
    rows = []
    for patch, modalities in patch_data.items():
        for name, info in modalities.items():
            clean = name.replace(f"{patch}_", "")
            flat = info["array"].flatten()
            rows.append({
                "patch": patch, "modality": clean,
                "min": np.min(flat), "max": np.max(flat),
                "mean": np.mean(flat), "median": np.median(flat),
                "std": np.std(flat)
            })
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index="patch", columns="modality",
                           values=["min", "max", "mean", "median", "std"])
    wide.columns = [f"{stat}_{mod}" for stat, mod in wide.columns]
    return wide.reset_index(drop=True)

def prepare_dataset(df_features, target):
    df_final = pd.concat([df_features, target["dominant"].reset_index(drop=True)], axis=1)
    return df_final.drop(columns=["dominant"]), df_final["dominant"]

# -------------------------------------------------
# Models
# -------------------------------------------------
def load_model(model_choice):
    if model_choice == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=10,
            min_samples_split=50, min_samples_leaf=20,
            max_features="sqrt", bootstrap=True,
            n_jobs=-1, class_weight="balanced_subsample",
            random_state=42
        )
    else:
        return LGBMClassifier(
            num_leaves=200, learning_rate=0.05,
            n_estimators=500, objective="multiclass",
            random_state=42
        )

def train_model(model_choice, X_train, y_train):
    model = load_model(model_choice)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_test, y_pred),
        "confusion": confusion_matrix(y_test, y_pred)
    }, y_pred

def compute_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    return explainer(X_test)

# -------------------------------------------------
# Pipeline
# -------------------------------------------------
def run_pipeline_progress(base_dir, areas_csv, model_choice, bar, status):
    bar.progress(10); status.write("Loading patches...")
    patch_data = load_all_patches(base_dir)

    bar.progress(25); status.write("Extracting labels...")
    target = extract_labels(areas_csv, len(patch_data))

    bar.progress(40); status.write("Extracting features...")
    features = extract_statistical_features(patch_data)

    bar.progress(55); status.write("Preparing dataset...")
    X, y = prepare_dataset(features, target)

    class_summary = pd.DataFrame({
        "Class": y.value_counts().index,
        "Samples": y.value_counts().values,
        "Percentage (%)": (y.value_counts(normalize=True) * 100).round(2)
    })

    bar.progress(70); status.write("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = train_model(model_choice, X_train, y_train)
    metrics, preds = evaluate_model(model, X_test, y_test)

    shap_values = compute_shap(model, X_test)

    return metrics, preds, "computed", {}, y_test, class_summary

# -------------------------------------------------
# UI
# -------------------------------------------------
st.markdown("<h1 style='text-align:center;'>EarthScape Surficial Geology Classifier</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.header("Input Panel")
base_dir = st.sidebar.text_input("Patch folder path")
areas_csv = st.sidebar.text_input("Areas CSV path")

tab_rf, tab_lgbm = st.tabs(["Random Forest", "LightGBM"])

def render_results(metrics, preds, y_test, class_summary):

    left, right = st.columns([1.3, 1.7])

    with left:
        st.subheader("Predictions")
        st.dataframe(pd.DataFrame({"Patch": range(len(preds)), "Predicted": preds}))
        st.subheader("Class Distribution")
        st.dataframe(class_summary)

    with right:
        st.subheader("Evaluation Metrics")
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)

        c1.metric("Accuracy", round(metrics["accuracy"], 3))
        c2.metric("Precision", round(metrics["precision"], 3))
        c3.metric("Recall", round(metrics["recall"], 3))
        c4.metric("F1-score", round(metrics["f1"], 3))

        st.markdown(f"**Cohen’s Kappa:** {round(metrics['kappa'], 3)}")

        labels = sorted(np.unique(y_test))
        cm_df = pd.DataFrame(metrics["confusion"],
                             index=[f"Actual {l}" for l in labels],
                             columns=[f"Pred {l}" for l in labels])

        st.subheader("Confusion Matrix")
        st.dataframe(cm_df)

# ---------------- Tabs ----------------

with tab_rf:
    st.subheader("Random Forest Model")
    if st.button("Run Random Forest"):
        bar = st.progress(0)
        status = st.empty()
        st.session_state.rf_results = run_pipeline_progress(
            base_dir, areas_csv, "Random Forest", bar, status
        )

    if st.session_state.rf_results:
        render_results(
            st.session_state.rf_results[0],
            st.session_state.rf_results[1],
            st.session_state.rf_results[4],
            st.session_state.rf_results[5]
        )

with tab_lgbm:
    st.subheader("LightGBM Model")
    if st.button("Run LightGBM"):
        bar = st.progress(0)
        status = st.empty()
        st.session_state.lgbm_results = run_pipeline_progress(
            base_dir, areas_csv, "LightGBM", bar, status
        )

    if st.session_state.lgbm_results:
        render_results(
            st.session_state.lgbm_results[0],
            st.session_state.lgbm_results[1],
            st.session_state.lgbm_results[4],
            st.session_state.lgbm_results[5]
        )

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("EarthScape dashboard – dual-model dominant-modality system.")


# In[ ]:




