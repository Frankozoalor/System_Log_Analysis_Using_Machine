from __future__ import annotations

import os
import tempfile
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from core.anomaly_detector import detect_anomalies
from core.parser import parse_log_file
from core.preprocessor import preprocess_features


st.set_page_config(page_title=" Log Anomaly Detector", page_icon="📊", layout="wide")
st.title("Log Anomaly Detector")
st.write("Upload an HDFS `.log` or `.csv` file and click **Analyze**.")


@st.cache_data(show_spinner=False)
def evaluate_results(df):
    df = df.copy()

    df["true_label"] = (
        (df["max_level"] >= 2)
        | (df["exception_count"] > 0)
        | (df["error_word_count"] > 0)
    ).astype(int)

    df["predicted_label"] = df["predicted_label"].astype(int)

    accuracy = accuracy_score(df["true_label"], df["predicted_label"])
    precision = precision_score(df["true_label"], df["predicted_label"], zero_division=0)
    recall = recall_score(df["true_label"], df["predicted_label"], zero_division=0)
    f1 = f1_score(df["true_label"], df["predicted_label"], zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        df["true_label"], df["predicted_label"], labels=[0, 1]
    ).ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    report = classification_report(
        df["true_label"],
        df["predicted_label"],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "report": report,
    }


uploaded_file = st.file_uploader("Upload HDFS log file", type=["log", "txt"])

st.sidebar.header("Model Settings")
contamination = st.sidebar.slider(
    "Contamination",
    min_value=0.001,
    max_value=0.05,
    value=0.01,
    step=0.001,
    help="Expected fraction of anomalies in the file.",
)


analyze = st.button("Analyze")


if analyze:
    if uploaded_file is None:
        st.warning("Please upload a file first.")
    else:
        temp_path = None
        try:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".log"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            df_parsed = parse_log_file(temp_path)
            df_features, df_meta = preprocess_features(df_parsed)

            df_anomalies = detect_anomalies(
                df_features,
                df_meta,
                contamination=contamination,
            
            )

            df_predicted = df_anomalies[df_anomalies["predicted_label"] == 1].copy()
            metrics = evaluate_results(df_anomalies)

            print("\n ------  ANOMALY METRICS ------")
            print(f"Accuracy           : {metrics['accuracy']:.4f}")
            print(f"Precision          : {metrics['precision']:.4f}")
            print(f"Recall             : {metrics['recall']:.4f}")
            print(f"F1 Score           : {metrics['f1']:.4f}")
            print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
            print(f"TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}  TP={metrics['tp']}")
            print("\nClassification Report:")
            print(metrics["report"])
            print("-------------------------------\n")

            st.success(
                f"Detected {len(df_predicted)} anomalous blocks out of {len(df_anomalies)} blocks."
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
            c2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
            c3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
            c4.metric("F1 Score", f"{metrics['f1'] * 100:.2f}%")
            c5.metric("FPR", f"{metrics['false_positive_rate'] * 100:.2f}%")

            st.subheader("Anomaly Score Overview")

            plot_df = df_anomalies.copy()
            plot_df["status"] = plot_df["predicted_label"].map({0: "Normal", 1: "Anomaly"})

            rng = np.random.default_rng(42)
            plot_df["x_jitter"] = plot_df["anomaly_probability"] + rng.uniform(-0.01, 0.01, len(plot_df))
            plot_df["x_jitter"] = plot_df["x_jitter"].clip(0, 1)

            hover_cols = []
            for col in [
                "block_id",
                "line_count",
                "unique_templates",
                "unique_components",
                "max_level",
                "warn_error_count",
                "exception_count",
                "error_word_count",
                "anomaly_probability",
            ]:
                if col in plot_df.columns:
                    hover_cols.append(col)

            fig = px.scatter(
                plot_df,
                x="x_jitter", 
                y="anomaly_probability",
                color="status",
                color_discrete_map={
                    "Normal": "green",
                    "Anomaly": "red",
                },
                hover_data=hover_cols,
                size="anomaly_probability",
                size_max=18,
                opacity=0.7,
                title="Scatter plot for Anomaly Scores",
                labels={
                    "x_jitter": "Anomaly score (spread for visibility)",
                    "anomaly_probability": "Normalized anomaly score",
                    "status": "Prediction",
                },
            )

            fig.update_traces(marker=dict(line=dict(width=1)))

            fig.update_layout(
                height=550,
                xaxis_title="Anomaly score (spread for visibility)",
                yaxis_title="Normalized anomaly score",
                legend_title="Prediction",
            )

            st.plotly_chart(fig, use_container_width=True)

            display_cols = [
                "block_id",
                "line_count",
                "unique_templates",
                "unique_components",
                "max_level",
                "warn_error_count",
                "exception_count",
                "error_word_count",
                "anomaly_probability",
                "predicted_label",
            ]
            display_cols = [col for col in display_cols if col in df_anomalies.columns]

            if df_predicted.empty:
                st.info("No anomalies detected with the current settings.")
            else:
                anomalous_block_ids = set(df_predicted["block_id"])
                raw_logs_for_anomalies = df_parsed[df_parsed["block_id"].isin(anomalous_block_ids)].copy()

                st.subheader("Raw Log Lines for Anomalous Blocks")
                st.dataframe(
                    raw_logs_for_anomalies[["block_id", "level", "component", "message"]],
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)