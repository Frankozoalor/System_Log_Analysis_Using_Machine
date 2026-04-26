from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_anomalies(
    df_features: pd.DataFrame,
    df_meta: pd.DataFrame,
    contamination: float = 0.01,
    prediction_quantile: float = 0.965,
    random_state: int = 42,
 ) -> pd.DataFrame:
    X = df_features.copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = float(np.clip(contamination, 0.001, 0.1))
    prediction_quantile = float(np.clip(prediction_quantile, 0.90, 0.9999))

    model = IsolationForest(
        n_estimators=400,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    raw_score = -model.score_samples(X_scaled)  # bigger = more anomalous

    df_result = df_meta.copy()
    df_result["anomaly_score"] = raw_score

    score_min = float(df_result["anomaly_score"].min())
    score_max = float(df_result["anomaly_score"].max())

    if score_max > score_min:
        df_result["anomaly_probability"] = (
            (df_result["anomaly_score"] - score_min) / (score_max - score_min)
        )
    else:
        df_result["anomaly_probability"] = 0.0

 
    threshold = df_result["anomaly_score"].quantile(prediction_quantile)


    suspicious_context = (
        (df_result["max_level"] >= 3)
        | (df_result["exception_count"] > 0)
        | (df_result["error_word_count"] > 0)
        | (df_result["missing_receive_after_allocate"] == 1)
        | (df_result["termination_without_success"] == 1)
    )

    df_result["predicted_label"] = (
        (df_result["anomaly_score"] >= threshold) & suspicious_context
    ).astype(int)

    df_result["strong_anomaly"] = df_result["predicted_label"]
    df_result["prediction_threshold"] = threshold

    return df_result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)