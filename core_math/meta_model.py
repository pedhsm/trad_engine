"""Frozen linear/logistic model application from FROZEN NUMBERS, with no sklearn dependency.

The idea: train a classifier once (offline, in your research code), then freeze it into
plain data -- coefficients + intercept + scaler mean/scale + the log1p column list +
feature order + a decision threshold. Applying it from those numbers is deterministic
and version-independent: a pickled sklearn model is fragile across sklearn versions, but
~30 floats and a sigmoid are not.

This is what lets a separate process (a live engine, an independent verifier) reproduce
the EXACT same probability -- hence the exact same decision fingerprint -- as the trainer,
without importing sklearn or trusting a pickle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_meta_model(features_df: pd.DataFrame, model: dict) -> pd.Series:
    """Frozen logistic predict_proba[:, 1] over ``features_df``.

    ``model`` carries: feature_order (column names, in training order), log1p_cols
    (the subset log1p-transformed before scaling), scaler_mean / scaler_scale (the
    StandardScaler stats), coef, intercept, decision_threshold. Mirrors a standard
    sklearn pipeline exactly: select in feature_order -> log1p the heavy-tailed
    columns -> standardize -> sigmoid(X @ coef + intercept). NaN in any feature
    propagates to a NaN probability (the caller's threshold then rejects it),
    exactly as an unusable row should be dropped rather than silently acted on.
    """
    X = features_df[model["feature_order"]].astype(float).copy()
    for c in model["log1p_cols"]:
        X[c] = np.log1p(X[c])
    mean = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    Xs = (X.to_numpy() - mean) / scale
    z = Xs @ np.asarray(model["coef"], dtype=float) + float(model["intercept"])
    proba = 1.0 / (1.0 + np.exp(-z))
    return pd.Series(proba, index=features_df.index)
