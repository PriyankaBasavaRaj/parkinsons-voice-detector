"""
Train and compare baseline classifiers on the Parkinson's voice dataset.

Uses GroupKFold cross-validation (grouped by patient) throughout, so no
patient's recordings ever appear in both a training fold and its
validation fold. Saves the best pipeline (scaler + model) to models/.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import load_data, get_feature_columns

RANDOM_STATE = 42
N_SPLITS = 5


def build_candidates():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=5,
                                            class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True,
                        class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def evaluate_grouped(pipeline, X, y, groups, n_splits=N_SPLITS):
    """Grouped cross-validated out-of-fold predictions -> honest metrics."""
    gkf = GroupKFold(n_splits=n_splits)
    y_pred = cross_val_predict(pipeline, X, y, groups=groups, cv=gkf, method="predict")
    y_proba = cross_val_predict(pipeline, X, y, groups=groups, cv=gkf, method="predict_proba")[:, 1]

    report = classification_report(y, y_pred, target_names=["healthy", "PD"], output_dict=True)
    auc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)
    return report, auc, cm


def main():
    df = load_data()
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["status"]
    groups = df["patient_id"]

    print(f"Dataset: {len(df)} recordings, {groups.nunique()} patients")
    print(f"Class balance: {y.value_counts().to_dict()} (0=healthy, 1=PD)\n")
    print("=" * 70)
    print(f"Grouped {N_SPLITS}-fold cross-validation (no patient leakage)")
    print("=" * 70)

    results = {}
    for name, pipeline in build_candidates().items():
        report, auc, cm = evaluate_grouped(pipeline, X, y, groups)
        results[name] = {"report": report, "auc": auc, "cm": cm, "pipeline": pipeline}

        print(f"\n--- {name} ---")
        print(f"Accuracy:        {report['accuracy']:.3f}")
        print(f"PD Recall:       {report['PD']['recall']:.3f}  "
              f"(of actual PD cases, % correctly flagged)")
        print(f"PD Precision:    {report['PD']['precision']:.3f}")
        print(f"ROC-AUC:         {auc:.3f}")
        print(f"Confusion matrix [[TN FP],[FN TP]]:\n{cm}")

    # Pick best by PD recall first (missing PD is the costlier error), then AUC
    best_name = max(results, key=lambda n: (results[n]["report"]["PD"]["recall"], results[n]["auc"]))
    print("\n" + "=" * 70)
    print(f"Best model by PD recall: {best_name}")
    print("=" * 70)

    # Refit the winning pipeline on ALL data for the final saved model
    final_pipeline = build_candidates()[best_name]
    final_pipeline.fit(X, y)
    joblib.dump({"pipeline": final_pipeline, "feature_columns": feature_cols}, "models/model.pkl")
    print(f"\nSaved final model (refit on all {len(df)} recordings) to models/model.pkl")
    print("NOTE: the CV metrics above (not training accuracy) are the honest estimate of real performance.")


if __name__ == "__main__":
    main()
