"""
Retrain using only the subset of features we can realistically compute
client-side in the browser via the Web Audio API:

  MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)   - pitch stats (autocorrelation)
  MDVP:Jitter(%)                             - cycle-to-cycle F0 variation
  MDVP:Shimmer                               - cycle-to-cycle amplitude variation
  HNR                                        - harmonic-to-noise ratio

Dropped: RAP/PPQ/DDP/APQ/DDA (higher-order jitter/shimmer variants) and
RPDE/DFA/D2/spread1/spread2/PPE (nonlinear dynamics), these need
Praat-grade DSP (parselmouth) that isn't practical to reproduce in plain
JS in a browser tab.

This is a real accuracy tradeoff, measured honestly below via grouped CV,
in exchange for a fully static, backend-free app.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import load_data

RANDOM_STATE = 42
N_SPLITS = 5

BROWSER_FEATURES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Shimmer",
    "HNR",
]


def build_candidates():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=4,
                                            class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def evaluate_grouped(pipeline, X, y, groups, n_splits=N_SPLITS):
    gkf = GroupKFold(n_splits=n_splits)
    y_pred = cross_val_predict(pipeline, X, y, groups=groups, cv=gkf, method="predict")
    y_proba = cross_val_predict(pipeline, X, y, groups=groups, cv=gkf, method="predict_proba")[:, 1]
    report = classification_report(y, y_pred, target_names=["healthy", "PD"], output_dict=True)
    auc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)
    return report, auc, cm


def main():
    df = load_data()
    X = df[BROWSER_FEATURES]
    y = df["status"]
    groups = df["patient_id"]

    print(f"Using reduced browser-computable feature set: {BROWSER_FEATURES}\n")
    print("=" * 70)
    print(f"Grouped {N_SPLITS}-fold cross-validation (no patient leakage)")
    print("=" * 70)

    results = {}
    for name, pipeline in build_candidates().items():
        report, auc, cm = evaluate_grouped(pipeline, X, y, groups)
        results[name] = {"report": report, "auc": auc, "pipeline": pipeline}
        print(f"\n--- {name} ---")
        print(f"Accuracy:     {report['accuracy']:.3f}")
        print(f"PD Recall:    {report['PD']['recall']:.3f}")
        print(f"PD Precision: {report['PD']['precision']:.3f}")
        print(f"ROC-AUC:      {auc:.3f}")
        print(f"Confusion matrix [[TN FP],[FN TP]]:\n{cm}")

    best_name = max(results, key=lambda n: (results[n]["report"]["PD"]["recall"], results[n]["auc"]))
    print(f"\nBest model on reduced feature set: {best_name}")

    final_pipeline = build_candidates()[best_name]
    final_pipeline.fit(X, y)
    joblib.dump(
        {"pipeline": final_pipeline, "feature_columns": BROWSER_FEATURES, "model_name": best_name},
        "models/model_browser.pkl",
    )
    print("Saved to models/model_browser.pkl")


if __name__ == "__main__":
    main()
