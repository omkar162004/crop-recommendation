import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parents[1]  # the backend/ folder
DATA = BASE / "data" / "processed" / "train.csv"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

SEED = 42
FEATURE_SETS = {
    "basic": ["temperature", "humidity", "rainfall", "ph"],  # live API values only
    "full": ["N", "P", "K", "temperature", "humidity", "rainfall", "ph"],  # plus user-entered NPK
}

# --- MLflow setup: local SQLite database (the old ./mlruns folder store is no longer supported) ---
EXPERIMENT = "crop-recommendation"
mlflow.set_tracking_uri(f"sqlite:///{(BASE / 'mlflow.db').as_posix()}")

# keep artifacts (like the per-class report) inside backend/, whatever folder you run from
if mlflow.get_experiment_by_name(EXPERIMENT) is None:
    mlflow.create_experiment(EXPERIMENT, artifact_location=(BASE / "mlartifacts").as_uri())
mlflow.set_experiment(EXPERIMENT)

# --- Data ---
df = pd.read_csv(DATA)
le = LabelEncoder()
y = le.fit_transform(df["label"])

# --- Train both feature sets, two models each ---
for set_name, features in FEATURE_SETS.items():
    X = df[features]

    # stratified split keeps all 22 crops in both the train and test sets
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, random_state=SEED
        ),
    }

    best_name, best_f1, best_model = None, -1.0, None

    for name, model in candidates.items():
        with mlflow.start_run(run_name=f"{set_name}-{name}"):
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            acc = accuracy_score(y_te, pred)
            f1 = f1_score(y_te, pred, average="macro")

            mlflow.log_param("feature_set", set_name)
            mlflow.log_param("model", name)
            mlflow.log_param("features", ",".join(features))
            mlflow.log_param("seed", SEED)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("macro_f1", f1)
            mlflow.log_text(
                classification_report(y_te, pred, target_names=le.classes_),
                "per_class_report.txt",
            )

            print(f"[{set_name}] {name}: accuracy={acc:.4f}  macro_f1={f1:.4f}")

            if f1 > best_f1:
                best_name, best_f1, best_model = name, f1, model

    # save the winner for this feature set, plus what the API and guard will need
    joblib.dump(best_model, MODELS / f"model_{set_name}.joblib")
    metadata = {
        "feature_set": set_name,
        "model": best_name,
        "macro_f1": round(best_f1, 4),
        "features": features,
        "classes": [str(c) for c in le.classes_],
        "feature_ranges": {
            col: [float(df[col].min()), float(df[col].max())] for col in features
        },
    }
    with open(MODELS / f"metadata_{set_name}.json", "w") as out_file:
        json.dump(metadata, out_file, indent=2)
    print(f"Saved {set_name} model ({best_name})\n")