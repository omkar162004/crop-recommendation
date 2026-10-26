import json

import joblib
import pandas as pd

from app import config

LABELS = {
    "temperature": "temperature", "humidity": "humidity", "rainfall": "rainfall",
    "ph": "soil pH", "N": "nitrogen", "P": "phosphorus", "K": "potassium",
}
UNITS = {"temperature": " °C", "humidity": "%", "rainfall": " mm/month"}


def _fmt(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


class Predictor:
    def __init__(self):
        self.models, self.meta = {}, {}
        for mode in ("basic", "full"):
            model_path = config.MODELS_DIR / f"model_{mode}.joblib"
            meta_path = config.MODELS_DIR / f"metadata_{mode}.json"
            if not model_path.exists() or not meta_path.exists():
                raise FileNotFoundError(
                    f"Missing {model_path.name} / {meta_path.name}. Run `python ml/train.py` first."
                )
            self.models[mode] = joblib.load(model_path)
            self.meta[mode] = json.loads(meta_path.read_text())

        # typical range (10th-90th percentile) of every feature for every crop, from training data
        df = pd.read_csv(config.TRAIN_CSV)
        self.profiles = {}
        for crop, group in df.groupby("label"):
            self.profiles[str(crop)] = {
                col: (group[col].quantile(0.10), group[col].median(), group[col].quantile(0.90))
                for col in self.meta["full"]["features"]
            }

    # --- helpers for the API ---
    def npk_ranges(self) -> dict:
        ranges = self.meta["full"]["feature_ranges"]
        return {k: ranges[k] for k in ("N", "P", "K")}

    def info(self) -> dict:
        return {
            mode: {"model": m["model"], "macro_f1": m["macro_f1"], "features": m["features"]}
            for mode, m in self.meta.items()
        }

    # --- guard ---
    def _out_of_range(self, row: dict, meta: dict) -> list:
        problems = []
        for feature in meta["features"]:
            low, high = meta["feature_ranges"][feature]
            margin = (high - low) * config.RANGE_TOLERANCE
            if not (low - margin) <= row[feature] <= (high + margin):
                problems.append(
                    f"{LABELS[feature]} ({_fmt(row[feature])}) is outside the {_fmt(low)}–{_fmt(high)} "
                    "the model was trained on"
                )
        return problems

    # --- explanation ---
    def _reason(self, crop: str, row: dict, features: list) -> str:
        profile = self.profiles[crop]
        inside, outside = [], []
        for feature in features:
            p10, median, p90 = profile[feature]
            width = max(p90 - p10, 1e-9)
            distance = abs(row[feature] - median) / width
            (inside if p10 <= row[feature] <= p90 else outside).append((distance, feature, p10, p90))

        def describe(item):
            _, feature, p10, p90 = item
            unit = UNITS.get(feature, "")
            return f"{LABELS[feature]} {_fmt(row[feature])}{unit} (usual {_fmt(p10)}–{_fmt(p90)})"

        parts = []
        if inside:
            best = sorted(inside)[:2]
            parts.append("Within its usual range: " + ", ".join(describe(i) for i in best) + ".")
        if outside:
            worst = sorted(outside, reverse=True)[:2]
            parts.append("Outside it: " + ", ".join(describe(i) for i in worst) + ".")
        return " ".join(parts) or "Conditions are close to this crop's training profile."

    # --- main entry ---
    def predict(self, readings: dict, npk: dict = None) -> dict:
        mode = "full" if npk else "basic"
        meta = self.meta[mode]
        row = {**readings, **(npk or {})}

        problems = self._out_of_range(row, meta)
        base = {"model_used": mode, "model": meta["model"], "macro_f1": meta["macro_f1"]}
        if problems:
            return {
                **base, "status": "out_of_range", "predictions": [],
                "message": "No confident match: current conditions fall outside what this model "
                           "was trained on. " + "; ".join(problems) + ".",
            }

        frame = pd.DataFrame([[row[f] for f in meta["features"]]], columns=meta["features"])
        probabilities = self.models[mode].predict_proba(frame)[0]
        classes = meta["classes"]
        top = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)[:3]

        predictions = [
            {"crop": crop, "confidence": round(float(p), 3),
             "reason": self._reason(crop, row, meta["features"])}
            for crop, p in top
        ]
        status, message = "ok", None
        if predictions[0]["confidence"] < config.LOW_CONFIDENCE_THRESHOLD:
            status = "low_confidence"
            message = "These conditions don't strongly match any single crop, so treat these as weak suggestions."
        return {**base, "status": status, "message": message, "predictions": predictions}