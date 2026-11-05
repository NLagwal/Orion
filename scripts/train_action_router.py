#!/usr/bin/env python3
"""
scripts/train_action_router.py
Train and evaluate a lightweight action router on Orion's structured athlete state.

The LLM handles rationale generation, but action selection is a tabular
classification problem. This script measures how much accuracy is available
from the same features with a gradient-boosted tree.

Outputs:
  models/action_router.joblib
  benchmarks/action_router_results.json
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

ACTIONS = [
    "MAINTAIN_PLAN",
    "SWAP_TO_ZONE2",
    "REST_DAY",
    "MODIFY_INTERVALS",
    "ESCALATE_TO_HUMAN",
]

FEATURE_NAMES = [
    "age",
    "max_hr",
    "baseline_5km_s",
    "resting_hr",
    "rhr_delta",
    "sleep_min",
    "sleep_efficiency",
    "deep_sleep_min",
    "rem_sleep_min",
    "wake_min",
    "acwr",
    "acute_load_7d",
    "chronic_load_28d",
    "today_load",
    "readiness",
    "fatigue",
    "soreness",
    "stress",
    "mood",
    "injury_reported",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Orion's action router")
    parser.add_argument("--train_file", default="data/synthetic_coaching_train.jsonl")
    parser.add_argument("--eval_file", default="data/synthetic_coaching_eval.jsonl")
    parser.add_argument("--output_model", default="models/action_router.joblib")
    parser.add_argument("--output_metrics", default="benchmarks/action_router_results.json")
    parser.add_argument("--cv_folds", type=int, default=5)
    return parser.parse_args()


def number(text, pattern):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else np.nan


def extract_features(prompt):
    return [
        number(prompt, r"Age:\s*(\d+)"),
        number(prompt, r"Max HR:\s*(\d+)"),
        number(prompt, r"5km Baseline Fitness:\s*([\d.]+)"),
        number(prompt, r"Resting HR:\s*([\d.]+)"),
        number(prompt, r"7d Trend Deviation:\s*(-?[\d.]+)"),
        number(prompt, r"Sleep:\s*(\d+)\s*min"),
        number(prompt, r"Efficiency:\s*(\d+)%"),
        number(prompt, r"Deep Sleep:\s*(\d+)m"),
        number(prompt, r"REM:\s*(\d+)m"),
        number(prompt, r"Wake:\s*(\d+)m"),
        number(prompt, r"ACWR\s*([\d.]+)"),
        number(prompt, r"Acute 7d Load:\s*([\d.]+)"),
        number(prompt, r"Chronic 28d Load:\s*([\d.]+)"),
        number(prompt, r"Today's Executed Load:\s*([\d.]+)"),
        number(prompt, r"Readiness:\s*([\d.]+)"),
        number(prompt, r"Fatigue:\s*([\d.]+)"),
        number(prompt, r"Soreness:\s*([\d.]+)"),
        number(prompt, r"Stress:\s*([\d.]+)"),
        number(prompt, r"Mood:\s*([\d.]+)"),
        number(prompt, r"Reported Injury/Pain:\s*(\d)"),
    ]


def load_split(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    X = np.array([extract_features(row["messages"][1]["content"]) for row in rows], dtype=float)
    y = np.array([json.loads(row["messages"][-1]["content"])["action"] for row in rows])
    return X, y


def main():
    args = parse_args()
    X_train, y_train = load_split(args.train_file)
    X_eval, y_eval = load_split(args.eval_file)

    model = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=0,
    )

    print(f"Train: {X_train.shape[0]} examples, {X_train.shape[1]} features")
    print(f"Eval:  {X_eval.shape[0]} examples")
    print("Running cross-validation on the training split...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=args.cv_folds, n_jobs=-1)

    print("Fitting final model on the full training split...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_eval)

    accuracy = accuracy_score(y_eval, predictions)
    labels = [action for action in ACTIONS if action in set(y_eval) | set(predictions)]
    report = classification_report(
        y_eval, predictions, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_eval, predictions, labels=labels)

    llm_accuracy = None
    llm_path = Path("benchmarks/eval_results.json")
    if llm_path.exists():
        with open(llm_path, "r", encoding="utf-8") as handle:
            llm_accuracy = json.load(handle).get("metrics", {}).get("action_accuracy")

    result = {
        "metadata": {
            "train_file": args.train_file,
            "eval_file": args.eval_file,
            "model": "HistGradientBoostingClassifier",
            "features": FEATURE_NAMES,
            "train_samples": int(X_train.shape[0]),
            "eval_samples": int(X_eval.shape[0]),
        },
        "metrics": {
            "cv_accuracy_mean": round(float(cv_scores.mean()) * 100, 2),
            "cv_accuracy_std": round(float(cv_scores.std()) * 100, 2),
            "eval_accuracy": round(float(accuracy) * 100, 2),
            "llm_eval_accuracy": llm_accuracy,
            "delta_vs_llm": (
                round(float(accuracy) * 100 - float(llm_accuracy), 2)
                if llm_accuracy is not None
                else None
            ),
        },
        "per_class": {
            action: {
                "precision": round(report[action]["precision"], 4),
                "recall": round(report[action]["recall"], 4),
                "f1": round(report[action]["f1-score"], 4),
                "support": int(report[action]["support"]),
            }
            for action in labels
        },
        "confusion_matrix": {
            label: {predicted: int(matrix[i][j]) for j, predicted in enumerate(labels)}
            for i, label in enumerate(labels)
        },
    }

    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_NAMES}, args.output_model)
    with open(args.output_metrics, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("\n=== ACTION ROUTER ===")
    print(f"CV accuracy:   {result['metrics']['cv_accuracy_mean']:.2f}% "
          f"(± {result['metrics']['cv_accuracy_std']:.2f})")
    print(f"Eval accuracy: {result['metrics']['eval_accuracy']:.2f}%")
    if llm_accuracy is not None:
        print(f"LLM accuracy:  {llm_accuracy:.2f}%")
        print(f"Delta:         {result['metrics']['delta_vs_llm']:+.2f} points")
    print("\nPer-class:")
    print(classification_report(y_eval, predictions, labels=labels, digits=3, zero_division=0))
    print(f"Model saved to {args.output_model}")
    print(f"Metrics saved to {args.output_metrics}")


if __name__ == "__main__":
    main()
