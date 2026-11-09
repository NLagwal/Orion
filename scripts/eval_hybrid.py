#!/usr/bin/env python3
"""
scripts/eval_hybrid.py
End-to-end evaluation of Orion's hybrid decision path:

  router (GBM) -> guardrails -> LLM rationale/JSON

The action is selected by the router and may be overridden by deterministic
guardrails. The LLM is asked to use that action and write the rationale/workout.
The LLM's raw action is still recorded so you can see whether it agreed.

Run:
  # terminal 1
  llama-server -m coach.gguf --host 127.0.0.1 --port 8082 -ngl 99 -c 2048
  # terminal 2
  python3 scripts/eval_hybrid.py --server_url http://127.0.0.1:8082 --max_samples 0
"""

import argparse
import datetime as dt
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.guardrails import apply_guardrails
from scripts.train_action_router import extract_features

ACTIONS = [
    "MAINTAIN_PLAN",
    "SWAP_TO_ZONE2",
    "REST_DAY",
    "MODIFY_INTERVALS",
    "ESCALATE_TO_HUMAN",
]
REASON_CATEGORIES = {
    "illness_pattern",
    "injury_signal",
    "overtraining_risk",
    "data_conflict_unresolvable",
    "none",
}
TRIGGER_SOURCES = {"model", "guardrail"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Orion's hybrid decision path")
    parser.add_argument("--eval_file", default="data/synthetic_coaching_eval.jsonl")
    parser.add_argument("--model_path", default="models/action_router.joblib")
    parser.add_argument("--server_url", default="http://127.0.0.1:8080")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output_file", default="benchmarks/hybrid_results.json")
    return parser.parse_args()


def extract_json(text):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None


def validate_schema(obj):
    if not isinstance(obj, dict):
        return False, ["output is not a JSON object"]
    errors = []

    if obj.get("action") not in ACTIONS:
        errors.append(f"invalid action: {obj.get('action')!r}")

    rationale = obj.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("rationale missing or empty")

    workout = obj.get("modified_workout")
    if not isinstance(workout, dict):
        errors.append("modified_workout missing or not an object")
    else:
        if not isinstance(workout.get("name"), str) or not workout["name"].strip():
            errors.append("modified_workout.name missing or empty")
        if not isinstance(workout.get("detail"), str) or not workout["detail"].strip():
            errors.append("modified_workout.detail missing or empty")
        duration = workout.get("duration_minutes")
        if isinstance(duration, bool) or not isinstance(duration, int):
            errors.append("modified_workout.duration_minutes is not an integer")
        for optional in ("target_pace", "target_zone"):
            if optional in workout and workout[optional] is not None and not isinstance(workout[optional], str):
                errors.append(f"modified_workout.{optional} is not a string or null")

    confidence = obj.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        errors.append("confidence missing or not numeric")
    elif not 0.0 <= float(confidence) <= 1.0:
        errors.append("confidence outside 0..1")

    escalation = obj.get("escalation")
    if not isinstance(escalation, dict):
        errors.append("escalation missing or not an object")
    else:
        if escalation.get("reason_category") not in REASON_CATEGORIES:
            errors.append(f"invalid escalation.reason_category: {escalation.get('reason_category')!r}")
        if escalation.get("trigger_source") not in TRIGGER_SOURCES:
            errors.append(f"invalid escalation.trigger_source: {escalation.get('trigger_source')!r}")

    return not errors, errors


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def get_model_id(base_url):
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=10)
        response.raise_for_status()
        models = response.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return "coach.gguf"


def build_messages(row, final_action):
    messages = [
        {"role": message["role"], "content": message["content"]}
        for message in row["messages"]
        if message["role"] != "assistant"
    ]
    instruction = (
        f"\n\nDETERMINISTIC ROUTER DECISION: {final_action}\n"
        "Return exactly one JSON object matching the schema above. "
        f'The action field MUST be "{final_action}". '
        "Write a rationale and workout consistent with that action."
    )
    for message in reversed(messages):
        if message["role"] == "user":
            message["content"] += instruction
            break
    return messages


def call_llm(base_url, model_id, messages, max_tokens, timeout):
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    return elapsed, body["choices"][0]["message"]["content"], body


def main():
    args = parse_args()
    base_url = args.server_url.rstrip("/")

    rows = []
    with open(args.eval_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        sys.exit(f"No samples found in {args.eval_file}")

    artifact = joblib.load(args.model_path)
    model = artifact["model"]
    feature_names = artifact["features"]
    model_id = get_model_id(base_url)

    print("=== Orion Hybrid Evaluation ===")
    print(f"Router:  {args.model_path}")
    print(f"Server:  {base_url} ({model_id})")
    print(f"Samples: {len(rows)}")

    results = []
    latencies = []
    completion_tokens = []
    gt_dist = Counter()
    router_dist = Counter()
    final_dist = Counter()
    llm_raw_dist = Counter()
    confusion = {action: Counter() for action in ACTIONS}
    overrides = 0
    json_valid = 0
    llm_schema_valid = 0
    hybrid_schema_valid = 0
    rationale_nonempty = 0
    router_correct = 0
    llm_correct = 0
    final_correct = 0

    for index, row in enumerate(rows):
        prompt = row["messages"][1]["content"]
        gt_obj = extract_json(row["messages"][-1]["content"])
        gt_action = gt_obj.get("action", "UNKNOWN") if isinstance(gt_obj, dict) else "UNKNOWN"
        gt_dist[gt_action] += 1

        values = extract_features(prompt)
        state = {
            name: (None if np.isnan(value) else float(value))
            for name, value in zip(feature_names, values)
        }
        vector = np.array(
            [[np.nan if value is None else value for value in state.values()]],
            dtype=float,
        )

        router_action = str(model.predict(vector)[0])
        final_guardrail = apply_guardrails(state, router_action)
        final_action = final_guardrail.action
        router_dist[router_action] += 1
        final_dist[final_action] += 1
        if final_guardrail.overridden:
            overrides += 1

        messages = build_messages(row, final_action)
        elapsed, raw_response, body = call_llm(
            base_url,
            model_id,
            messages,
            args.max_new_tokens,
            args.timeout,
        )
        latencies.append(elapsed)
        usage = body.get("usage", {}) or {}
        if isinstance(usage.get("completion_tokens"), int):
            completion_tokens.append(usage["completion_tokens"])

        pred_obj = extract_json(raw_response)
        is_valid = pred_obj is not None
        if is_valid:
            json_valid += 1
            llm_action_raw = pred_obj.get("action", "UNKNOWN")
            schema_ok, schema_errors = validate_schema(pred_obj)
        else:
            llm_action_raw = "PARSE_ERROR"
            schema_ok, schema_errors = False, ["unparseable JSON"]

        rationale = pred_obj.get("rationale", "") if isinstance(pred_obj, dict) else ""
        if isinstance(rationale, str) and rationale.strip():
            rationale_nonempty += 1
        if schema_ok:
            llm_schema_valid += 1

        # The hybrid output is the LLM object with the action forced to the
        # router/guardrail decision. Measure that schema separately.
        if isinstance(pred_obj, dict):
            hybrid_obj = dict(pred_obj)
            hybrid_obj["action"] = final_action
            hybrid_ok, _ = validate_schema(hybrid_obj)
        else:
            hybrid_ok = False
        if hybrid_ok:
            hybrid_schema_valid += 1

        llm_raw_dist[llm_action_raw] += 1
        if gt_action in confusion:
            confusion[gt_action][final_action] += 1

        router_match = router_action == gt_action
        llm_match = llm_action_raw == gt_action
        final_match = final_action == gt_action
        router_correct += int(router_match)
        llm_correct += int(llm_match)
        final_correct += int(final_match)

        results.append(
            {
                "idx": index,
                "gt_action": gt_action,
                "router_action": router_action,
                "guardrail_action": final_action,
                "guardrail_overridden": final_guardrail.overridden,
                "guardrail_reason": final_guardrail.reason,
                "llm_action_raw": llm_action_raw,
                "router_match": router_match,
                "llm_match": llm_match,
                "match": final_match,
                "json_valid": is_valid,
                "llm_schema_valid": schema_ok,
                "hybrid_schema_valid": hybrid_ok,
                "schema_errors": schema_errors,
                "rationale": rationale,
                "latency_s": round(elapsed, 3),
                "raw_response": raw_response[:2000],
            }
        )

        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(
                f"[{index + 1}/{len(rows)}] "
                f"router {router_correct}/{index + 1} | "
                f"hybrid {final_correct}/{index + 1} | "
                f"JSON {json_valid}/{index + 1} | "
                f"overrides {overrides}"
            )

    total = len(rows)
    per_class = {}
    for action in ACTIONS:
        tp = confusion[action][action]
        support = sum(confusion[action].values())
        predicted = final_dist.get(action, 0)
        fp = predicted - tp
        fn = support - tp
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[action] = {
            "support": support,
            "predicted": predicted,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    llm_full_baseline = None
    baseline_path = ROOT / "benchmarks" / "eval_results.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as handle:
            llm_full_baseline = json.load(handle).get("metrics", {}).get("action_accuracy")

    summary = {
        "metadata": {
            "eval_file": args.eval_file,
            "router_model": args.model_path,
            "server_url": base_url,
            "model_id": model_id,
            "samples": total,
            "decoding": "greedy (temperature=0)",
            "max_new_tokens": args.max_new_tokens,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
        "metrics": {
            "router_action_accuracy": round(router_correct / total * 100, 2),
            "hybrid_action_accuracy": round(final_correct / total * 100, 2),
            "llm_raw_action_accuracy_conditioned_on_router": round(llm_correct / total * 100, 2),
            "llm_full_eval_action_accuracy": llm_full_baseline,
            "guardrail_override_rate": round(overrides / total * 100, 2),
            "json_validity_rate": round(json_valid / total * 100, 2),
            "llm_schema_compliance_rate": round(llm_schema_valid / total * 100, 2),
            "hybrid_schema_compliance_rate": round(hybrid_schema_valid / total * 100, 2),
            "rationale_nonempty_rate": round(rationale_nonempty / total * 100, 2),
        },
        "distributions": {
            "ground_truth": dict(gt_dist),
            "router": dict(router_dist),
            "guardrail_final": dict(final_dist),
            "llm_raw": dict(llm_raw_dist),
        },
        "per_class": per_class,
        "confusion_matrix": {gt: dict(row) for gt, row in confusion.items()},
        "latency": {
            "mean_s": round(statistics.mean(latencies), 3),
            "p50_s": round(percentile(latencies, 50), 3),
            "p95_s": round(percentile(latencies, 95), 3),
            "mean_completion_tokens": (
                round(statistics.mean(completion_tokens), 1) if completion_tokens else None
            ),
            "client_tokens_per_second": (
                round(sum(completion_tokens) / sum(latencies), 2)
                if completion_tokens
                else None
            ),
        },
        "results": results,
    }

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 56)
    print("HYBRID SUMMARY")
    print("=" * 56)
    print(f"Router accuracy:   {summary['metrics']['router_action_accuracy']:.2f}%")
    print(f"Hybrid accuracy:   {summary['metrics']['hybrid_action_accuracy']:.2f}%")
    print(f"LLM raw (router-forced): {summary['metrics']['llm_raw_action_accuracy_conditioned_on_router']:.2f}%")
    if summary["metrics"]["llm_full_eval_action_accuracy"] is not None:
        print(f"LLM-only (independent):  {summary['metrics']['llm_full_eval_action_accuracy']:.2f}%")
    print(f"Guardrail overrides: {summary['metrics']['guardrail_override_rate']:.2f}%")
    print(f"JSON validity:     {summary['metrics']['json_validity_rate']:.2f}%")
    print(f"LLM schema:        {summary['metrics']['llm_schema_compliance_rate']:.2f}%")
    print(f"Hybrid schema:     {summary['metrics']['hybrid_schema_compliance_rate']:.2f}%")
    print(f"Latency p50/p95:   {summary['latency']['p50_s']:.2f}s / {summary['latency']['p95_s']:.2f}s")
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
