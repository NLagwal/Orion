#!/usr/bin/env python3
"""
scripts/eval_merged.py
Evaluate the merged Orion coach model (or any causal LM) on the held-out split.

Uses plain `transformers` + `torch` — no Unsloth, PEFT, or bitsandbytes required
when the adapter has already been merged into a full 16-bit checkpoint.

Metrics:
  1. JSON validity rate
  2. Schema compliance rate
  3. Action accuracy + per-class precision / recall / F1
  4. Rationale non-emptiness
  5. Generation latency (p50 / p95) and tokens/sec

Example:
  python scripts/eval_merged.py \
    --model_dir coach_merged_16bit \
    --eval_file data/synthetic_coaching_eval.jsonl \
    --max_samples 50 \
    --output_file benchmarks/eval_results.json
"""

import argparse
import datetime as dt
import json
import re
import statistics
import sys
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parser = argparse.ArgumentParser(description="Evaluate a merged Orion coach model")
    parser.add_argument(
        "--model_dir",
        default="coach_merged_16bit",
        help="Path to a merged HF checkpoint, or a HF model id",
    )
    parser.add_argument(
        "--tokenizer_dir",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Tokenizer source (merged tokenizer_config.json may be incompatible)",
    )
    parser.add_argument(
        "--eval_file",
        default="data/synthetic_coaching_eval.jsonl",
        help="Path to evaluation JSONL file",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Max samples to run; 0 = all",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=384,
        help="Maximum generated tokens per sample",
    )
    parser.add_argument(
        "--output_file",
        default="benchmarks/eval_results.json",
        help="Path to output JSON results",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Human-readable model label stored in results (default: model_dir)",
    )
    return parser.parse_args()


def extract_json(text):
    """Extract and parse the first valid JSON object from model output."""
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
    """Return (is_valid, list_of_errors) for the coaching-decision schema."""
    if not isinstance(obj, dict):
        return False, ["output is not a JSON object"]
    errors = []

    action = obj.get("action")
    if action not in ACTIONS:
        errors.append(f"invalid action: {action!r}")

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
    """Nearest-rank percentile (pct in 0..100)."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def main():
    args = parse_args()
    eval_path = args.eval_file
    output_path = args.output_file

    samples = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    if not samples:
        sys.exit(f"No samples found in {eval_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=== Orion Merged-Model Evaluation ===")
    print(f"Model:      {args.model_dir}")
    print(f"Tokenizer:  {args.tokenizer_dir}")
    print(f"Eval file:  {eval_path}")
    print(f"Samples:    {len(samples)}")
    print(f"Device:     {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")
    print(f"Dtype:      {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()

    results = []
    valid_json_count = 0
    schema_valid_count = 0
    action_matches = 0
    rationale_nonempty = 0
    gt_dist = Counter()
    pred_dist = Counter()
    confusion = {gt: Counter() for gt in ACTIONS}
    confidences = []
    latencies = []
    token_counts = []

    total = len(samples)
    started = time.perf_counter()

    for idx, sample in enumerate(samples):
        messages = sample["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        gt_content = next(m["content"] for m in messages if m["role"] == "assistant")
        gt_obj = extract_json(gt_content)
        gt_action = gt_obj.get("action", "UNKNOWN") if isinstance(gt_obj, dict) else "UNKNOWN"
        gt_dist[gt_action] += 1

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy, deterministic
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output[0][input_len:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        n_generated = int(generated_ids.shape[0])
        latencies.append(elapsed)
        token_counts.append(n_generated)

        pred_obj = extract_json(response_text)
        is_valid = pred_obj is not None
        schema_ok, schema_errors = validate_schema(pred_obj) if is_valid else (False, ["unparseable JSON"])
        pred_action = pred_obj.get("action", "UNKNOWN") if isinstance(pred_obj, dict) else "PARSE_ERROR"
        rationale = pred_obj.get("rationale", "") if isinstance(pred_obj, dict) else ""
        confidence = pred_obj.get("confidence") if isinstance(pred_obj, dict) else None

        if is_valid:
            valid_json_count += 1
        if schema_ok:
            schema_valid_count += 1
        if isinstance(rationale, str) and rationale.strip():
            rationale_nonempty += 1
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            confidences.append(float(confidence))

        pred_dist[pred_action] += 1
        if gt_action in confusion:
            confusion[gt_action][pred_action] += 1
        match = pred_action == gt_action
        if match:
            action_matches += 1

        results.append(
            {
                "idx": idx,
                "gt_action": gt_action,
                "pred_action": pred_action,
                "match": match,
                "valid_json": is_valid,
                "schema_valid": schema_ok,
                "schema_errors": schema_errors,
                "rationale": rationale,
                "confidence": confidence,
                "latency_s": round(elapsed, 3),
                "generated_tokens": n_generated,
                "raw_response": response_text[:2000],
            }
        )

        if (idx + 1) % 10 == 0 or idx + 1 == total:
            done = idx + 1
            print(
                f"[{done}/{total}] action {action_matches}/{done} "
                f"({action_matches / done * 100:.1f}%) | "
                f"JSON {valid_json_count}/{done} ({valid_json_count / done * 100:.1f}%) | "
                f"schema {schema_valid_count}/{done} ({schema_valid_count / done * 100:.1f}%)"
            )

    wall_time = time.perf_counter() - started
    total_tokens = sum(token_counts)

    per_class = {}
    for action in ACTIONS:
        tp = confusion[action][action]
        support = sum(confusion[action].values())
        predicted = pred_dist.get(action, 0)
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

    summary = {
        "metadata": {
            "label": args.label or args.model_dir,
            "model_dir": args.model_dir,
            "tokenizer_dir": args.tokenizer_dir,
            "eval_file": eval_path,
            "samples": total,
            "device": device,
            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "dtype": str(dtype).replace("torch.", ""),
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "torch_version": torch.__version__,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
        "metrics": {
            "json_validity_rate": round(valid_json_count / total * 100, 2),
            "schema_compliance_rate": round(schema_valid_count / total * 100, 2),
            "schema_compliance_rate_given_valid_json": (
                round(schema_valid_count / valid_json_count * 100, 2) if valid_json_count else 0.0
            ),
            "action_accuracy": round(action_matches / total * 100, 2),
            "rationale_nonempty_rate": round(rationale_nonempty / total * 100, 2),
            "mean_confidence": round(statistics.mean(confidences), 4) if confidences else None,
        },
        "latency": {
            "wall_time_s": round(wall_time, 2),
            "mean_s_per_sample": round(statistics.mean(latencies), 3),
            "p50_s_per_sample": round(percentile(latencies, 50), 3),
            "p95_s_per_sample": round(percentile(latencies, 95), 3),
            "total_generated_tokens": total_tokens,
            "mean_generated_tokens": round(statistics.mean(token_counts), 1),
            "tokens_per_second": round(total_tokens / sum(latencies), 2),
        },
        "distributions": {
            "ground_truth": dict(gt_dist),
            "predicted": dict(pred_dist),
        },
        "per_class": per_class,
        "confusion_matrix": {gt: dict(row) for gt, row in confusion.items()},
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 52)
    print("EVALUATION SUMMARY")
    print("=" * 52)
    for key, value in summary["metrics"].items():
        print(f"{key:<44} {value}")
    print("-" * 52)
    for action, row in per_class.items():
        print(
            f"{action:<20} P={row['precision']:.3f} R={row['recall']:.3f} "
            f"F1={row['f1']:.3f} (n={row['support']})"
        )
    print("-" * 52)
    print(f"Mean / p50 / p95 latency: {summary['latency']['mean_s_per_sample']:.2f}s / "
          f"{summary['latency']['p50_s_per_sample']:.2f}s / {summary['latency']['p95_s_per_sample']:.2f}s")
    print(f"Throughput: {summary['latency']['tokens_per_second']:.2f} tok/s")
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
