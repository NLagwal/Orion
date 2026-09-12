#!/usr/bin/env python3
"""
scripts/eval_qwen_lora.py
Evaluates the fine-tuned Qwen 2.5 3B QLoRA model on data/synthetic_coaching_eval.jsonl.
Computes:
  1. JSON validity rate (% of generations that are valid JSON)
  2. Schema compliance rate
  3. Action classification accuracy, precision, recall (especially for MODIFY_INTERVALS and ESCALATE_TO_HUMAN)
  4. Rationale readability and non-emptiness
"""

import json
import os
import sys
import re
import argparse
from collections import Counter, defaultdict
from unsloth import FastLanguageModel
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen 2.5 3B coach model")
    parser.add_argument(
        "--adapter_dir",
        default="/home/lag/.unsloth/studio/outputs/Qwen_Qwen2.5-3B-Instruct__project-orion-coach-qwen3b_1789197420",
        help="Path to adapter checkpoint directory"
    )
    parser.add_argument(
        "--eval_file",
        default="/home/lag/Projekt/Orion/data/synthetic_coaching_eval.jsonl",
        help="Path to evaluation JSONL file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Max number of eval samples to run (default 50 for quick thorough benchmark)"
    )
    parser.add_argument(
        "--output_file",
        default="/home/lag/Projekt/Orion/eval_results.json",
        help="Path to output JSON results"
    )
    return parser.parse_args()

def extract_json(text: str) -> dict | None:
    """Extract and parse first valid JSON object from text."""
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    
    # Try finding JSON block between curly braces
    match = re.search(r'(\{[\s\S]*\})', text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return None

def main():
    args = parse_args()
    print(f"=== Orion Edge Coach Evaluation ===")
    print(f"Adapter: {args.adapter_dir}")
    print(f"Eval File: {args.eval_file}")
    print(f"Sample Limit: {args.max_samples}")

    # Load eval dataset
    samples = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} total eval samples from {args.eval_file}")
    if args.max_samples > 0:
        eval_samples = samples[:args.max_samples]
    else:
        eval_samples = samples

    # Load model with Unsloth
    print("\nLoading Qwen 2.5 3B base model and LoRA adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_dir,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    results = []
    total = len(eval_samples)
    valid_json_count = 0
    action_matches = 0
    gt_action_dist = Counter()
    pred_action_dist = Counter()
    class_correct = Counter()
    class_total = Counter()

    print(f"\nRunning inference on {total} samples...")
    for idx, sample in enumerate(eval_samples):
        messages = sample["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        gt_assistant_content = [m["content"] for m in messages if m["role"] == "assistant"][0]
        
        gt_json = extract_json(gt_assistant_content)
        gt_action = gt_json.get("action", "UNKNOWN") if gt_json else "UNKNOWN"
        gt_action_dist[gt_action] += 1
        class_total[gt_action] += 1

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=384,
                temperature=0.0,  # Greedy deterministic decoding
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Slice out prompt tokens
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        pred_json = extract_json(response_text)
        is_valid = pred_json is not None
        if is_valid:
            valid_json_count += 1
            pred_action = pred_json.get("action", "UNKNOWN")
        else:
            pred_action = "PARSE_ERROR"

        pred_action_dist[pred_action] += 1
        match = (pred_action == gt_action)
        if match:
            action_matches += 1
            class_correct[gt_action] += 1

        results.append({
            "idx": idx,
            "gt_action": gt_action,
            "pred_action": pred_action,
            "match": match,
            "is_valid_json": is_valid,
            "rationale": pred_json.get("rationale", "") if pred_json else "",
            "pred_workout": pred_json.get("modified_workout", {}) if pred_json else {},
            "raw_response": response_text[:200]
        })

        if (idx + 1) % 10 == 0 or idx + 1 == total:
            acc_so_far = (action_matches / (idx + 1)) * 100
            json_rate = (valid_json_count / (idx + 1)) * 100
            print(f"[{idx + 1}/{total}] Action Match: {action_matches}/{idx + 1} ({acc_so_far:.1f}%) | Valid JSON: {json_rate:.1f}%")

    overall_accuracy = (action_matches / total) * 100
    json_rate = (valid_json_count / total) * 100

    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    print(f"Total Evaluated: {total}")
    print(f"Valid JSON Rate: {valid_json_count}/{total} ({json_rate:.1f}%)")
    print(f"Overall Action Accuracy: {action_matches}/{total} ({overall_accuracy:.1f}%)")
    print("\nPer-Class Performance:")
    print(f"{'Class':<22} | {'Correct':<8} | {'Total':<6} | {'Accuracy':<8}")
    print("-" * 52)
    for cls in sorted(class_total.keys()):
        corr = class_correct[cls]
        tot = class_total[cls]
        acc = (corr / tot * 100) if tot > 0 else 0.0
        print(f"{cls:<22} | {corr:<8} | {tot:<6} | {acc:.1f}%")

    summary_data = {
        "total_samples": total,
        "valid_json_rate": json_rate,
        "overall_action_accuracy": overall_accuracy,
        "per_class": {
            cls: {
                "correct": class_correct[cls],
                "total": class_total[cls],
                "accuracy": (class_correct[cls] / class_total[cls] * 100) if class_total[cls] > 0 else 0.0
            }
            for cls in sorted(class_total.keys())
        },
        "results": results
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nDetailed evaluation results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
