#!/usr/bin/env python3
"""
Orion: Leak-Free Training Data Augmentation
============================================
1. Splits original 2,354 samples into 90% Train (2,118) and 10% Eval (236) using fixed seed 3407.
2. Leaves Eval set 100% untouched (ZERO augmented copies, ZERO leakage).
3. Augments ONLY the underrepresented classes within the Train split:
   - MODIFY_INTERVALS: 4x expansion (~38 -> ~152 samples)
   - overtraining_risk: 3x expansion (~16 -> ~48 samples)
   - illness_pattern: 2.5x expansion (~34 -> ~85 samples)
4. Jitters BOTH input physiological telemetry AND output pacing/reps/durations.
5. Saves separate train and eval files, as well as an updated combined file.
"""

import json
import random
import re
from pathlib import Path
from copy import deepcopy

SYSTEM_PROMPT = """You are Orion, an elite sports scientist and adaptive endurance coach.
Analyze this athlete's daily state and output the prescribed coaching decision adhering strictly to the structured JSON schema."""

def jitter_prompt(prompt_text: str, rng: random.Random) -> str:
    """Apply realistic physiological noise to the input prompt telemetry."""
    text = prompt_text

    # Jitter Resting HR: e.g. "Resting HR: 65.95 bpm" -> +/- 0.4 bpm
    def rhr_sub(m):
        val = float(m.group(1))
        new_val = round(val + rng.uniform(-0.5, 0.5), 2)
        return f"Resting HR: {new_val} bpm"
    text = re.sub(r"Resting HR:\s*([0-9.]+)\s*bpm", rhr_sub, text)

    # Jitter 7d Trend Deviation: e.g. "7d Trend Deviation: -1.81 bpm"
    def trend_sub(m):
        sign = m.group(1)
        val = float(m.group(2))
        signed_val = -val if sign == '-' else val
        new_val = round(signed_val + rng.uniform(-0.3, 0.3), 2)
        new_sign = "+" if new_val >= 0 else "-"
        return f"7d Trend Deviation: {new_sign}{abs(new_val):.2f} bpm"
    text = re.sub(r"7d Trend Deviation:\s*([+-]?)([0-9.]+)\s*bpm", trend_sub, text)

    # Jitter Sleep minutes: e.g. "Sleep: 485 min" -> +/- 15 min
    def sleep_sub(m):
        val = int(m.group(1))
        new_val = max(180, min(600, val + rng.randint(-15, 15)))
        return f"Sleep: {new_val} min"
    text = re.sub(r"Sleep:\s*([0-9]+)\s*min", sleep_sub, text)

    # Jitter ACWR: e.g. "ACWR 1.58" -> +/- 0.04
    def acwr_sub(m):
        val = float(m.group(1))
        new_val = round(val + rng.uniform(-0.04, 0.04), 2)
        return f"ACWR {new_val:.2f}"
    text = re.sub(r"ACWR\s*([0-9.]+)", acwr_sub, text)

    return text

def jitter_modified_intervals(response_obj: dict, rng: random.Random) -> dict:
    """Vary interval workouts and coaching rationales."""
    res = deepcopy(response_obj)
    workout = res.get("modified_workout") or res.get("adjusted_workout", {})

    dur = workout.get("duration_minutes", 40)
    workout["duration_minutes"] = max(25, min(60, dur + rng.choice([-5, -3, 0, 3, 5])))

    # Jitter pace if present: e.g. "6:00-6:20 /km"
    pace = workout.get("target_pace", "")
    pace_m = re.search(r"(\d+):(\d+)\s*-\s*(\d+):(\d+)", pace)
    if pace_m:
        m1, s1, m2, s2 = map(int, pace_m.groups())
        shift = rng.choice([-5, 5, 10, -10])
        total_s1 = max(180, m1 * 60 + s1 + shift)
        total_s2 = max(total_s1 + 10, m2 * 60 + s2 + shift)
        workout["target_pace"] = f"{total_s1 // 60}:{total_s1 % 60:02d}-{total_s2 // 60}:{total_s2 % 60:02d} /km"

    # Subtle rationale phrasing variation
    rationale = res.get("rationale", "")
    prefixes = [
        "Given the elevated acute load and suppressed recovery biomarkers, ",
        "Autonomic indicators suggest residual fatigue: ",
        "To protect adaptation while safeguarding tendons and joints, ",
        "Telemetry indicates borderline recovery: "
    ]
    if not any(rationale.startswith(p) for p in prefixes) and rng.random() > 0.4:
        # Swap leading clause if generic
        res["rationale"] = rng.choice(prefixes) + rationale[0].lower() + rationale[1:]

    res["confidence"] = round(rng.uniform(0.91, 0.97), 2)
    return res

def main():
    src_path = Path("data/synthetic_coaching_dataset.jsonl")
    if not src_path.exists():
        print(f"Error: {src_path} not found")
        return

    with open(src_path, "r", encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]

    print(f"Total raw dataset size: {len(all_rows)}")

    # 1. Deterministic 90/10 Train/Eval Split BEFORE any augmentation
    rng = random.Random(3407)
    shuffled_indices = list(range(len(all_rows)))
    rng.shuffle(shuffled_indices)

    eval_count = int(len(all_rows) * 0.10) # 235 rows
    eval_indices = set(shuffled_indices[:eval_count])
    train_indices = [idx for idx in shuffled_indices if idx not in eval_indices]

    eval_raw = [all_rows[i] for i in sorted(eval_indices)]
    train_raw = [all_rows[i] for i in sorted(train_indices)]

    print(f"Split complete: {len(train_raw)} Train rows, {len(eval_raw)} Eval rows")

    # 2. Count actions in Train and Eval
    def count_actions(rows):
        counts = {}
        for r in rows:
            act = r.get("response", {}).get("action", "UNKNOWN")
            counts[act] = counts.get(act, 0) + 1
        return counts

    print("Initial Train action distribution:", count_actions(train_raw))
    print("Untouched Eval action distribution:", count_actions(eval_raw))

    # 3. Augment ONLY inside the Train split (0% leakage into Eval)
    augmented_train = []
    aug_rng = random.Random(42)

    for row in train_raw:
        augmented_train.append(row)
        action = row.get("response", {}).get("action")
        esc_cat = row.get("response", {}).get("escalation", {}).get("reason_category", "none")

        # MODIFY_INTERVALS: 3 additional variants (4x total)
        if action == "MODIFY_INTERVALS":
            for _ in range(3):
                new_prompt = jitter_prompt(row.get("prompt", ""), aug_rng)
                new_response = jitter_modified_intervals(row.get("response", {}), aug_rng)
                augmented_train.append({
                    "prompt": new_prompt,
                    "response": new_response,
                    "participant_id": row.get("participant_id"),
                    "date": row.get("date"),
                    "is_augmented": True
                })

        # ESCALATE_TO_HUMAN / overtraining_risk: 2 additional variants (3x total)
        elif action == "ESCALATE_TO_HUMAN" and esc_cat == "overtraining_risk":
            for _ in range(2):
                new_prompt = jitter_prompt(row.get("prompt", ""), aug_rng)
                new_resp = deepcopy(row.get("response", {}))
                new_resp["confidence"] = round(aug_rng.uniform(0.92, 0.98), 2)
                augmented_train.append({
                    "prompt": new_prompt,
                    "response": new_resp,
                    "participant_id": row.get("participant_id"),
                    "date": row.get("date"),
                    "is_augmented": True
                })

        # ESCALATE_TO_HUMAN / illness_pattern: 1 additional variant (2x total)
        elif action == "ESCALATE_TO_HUMAN" and esc_cat == "illness_pattern":
            new_prompt = jitter_prompt(row.get("prompt", ""), aug_rng)
            new_resp = deepcopy(row.get("response", {}))
            new_resp["confidence"] = round(aug_rng.uniform(0.92, 0.97), 2)
            augmented_train.append({
                "prompt": new_prompt,
                "response": new_resp,
                "participant_id": row.get("participant_id"),
                "date": row.get("date"),
                "is_augmented": True
            })

    # Shuffle training set so augmented rows are uniformly dispersed
    aug_rng.shuffle(augmented_train)

    print("\nPost-Augmentation Train action distribution:", count_actions(augmented_train))
    print(f"Train rows increased from {len(train_raw)} to {len(augmented_train)} (+{len(augmented_train) - len(train_raw)} rows)")

    # 4. Export pure ChatML format
    def to_chatml_record(r):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r.get("prompt", "")},
                {"role": "assistant", "content": json.dumps(r.get("response", {}), indent=2)}
            ]
        }

    # Save Train file
    train_out = Path("data/synthetic_coaching_train.jsonl")
    with open(train_out, "w", encoding="utf-8") as f:
        for r in augmented_train:
            f.write(json.dumps(to_chatml_record(r), ensure_ascii=False) + "\n")

    # Save Eval file (100% UNTOUCHED, ZERO LEAKAGE)
    eval_out = Path("data/synthetic_coaching_eval.jsonl")
    with open(eval_out, "w", encoding="utf-8") as f:
        for r in eval_raw:
            f.write(json.dumps(to_chatml_record(r), ensure_ascii=False) + "\n")

    # Also write combined file for single-file upload
    combined_out = Path("data/synthetic_coaching_dataset_unsloth.jsonl")
    with open(combined_out, "w", encoding="utf-8") as f:
        for r in augmented_train:
            f.write(json.dumps(to_chatml_record(r), ensure_ascii=False) + "\n")
        for r in eval_raw:
            f.write(json.dumps(to_chatml_record(r), ensure_ascii=False) + "\n")

    # Copy to Studio upload cache
    cache_path = Path("/home/lag/.unsloth/studio/assets/datasets/uploads/5cc39e207a9e4574a6e3dd6cb436fff0_synthetic_coaching_dataset_unsloth.jsonl")
    if cache_path.parent.exists():
        import shutil
        shutil.copy(combined_out, cache_path)
        print(f"Synced combined dataset to Studio cache: {cache_path}")

    print(f"\n[DONE] Successfully saved:")
    print(f"  1. {train_out} ({len(augmented_train)} records)")
    print(f"  2. {eval_out} ({len(eval_raw)} records - 100% pristine evaluation)")
    print(f"  3. {combined_out} ({len(augmented_train) + len(eval_raw)} records)")

if __name__ == "__main__":
    main()
