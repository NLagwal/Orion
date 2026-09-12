#!/usr/bin/env python3
"""
Orion Dataset Formatter for Unsloth Training
=============================================
Converts raw synthetic coaching dataset into Unsloth-native ChatML / Conversational
and Alpaca formats for direct upload into Unsloth Studio or Hugging Face.
"""

import json
import sys
from pathlib import Path

SYSTEM_PROMPT = """You are Orion, an elite sports scientist and adaptive endurance coach.
Analyze this athlete's daily state and output the prescribed coaching decision adhering strictly to the structured JSON schema."""

def convert_file(input_path: Path, output_path: Path):
    if not input_path.exists():
        print(f"Error: {input_path} does not exist.")
        sys.exit(1)

    count = 0
    with open(input_path, "r", encoding="utf-8") as in_f, open(output_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt", "")
            response_obj = row.get("response", {})
            response_json_str = json.dumps(response_obj, indent=2)

            # Pure ChatML Conversational format for Unsloth / Qwen
            unsloth_record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_json_str}
                ]
            }
            out_f.write(json.dumps(unsloth_record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Successfully converted {count} records into {output_path}")

if __name__ == "__main__":
    src = Path("data/synthetic_coaching_dataset.jsonl")
    dst = Path("data/synthetic_coaching_dataset_unsloth.jsonl")
    convert_file(src, dst)
