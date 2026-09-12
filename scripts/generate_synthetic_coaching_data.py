#!/usr/bin/env python3
"""
Orion Synthetic Coaching Dataset Generator (CLI)
================================================
Reads data/pmdata_flattened.csv and calls the coaching reasoning LLM
(antigravity/gemini-3.7-flash-high) via the local omniroute proxy.
Produces structured coaching decisions validated against the Orion schema.

Features:
- Streaming JSONL writes (zero RAM buildup, crash-proof).
- Automatic resume (skips already processed records).
- Strict JSON validation & markdown-stripping.
- Bounded concurrency (default: 2 parallel requests).
- Exponential backoff retry logic.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orion_generator")

DEFAULT_ENDPOINT = "http://localhost:20128/v1/chat/completions"
DEFAULT_API_KEY = "your_omniroute_api_key"
DEFAULT_MODEL = "antigravity/gemini-3.7-flash-high"
DEFAULT_CSV_PATH = Path("data/pmdata_flattened.csv")
DEFAULT_OUTPUT_PATH = Path("data/synthetic_coaching_dataset.jsonl")

SYSTEM_PROMPT = """You are Orion, an elite sports scientist and adaptive endurance coach.
Analyze this athlete's daily state and output the prescribed coaching decision adhering strictly to the structured JSON schema."""

PROMPT_TEMPLATE = """ATHLETE PROFILE:
- ID: {participant_id}
- Age: {age}, Gender: {gender}, Max HR: {max_hr} bpm
- 5km Baseline Fitness: {baseline_5k_sec} seconds

TODAY'S OBJECTIVE TELEMETRY:
- Resting HR: {resting_heart_rate} bpm (7d Trend Deviation: {rhr_trend_delta} bpm)
- Sleep: {sleep_duration_min} min (Efficiency: {sleep_efficiency}%, Deep Sleep: {sleep_deep_min}m, REM: {sleep_rem_min}m, Wake: {sleep_wake_min}m)
- Training Load: ACWR {acwr} (Acute 7d Load: {rolling_load_7d}, Chronic 28d Load: {rolling_load_28d}, Today's Executed Load: {daily_training_load})

SUBJECTIVE CHECK-IN:
- Readiness: {readiness}/10, Fatigue: {fatigue}/5, Soreness: {soreness}/5, Stress: {stress}/5, Mood: {mood}/5
- Reported Injury/Pain: {injury_reported} ({injury_details})

COACHING RULES & RUBRIC:
1. ACWR >= 1.50: Danger zone. MUST modify intervals or force REST_DAY to prevent soft tissue injury.
2. HRV / RHR Alert: If resting HR is elevated (>1.5 bpm vs 7d) AND fatigue >= 4 -> SWAP_TO_ZONE2 or REST_DAY.
3. Injury Signal: If injury_reported == 1 with joint/tendon/pain details -> ESCALATE_TO_HUMAN with reason_category "injury_signal".
4. Safe / Optimal: If ACWR is 0.8-1.3, RHR trend is stable, and soreness is low -> MAINTAIN_PLAN.
5. Voice & Rationale: Output exactly one calm, honest, concise sentence explaining why (e.g. "Sleep was short and HRV dropped 9pts — swapped intervals for 40min easy run."). Never use generic filler jargon.

Respond ONLY with a JSON object matching this schema:
{{
  "action": "MODIFY_INTERVALS" | "SWAP_TO_ZONE2" | "REST_DAY" | "MAINTAIN_PLAN" | "ESCALATE_TO_HUMAN",
  "rationale": "<one calm, precise, non-generic sentence>",
  "modified_workout": {{
    "name": "<workout name>",
    "detail": "<specific instructions>",
    "duration_minutes": <integer>,
    "target_pace": "<optional pace range e.g. 5:30-5:50 /km>",
    "target_zone": "<optional zone e.g. Zone 2 Aerobic>"
  }},
  "confidence": <float 0.0 to 1.0>,
  "escalation": {{
    "reason_category": "illness_pattern" | "injury_signal" | "overtraining_risk" | "data_conflict_unresolvable" | "none",
    "trigger_source": "model" | "guardrail"
  }}
}}"""

VALID_ACTIONS = {
    "MODIFY_INTERVALS",
    "SWAP_TO_ZONE2",
    "REST_DAY",
    "MAINTAIN_PLAN",
    "ESCALATE_TO_HUMAN",
}
VALID_REASONS = {
    "illness_pattern",
    "injury_signal",
    "overtraining_risk",
    "data_conflict_unresolvable",
    "none",
}


def clean_json_text(text: str) -> str:
    """Strip markdown backticks or extra preamble if present."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def validate_decision(obj: Any) -> dict[str, Any]:
    """Ensure generated JSON strictly adheres to the Orion schema."""
    if not isinstance(obj, dict):
        raise ValueError("Root response is not a JSON object")

    action = obj.get("action")
    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid action: {action}")

    rationale = obj.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Missing or empty rationale")

    workout = obj.get("modified_workout")
    if not isinstance(workout, dict):
        raise ValueError("Missing modified_workout object")
    if "name" not in workout or "detail" not in workout or "duration_minutes" not in workout:
        raise ValueError("modified_workout missing required keys (name, detail, duration_minutes)")

    confidence = obj.get("confidence")
    if not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a float")

    escalation = obj.get("escalation")
    if not isinstance(escalation, dict):
        raise ValueError("Missing escalation object")
    if escalation.get("reason_category") not in VALID_REASONS:
        raise ValueError(f"Invalid reason_category: {escalation.get('reason_category')}")

    return obj


def format_row_value(val: str | None) -> str:
    if val is None or val == "" or str(val).lower() == "nan":
        return "None"
    return str(val)


async def generate_single_record(
    client: httpx.AsyncClient,
    endpoint: str,
    headers: dict[str, str],
    model: str,
    row: dict[str, Any],
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict[str, Any] | None:
    """Generate and validate a single coaching response with retries."""
    formatted_kwargs = {k: format_row_value(v) for k, v in row.items()}
    user_prompt = PROMPT_TEMPLATE.format(**formatted_kwargs)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.6,
    }

    for attempt in range(1, max_retries + 1):
        async with semaphore:
            try:
                resp = await client.post(endpoint, json=payload, headers=headers, timeout=45.0)
                if resp.status_code != 200:
                    logger.warning(
                        f"HTTP {resp.status_code} for {row.get('participant_id')} "
                        f"on {row.get('date')} (attempt {attempt}/{max_retries}): {resp.text[:120]}"
                    )
                    await asyncio.sleep(1.5 * attempt)
                    continue

                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                cleaned = clean_json_text(content)
                parsed = json.loads(cleaned)
                validated = validate_decision(parsed)

                # Return complete training pair
                return {
                    "participant_id": row.get("participant_id"),
                    "date": row.get("date"),
                    "athlete_state": row,
                    "prompt": user_prompt,
                    "response": validated,
                }

            except (httpx.RequestError, httpx.TimeoutException) as e:
                logger.warning(f"Network error on attempt {attempt}/{max_retries}: {e}")
                await asyncio.sleep(2.0 * attempt)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Validation error on attempt {attempt}/{max_retries}: {e}")
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt}/{max_retries}: {e}")
                await asyncio.sleep(2.0)

    logger.error(f"Failed record {row.get('participant_id')} {row.get('date')} after {max_retries} attempts.")
    return None


async def main():
    parser = argparse.ArgumentParser(description="Orion Synthetic Dataset Generator")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Path to input flattened CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSONL path")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model identifier")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="OpenAI-compatible endpoint")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--concurrency", type=int, default=2, help="Max parallel requests (default: 2)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to process")
    args = parser.parse_args()

    if not args.csv.exists():
        logger.error(f"Input file not found: {args.csv}")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Count existing records for resume capability
    existing_keys: set[tuple[str, str]] = set()
    if args.output.exists():
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        existing_keys.add((record["participant_id"], record["date"]))
                    except Exception:
                        pass
        logger.info(f"Resuming: found {len(existing_keys)} already completed records in {args.output}")

    # Read CSV rows
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # Filter out already processed
    rows_to_process = [
        row for row in all_rows if (row.get("participant_id"), row.get("date")) not in existing_keys
    ]

    if args.limit:
        rows_to_process = rows_to_process[: args.limit]

    total_target = len(rows_to_process)
    logger.info(
        f"Starting generation: {total_target} records to generate (Concurrency: {args.concurrency}, Model: {args.model})"
    )

    if total_target == 0:
        logger.info("All records already generated! Output is complete.")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }

    start_time = time.perf_counter()
    completed = 0
    succeeded = 0
    failed = 0

    # Process in streaming chunks so output file is continuously flushed
    chunk_size = 20
    limits = httpx.Limits(max_keepalive_connections=args.concurrency * 2, max_connections=args.concurrency * 4)

    async with httpx.AsyncClient(limits=limits) as client:
        with open(args.output, "a", encoding="utf-8") as out_f:
            for i in range(0, total_target, chunk_size):
                chunk = rows_to_process[i : i + chunk_size]
                tasks = [
                    generate_single_record(
                        client=client,
                        endpoint=args.endpoint,
                        headers=headers,
                        model=args.model,
                        row=row,
                        semaphore=semaphore,
                    )
                    for row in chunk
                ]

                results = await asyncio.gather(*tasks)

                for res in results:
                    completed += 1
                    if res is not None:
                        succeeded += 1
                        out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        out_f.flush()
                    else:
                        failed += 1

                elapsed = time.perf_counter() - start_time
                rate = completed / max(1.0, elapsed)
                eta = (total_target - completed) / max(0.01, rate)

                logger.info(
                    f"Progress: {completed}/{total_target} ({completed / total_target * 100:.1f}%) | "
                    f"Success: {succeeded} | Failed: {failed} | "
                    f"Speed: {rate:.2f} rec/s | ETA: {int(eta // 60)}m {int(eta % 60)}s"
                )

    total_time = time.perf_counter() - start_time
    logger.info(
        f"Finished generation in {int(total_time // 60)}m {int(total_time % 60)}s. "
        f"Total saved: {succeeded}/{total_target} records to {args.output}"
    )


if __name__ == "__main__":
    asyncio.run(main())
