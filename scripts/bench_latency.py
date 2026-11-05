#!/usr/bin/env python3
"""
scripts/bench_latency.py
Measure end-to-end latency of the Orion coach GGUF served by `llama-server`.

Sends the held-out eval prompts to the OpenAI-compatible
`/v1/chat/completions` endpoint, using greedy decoding, and reports:
  - client-side end-to-end p50 / p95 latency
  - server-reported prompt / generation timings (if exposed)
  - output JSON validity and action accuracy (sanity check)

Start the server first, e.g.:
  llama-server -m coach.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 2048

Then:
  python scripts/bench_latency.py \
    --eval_file data/synthetic_coaching_eval.jsonl \
    --max_samples 50 \
    --output_file benchmarks/latency_results.json
"""

import argparse
import datetime as dt
import json
import re
import statistics
import sys
import time

import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark llama-server latency for Orion")
    parser.add_argument("--server_url", default="http://127.0.0.1:8080")
    parser.add_argument("--eval_file", default="data/synthetic_coaching_eval.jsonl")
    parser.add_argument("--max_samples", type=int, default=50, help="0 = all")
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output_file", default="benchmarks/latency_results.json")
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
    return "coach"


def main():
    args = parse_args()
    base_url = args.server_url.rstrip("/")

    samples = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        sys.exit(f"No samples found in {args.eval_file}")

    model_id = get_model_id(base_url)
    print("=== Orion llama-server Latency Benchmark ===")
    print(f"Server:  {base_url}")
    print(f"Model:   {model_id}")
    print(f"Samples: {len(samples)}  (+{args.warmup} warmup)")

    def request(messages, max_tokens):
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        started = time.perf_counter()
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=args.timeout,
        )
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        return elapsed, response.json()

    for i in range(min(args.warmup, len(samples))):
        prompt = [m for m in samples[i]["messages"] if m["role"] != "assistant"]
        t, _ = request(prompt, args.max_new_tokens)
        print(f"warmup {i + 1}/{args.warmup}: {t:.2f}s")

    results = []
    latencies = []
    server_gen_ms = []
    server_prompt_ms = []
    completion_tokens = []
    correct = 0
    valid_json = 0

    for idx, sample in enumerate(samples):
        prompt = [m for m in sample["messages"] if m["role"] != "assistant"]
        gt_obj = extract_json(sample["messages"][-1]["content"])
        gt_action = gt_obj.get("action", "UNKNOWN") if isinstance(gt_obj, dict) else "UNKNOWN"

        elapsed, body = request(prompt, args.max_new_tokens)
        content = body["choices"][0]["message"]["content"]
        pred_obj = extract_json(content)
        pred_action = pred_obj.get("action", "UNKNOWN") if isinstance(pred_obj, dict) else "PARSE_ERROR"
        match = pred_action == gt_action
        correct += int(match)
        valid_json += int(pred_obj is not None)

        usage = body.get("usage", {}) or {}
        n_completion = usage.get("completion_tokens")
        if isinstance(n_completion, int):
            completion_tokens.append(n_completion)
        timings = body.get("timings", {}) or {}
        if isinstance(timings.get("predicted_ms"), (int, float)):
            server_gen_ms.append(float(timings["predicted_ms"]))
        if isinstance(timings.get("prompt_ms"), (int, float)):
            server_prompt_ms.append(float(timings["prompt_ms"]))

        latencies.append(elapsed)
        results.append(
            {
                "idx": idx,
                "gt_action": gt_action,
                "pred_action": pred_action,
                "match": match,
                "valid_json": pred_obj is not None,
                "client_latency_s": round(elapsed, 3),
                "completion_tokens": n_completion,
                "server_timings": timings,
            }
        )

        if (idx + 1) % 10 == 0 or idx + 1 == len(samples):
            print(
                f"[{idx + 1}/{len(samples)}] p50={percentile(latencies, 50):.2f}s "
                f"p95={percentile(latencies, 95):.2f}s | "
                f"JSON {valid_json}/{idx + 1} | action {correct}/{idx + 1}"
            )

    total = len(samples)
    summary = {
        "metadata": {
            "server_url": base_url,
            "model_id": model_id,
            "eval_file": args.eval_file,
            "samples": total,
            "warmup_requests": args.warmup,
            "decoding": "greedy (temperature=0)",
            "max_new_tokens": args.max_new_tokens,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
        "metrics": {
            "json_validity_rate": round(valid_json / total * 100, 2),
            "action_accuracy_sanity_check": round(correct / total * 100, 2),
        },
        "latency": {
            "mean_s": round(statistics.mean(latencies), 3),
            "p50_s": round(percentile(latencies, 50), 3),
            "p95_s": round(percentile(latencies, 95), 3),
            "min_s": round(min(latencies), 3),
            "max_s": round(max(latencies), 3),
            "mean_completion_tokens": round(statistics.mean(completion_tokens), 1) if completion_tokens else None,
            "client_tokens_per_second": (
                round(sum(completion_tokens) / sum(latencies), 2) if completion_tokens else None
            ),
            "server_prompt_ms_p50": round(percentile(server_prompt_ms, 50), 2) if server_prompt_ms else None,
            "server_prompt_ms_p95": round(percentile(server_prompt_ms, 95), 2) if server_prompt_ms else None,
            "server_generation_ms_p50": round(percentile(server_gen_ms, 50), 2) if server_gen_ms else None,
            "server_generation_ms_p95": round(percentile(server_gen_ms, 95), 2) if server_gen_ms else None,
        },
        "results": results,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 52)
    print("LATENCY SUMMARY")
    print("=" * 52)
    print(f"End-to-end p50: {summary['latency']['p50_s']:.2f}s")
    print(f"End-to-end p95: {summary['latency']['p95_s']:.2f}s")
    if summary["latency"]["server_generation_ms_p50"] is not None:
        print(f"Server generation p50: {summary['latency']['server_generation_ms_p50']:.0f} ms")
        print(f"Server generation p95: {summary['latency']['server_generation_ms_p95']:.0f} ms")
    print(f"JSON validity:  {summary['metrics']['json_validity_rate']:.1f}%")
    print(f"Action accuracy (sanity check): {summary['metrics']['action_accuracy_sanity_check']:.1f}%")
    print(f"\nDetailed results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
