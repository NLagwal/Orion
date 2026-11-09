# Orion — Training & Evaluation

This document describes how the Orion edge coach was built and how to reproduce
its evaluation. It replaces the earlier planning notes with the configuration
that was actually used.

## 1. Data

Orion's coaching-decision dataset is synthetic, but its input states are sampled
from real wearable telemetry:

- **PMData** (Simula Research Laboratory): 16 participants, roughly five months
  of Fitbit and PMSys daily logs.
- **Wearable HRV + sleep-diary dataset (2025)**: HRV noise and
  mental-health-adjacent signals used for escalation examples.

`scripts/flatten_pmdata.py` produces `data/pmdata_flattened.csv` (2,354 athlete
days). `scripts/generate_synthetic_coaching_data.py` sends each state to a
frontier model with an explicit coaching rubric and writes the raw
`(state → decision)` records to `data/synthetic_coaching_dataset.jsonl`.

Generated labels are checked against the same deterministic guardrail rules
used at inference time before they become training data. The guardrails cover
ACWR ceilings, HRV/RHR deviations, and injury escalation.

`scripts/augment_leak_free.py` then:

1. Holds out 235 examples (10%, fixed seed 3407) as the evaluation split.
2. Leaves that split untouched — no augmented copies.
3. Augments only the underrepresented training classes.
4. Writes `data/synthetic_coaching_train.jsonl` (2,230 examples) and
   `data/synthetic_coaching_eval.jsonl` (235 examples).

### Label distribution

| Action | Train | Eval |
|---|---:|---:|
| `MAINTAIN_PLAN` | 970 | 103 |
| `SWAP_TO_ZONE2` | 590 | 73 |
| `REST_DAY` | 376 | 32 |
| `MODIFY_INTERVALS` | 148 | 5 |
| `ESCALATE_TO_HUMAN` | 146 | 22 |

`MODIFY_INTERVALS` has only five held-out examples, so its per-class
precision/recall has wide error bars. Report it with that caveat.

## 2. Fine-tuning

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Method | QLoRA (4-bit) via Unsloth |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.0 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Epochs | 2 |
| Optimizer steps | 280 |
| Final train loss | 0.2745 |
| Validation cross-entropy | 0.3062 |

The 235-example evaluation split was held out from gradient updates and used for
validation loss during training. It is never trained on.

The adapter was merged into a 16-bit checkpoint (`coach_merged_16bit/`) and
converted to a Q8_0 GGUF (`coach.gguf`, 3.28 GB) with llama.cpp's
`convert_hf_to_gguf.py`:

```bash
python /path/to/llama.cpp/convert_hf_to_gguf.py coach_merged_16bit \
  --outfile coach.gguf --outtype q8_0
```

## 3. Evaluation

`scripts/eval_merged.py` evaluates the merged checkpoint with plain
`transformers` — Unsloth, PEFT, and bitsandbytes are not required:

```bash
python scripts/eval_merged.py \
  --model_dir coach_merged_16bit \
  --eval_file data/synthetic_coaching_eval.jsonl \
  --max_samples 0 \
  --output_file benchmarks/eval_results.json
```

It reports:

- JSON validity rate
- schema compliance rate
- action accuracy and per-class precision/recall/F1
- rationale completeness
- generation latency and throughput

### LLM-only held-out results (235 examples)

Model-only results, before deterministic guardrail overrides:

| Metric | Value |
|---|---:|
| JSON validity | 100.0% (235/235) |
| Schema compliance | 99.1% (233/235) |
| Action accuracy | 68.5% (161/235) |
| Non-empty rationale | 100.0% (235/235) |

| Action | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `MAINTAIN_PLAN` | 0.740 | 0.913 | 0.817 | 103 |
| `SWAP_TO_ZONE2` | 0.850 | 0.466 | 0.602 | 73 |
| `REST_DAY` | 0.433 | 0.406 | 0.419 | 32 |
| `MODIFY_INTERVALS` | 0.133 | 0.400 | 0.200 | 5 |
| `ESCALATE_TO_HUMAN` | 0.783 | 0.818 | 0.800 | 22 |

The weak `SWAP_TO_ZONE2` and `REST_DAY` recall is why action selection moved to
the router. `MODIFY_INTERVALS` has only five held-out examples; its per-class
numbers are not reliable at that sample size.

### Hybrid evaluation (235 examples)

Router → guardrails → LLM rationale:

| Metric | Value |
|---|---:|
| Router action accuracy | 88.5% (208/235) |
| Hybrid action accuracy | 88.5% (208/235) |
| Guardrail override rate | 0.0% (0/235) |
| JSON validity | 100.0% (235/235) |
| Schema compliance | 100.0% (235/235) |
| End-to-end p50 / p95 | 2.79 s / 3.10 s |

The hybrid matches the router because the router already avoided the hard
safety violations on this split. Guardrails are safety overrides, not an
accuracy booster. They fire on injury, ACWR danger, and elevated-RHR/high-fatigue
cases, and are covered by adversarial tests.

The LLM was told the router's action and followed it in 235/235 cases. Its
independent LLM-only accuracy remains **68.5%**, not 88.5%.

Reproduce:

```bash
llama-server -m coach.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 2048
python3 scripts/eval_hybrid.py \
  --server_url http://127.0.0.1:8080 \
  --max_samples 0 \
  --output_file benchmarks/hybrid_results.json
```

`scripts/eval_qwen_lora.py` is the original Unsloth-based adapter harness. It
requires `unsloth`, `peft`, `bitsandbytes`, and `accelerate`.

Serving latency is measured separately against `llama-server`:

```bash
llama-server -m coach.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 2048
python scripts/bench_latency.py --max_samples 50
```
