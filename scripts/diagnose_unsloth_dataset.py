import json
from transformers import AutoTokenizer
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained('unsloth/Qwen2.5-3B-Instruct')
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

dataset = load_dataset('json', data_files='data/synthetic_coaching_dataset_unsloth.jsonl', split='train')
print(f"Loaded {len(dataset)} samples")
print(f"Dataset columns: {dataset.column_names}")

# Check sample 0
sample_0 = dataset[0]
print("\nSample 0 keys:", sample_0.keys())

# Test formatting
def formatting_prompts_func(examples):
    convs = examples["messages"]
    texts = [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in convs]
    return { "text" : texts }

formatted = dataset.select(range(75)).map(formatting_prompts_func, batched=True)
print("\nFirst sample formatted text snippet:\n", repr(formatted[0]["text"][:200]))

# Check response template in Qwen 2.5
# In Qwen 2.5: <|im_start|>assistant\n
print("\nTesting train_on_responses_only...")
try:
    trainer_dataset = train_on_responses_only(
        formatted,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        tokenizer=tokenizer,
    )
    print("train_on_responses_only succeeded!")
except Exception as e:
    print("train_on_responses_only failed with:", e)
