"""
inference.py
Loads the base model with the fine-tuned LoRA adapter attached and
classifies a support ticket. Also exposes a zero-shot mode (base
model, no adapter) so evaluate.py can compare the two fairly using
the exact same generation code path.
"""

from __future__ import annotations

import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.prepare_data import LABELS, PROMPT_TEMPLATE

BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
ADAPTER_DIR = os.getenv("OUTPUT_DIR", "./adapters/ticket-intent-lora")


def load_model(use_adapter: bool):
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config, device_map="auto")

    if use_adapter:
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    model.eval()
    return model, tokenizer


def _extract_label(generated_text: str) -> str:
    """Pull the first known label out of the model's raw completion."""
    for label in LABELS:
        if label in generated_text:
            return label
    return "unknown"


def classify(text: str, model, tokenizer) -> str:
    prompt = PROMPT_TEMPLATE.format(labels=", ".join(LABELS), text=text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return _extract_label(generated)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify a support ticket")
    parser.add_argument("--text", required=True)
    parser.add_argument("--zero-shot", action="store_true", help="Use base model without the fine-tuned adapter")
    args = parser.parse_args()

    model, tokenizer = load_model(use_adapter=not args.zero_shot)
    label = classify(args.text, model, tokenizer)
    print(f"Predicted label: {label}")
