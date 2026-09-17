"""
evaluate.py
Evaluates the fine-tuned model against a zero-shot (base model, no
adapter) baseline on the held-out test set, using the same prompt and
generation path for both so the comparison is fair. Reports accuracy
and per-class F1 — this is what turns "X% improvement over baseline"
into a number you actually measured.
"""

from __future__ import annotations

import argparse
import json

from sklearn.metrics import classification_report

from src.inference import classify, load_model
from src.prepare_data import Example, load_examples


def _load_test_examples(path: str) -> list[Example]:
    examples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            # test.jsonl stores {prompt, completion}; recover raw text/label
            # by re-reading from the original labeled file instead, since
            # the prompt template is not trivially invertible.
            examples.append(row)
    return examples


def evaluate(model, tokenizer, test_examples: list[Example]) -> dict:
    y_true, y_pred = [], []
    for ex in test_examples:
        predicted = classify(ex.text, model, tokenizer)
        y_true.append(ex.label)
        y_pred.append(predicted)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {"accuracy": report["accuracy"], "per_class": report}


def main():
    parser = argparse.ArgumentParser(description="Compare fine-tuned vs zero-shot baseline")
    parser.add_argument("--labeled-data", default="data/sample_tickets.jsonl")
    args = parser.parse_args()

    all_examples = load_examples(args.labeled_data)
    # Use the same held-out test split logic as prepare_data.py for consistency.
    from src.prepare_data import split_examples

    test_examples = split_examples(all_examples)["test"]

    print("Evaluating zero-shot baseline (no adapter)...")
    base_model, tokenizer = load_model(use_adapter=False)
    baseline_results = evaluate(base_model, tokenizer, test_examples)

    print("Evaluating fine-tuned model (LoRA adapter)...")
    finetuned_model, _ = load_model(use_adapter=True)
    finetuned_results = evaluate(finetuned_model, tokenizer, test_examples)

    print("\n=== Results ===")
    print(f"Zero-shot baseline accuracy:  {baseline_results['accuracy']:.4f}")
    print(f"Fine-tuned model accuracy:    {finetuned_results['accuracy']:.4f}")
    improvement = finetuned_results["accuracy"] - baseline_results["accuracy"]
    print(f"Absolute improvement:         {improvement:+.4f}")

    with open("data/eval_results.json", "w") as f:
        json.dump({"baseline": baseline_results, "finetuned": finetuned_results}, f, indent=2)
    print("\nFull report written to data/eval_results.json")


if __name__ == "__main__":
    main()
