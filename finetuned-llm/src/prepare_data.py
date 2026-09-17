"""
prepare_data.py
Loads the labeled support-ticket dataset, splits it into
train/val/test, and formats each example into the instruction-tuned
prompt format the base model expects. Kept dependency-light (no
torch/transformers import) so it can be unit tested without the full
ML stack installed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

LABELS = ["billing_issue", "account_access", "bug_report", "feature_request"]

PROMPT_TEMPLATE = """Classify the following support ticket into exactly one \
category: {labels}.

Ticket: "{text}"

Category:"""


@dataclass
class Example:
    text: str
    label: str

    def to_prompt(self) -> str:
        return PROMPT_TEMPLATE.format(labels=", ".join(LABELS), text=self.text)

    def to_training_pair(self) -> dict:
        """Prompt + target completion, the format train.py consumes."""
        return {"prompt": self.to_prompt(), "completion": f" {self.label}"}


def load_examples(path: str) -> list[Example]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["label"] not in LABELS:
                raise ValueError(f"Unknown label '{row['label']}' — must be one of {LABELS}")
            examples.append(Example(text=row["text"], label=row["label"]))
    return examples


def split_examples(
    examples: list[Example],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Example]]:
    """Stratified-ish shuffle split. For small demo datasets a full
    stratified split is overkill; a seeded shuffle is enough to keep
    splits reproducible."""
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare train/val/test splits from labeled tickets")
    parser.add_argument("--input", default="data/sample_tickets.jsonl")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    examples = load_examples(args.input)
    splits = split_examples(examples)

    for split_name, split_examples_list in splits.items():
        out_path = f"{args.output_dir}/{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in split_examples_list:
                f.write(json.dumps(ex.to_training_pair()) + "\n")
        print(f"{split_name}: {len(split_examples_list)} examples -> {out_path}")
