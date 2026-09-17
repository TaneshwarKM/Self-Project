"""
test_prepare_data.py
Tests for prompt formatting and dataset splitting. Deliberately
avoids importing torch/transformers/peft so these tests run fast and
without a GPU or the full ML dependency stack installed.
"""

import json

from src.prepare_data import LABELS, Example, load_examples, split_examples


def test_example_to_prompt_includes_all_labels_and_text():
    ex = Example(text="I can't log in", label="account_access")
    prompt = ex.to_prompt()
    for label in LABELS:
        assert label in prompt
    assert "I can't log in" in prompt


def test_example_to_training_pair_completion_matches_label():
    ex = Example(text="App crashes on launch", label="bug_report")
    pair = ex.to_training_pair()
    assert pair["completion"].strip() == "bug_report"
    assert "App crashes on launch" in pair["prompt"]


def test_load_examples_rejects_unknown_label(tmp_path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(json.dumps({"text": "hello", "label": "not_a_real_label"}) + "\n")

    try:
        load_examples(str(bad_file))
        assert False, "expected ValueError for unknown label"
    except ValueError as exc:
        assert "not_a_real_label" in str(exc)


def test_split_examples_covers_all_data_without_overlap():
    examples = [Example(text=f"ticket {i}", label=LABELS[i % len(LABELS)]) for i in range(20)]
    splits = split_examples(examples, train_frac=0.7, val_frac=0.15, seed=42)

    all_texts = [e.text for group in splits.values() for e in group]
    assert len(all_texts) == len(examples)
    assert len(set(all_texts)) == len(examples)  # no duplicates across splits
    assert len(splits["train"]) == 14
    assert len(splits["val"]) == 3
    assert len(splits["test"]) == 3


def test_split_examples_is_deterministic_with_same_seed():
    examples = [Example(text=f"ticket {i}", label=LABELS[0]) for i in range(10)]
    split_a = split_examples(examples, seed=7)
    split_b = split_examples(examples, seed=7)
    assert [e.text for e in split_a["train"]] == [e.text for e in split_b["train"]]
