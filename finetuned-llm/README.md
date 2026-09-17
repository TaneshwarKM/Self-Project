# Fine-Tuned LLM for Support Ticket Classification (QLoRA)

Fine-tunes an open-source instruction model (Mistral-7B-Instruct)
using QLoRA to classify support tickets into intent categories
(`billing_issue`, `account_access`, `bug_report`, `feature_request`),
and benchmarks it against zero-shot prompting on the same base model
to quantify the actual lift from fine-tuning.

## Why this exists

General-purpose LLMs are expensive to run at scale and often
underperform on narrow, domain-specific tasks compared to a small
model fine-tuned specifically for that task. QLoRA makes fine-tuning
feasible on a single consumer/prosumer GPU by loading the base model
in 4-bit and training only a small LoRA adapter instead of the full
model.

## Architecture

```
 sample_tickets.jsonl (labeled data)
         │
         ▼
  prepare_data.py ──▶ train/val/test split, prompt formatting
         │
         ▼
    train.py     ──▶ load base model in 4-bit (bitsandbytes)
         │            + LoRA adapter (PEFT) on attention projections
         │            + Trainer loop, tracked in W&B
         ▼
  adapters/ticket-intent-lora/   (saved LoRA weights, a few MB)
         │
         ▼
   evaluate.py   ──▶ same prompt + generation path for:
         │              - zero-shot baseline (base model, no adapter)
         │              - fine-tuned model (base + adapter)
         ▼
   accuracy / per-class F1 comparison
```

## Features

- **QLoRA fine-tuning** — 4-bit quantized base model + LoRA adapter
  via PEFT, so training is feasible on a single GPU instead of
  requiring full-model fine-tuning infrastructure.
- **Fair baseline comparison** — `evaluate.py` runs the exact same
  prompt template and generation code for both the zero-shot base
  model and the fine-tuned model, so the accuracy delta is
  attributable to fine-tuning, not prompt differences.
- **Reproducible splits** — seeded train/val/test split so results
  are comparable across runs.
- **Experiment tracking** — training runs log to Weights & Biases
  when `WANDB_API_KEY` is set.
- **Dependency-light tests** — `tests/test_prepare_data.py` covers
  prompt formatting and data splitting without requiring torch or a
  GPU, so the test suite runs anywhere.

## Setup

```bash
git clone <this-repo>
cd finetuned-llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add HF_TOKEN (and WANDB_API_KEY if tracking)
```

**Hardware note:** `train.py` and `inference.py` require a CUDA GPU
with `bitsandbytes` support (a single 16GB+ GPU is enough for a 7B
model in 4-bit). `prepare_data.py` and its tests have no such
requirement and run anywhere.

## Usage

```bash
# 1. Split the labeled data into train/val/test
python -m src.prepare_data --input data/sample_tickets.jsonl

# 2. Fine-tune (requires GPU)
python -m src.train

# 3. Compare fine-tuned vs zero-shot baseline (requires GPU)
python -m src.evaluate

# 4. Classify a new ticket
python -m src.inference --text "I was charged twice this month"
python -m src.inference --text "..." --zero-shot   # baseline, for comparison
```

## Results

Measured with `evaluate.py` on the held-out test split of the sample
dataset (20 labeled tickets, 4 categories):

| Metric | Zero-shot baseline | Fine-tuned (LoRA) |
|---|---|---|
| Accuracy | *run `evaluate.py` and fill in* | *run `evaluate.py` and fill in* |
| Macro F1 | *run `evaluate.py` and fill in* | *run `evaluate.py` and fill in* |

> The included sample dataset (20 examples) is enough to prove the
> pipeline works end-to-end, but is too small for a statistically
> meaningful accuracy comparison. For a real resume claim, replace
> `data/sample_tickets.jsonl` with a few hundred labeled examples
> (e.g. from a public intent-classification dataset) and re-run.

## Project structure

```
finetuned-llm/
├── src/
│   ├── prepare_data.py   # load, split, format labeled tickets
│   ├── train.py           # QLoRA fine-tuning script
│   ├── inference.py       # load base+adapter (or base alone), classify
│   └── evaluate.py        # fine-tuned vs zero-shot baseline comparison
├── data/
│   └── sample_tickets.jsonl
├── tests/
│   └── test_prepare_data.py   # dependency-light, no GPU needed
└── requirements.txt
```

## Design decisions worth noting

- **Why QLoRA over full fine-tuning?** Full fine-tuning a 7B model
  needs far more GPU memory and storage per checkpoint. QLoRA trains
  <1% of parameters and produces an adapter a few MB in size, while
  typically recovering most of the performance gain full fine-tuning
  would give on a narrow task.
- **Why compare against zero-shot, not few-shot?** Few-shot prompting
  narrows the gap fine-tuning would show, and burns context-window
  tokens on every inference call. Zero-shot is the more common
  production baseline for a real classification endpoint, so it's
  the fairer "what would we deploy without fine-tuning" comparison.
- **Why keep data prep dependency-light?** Splitting/formatting logic
  is pure Python with no need for the ML stack, so it's tested
  without a GPU — keeping CI fast and not requiring GPU runners just
  to catch a data-formatting bug.

## Possible extensions

- Swap the label-substring extraction in `inference.py` for
  constrained decoding (force the model to only generate a valid
  label token) to eliminate ambiguous completions entirely
- Add a confusion matrix visualization to `evaluate.py`
- Try QLoRA on a second base model and compare
- Package `inference.py` behind a FastAPI endpoint for serving

## License

MIT
