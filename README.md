# popper_autoresearch_golf

Hackathon scaffold for an "AutoResearch" loop focused on OpenAI's Parameter Golf benchmark.

## Ideator + Novelty Reviewer (Gemini + OpenAI)

The Ideator reads the current knowledge state from `./knowledge_graph/` and proposes **one** novel, testable idea per run.

By default it runs a pessimistic OpenAI-powered **novelty reviewer**. If the reviewer says the idea is not novel enough, the Ideator revises and retries (up to a max number of rounds). Only an idea that **passes** the reviewer is emitted (so it can be sent on to the falsifier).

### Setup

1. Create a Gemini API key in Google AI Studio (AI Studio → **Get API key** / **API keys**).
2. Export it:

```bash
export GEMINI_API_KEY="..."
```

3. Create an OpenAI API key and export it (used by the novelty reviewer):

```bash
export OPENAI_API_KEY="..."
```

Optional:

```bash
export GEMINI_MODEL="gemini-2.5-flash"
export OPENAI_REVIEWER_MODEL="gpt-4o-mini"
export IDEATOR_MAX_REVIEW_ROUNDS="4"
export IDEATOR_REVIEWER_MIN_SCORE="6"
export GEMINI_TIMEOUT_S="300"
```

4. Provide a parent `train_gpt.py` to modify (recommended: clone OpenAI Parameter Golf locally):

```bash
git clone https://github.com/openai/parameter-golf.git parameter-golf
```

### Verify your API access

```bash
python3 -m ideator list-models
```

### Generate one idea

```bash
python3 -m ideator
```

Outputs a single JSON object to stdout and saves it to `knowledge_graph/outbox/ideator/latest.json`.

It also saves the generated candidate implementation files:

- `knowledge_graph/outbox/ideator/latest_train_gpt.py`
- `knowledge_graph/outbox/ideator/latest_train_gpt.patch`
- `knowledge_graph/outbox/ideator/latest_review.json`

For reviewer-approved ideas, it also writes per-idea convenience copies (for falsifier runs):

- `knowledge_graph/outbox/ideator/<idea_id>_train_gpt.py`
- `knowledge_graph/outbox/ideator/<idea_id>_parent_train_gpt.py`
- `knowledge_graph/outbox/ideator/<idea_id>_train_gpt.patch`
- `knowledge_graph/outbox/ideator/<idea_id>_review.json`
- `knowledge_graph/outbox/ideator/<idea_id>.json`

And a run bundle under:

- `knowledge_graph/outbox/ideator/runs/<run_id>/`

If your parent code is elsewhere:

```bash
python3 -m ideator --parent-train-gpt /path/to/parameter-golf/train_gpt.py
```

To disable the reviewer loop (debugging only):

```bash
python3 -m ideator --no-reviewer
```
