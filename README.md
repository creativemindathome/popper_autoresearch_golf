# popper_autoresearch_golf

Hackathon scaffold for an "AutoResearch" loop focused on OpenAI's Parameter Golf benchmark.

## Ideator MVP (Gemini)

The Ideator reads the current knowledge state from `./knowledge_graph/` and proposes **one** novel, testable idea per run.

### Setup

1. Create a Gemini API key in Google AI Studio (AI Studio → **Get API key** / **API keys**).
2. Export it:

```bash
export GEMINI_API_KEY="..."
```

Optional:

```bash
export GEMINI_MODEL="gemini-2.5-flash"
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
