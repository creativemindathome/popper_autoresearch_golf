# API Key Setup Guide

## Understanding the Architecture

When you say "Cursor for the falsifier", here's what that actually means:

| Component | What It Is | Needs API Key? |
|-----------|------------|----------------|
| **Ideator** | LLM service (Gemini + optional Claude fallback) | ✅ Yes - `GEMINI_API_KEY` (optional fallback: `ANTHROPIC_API_KEY`) |
| **Falsifier** | Your local Python code running in Cursor | ❌ No - just runs on your CPU |
| **Falsifier Stage 2** | Optional LLM for kill hypotheses | ✅ Optional - `ANTHROPIC_API_KEY` |
| **Reviewer** | LLM that approves ideas | ✅ Yes - `OPENAI_API_KEY` |

**Key Point**: Cursor is just your IDE (like VS Code). The falsifier runs Python code locally on your machine. It doesn't need a "Cursor API key" - it just needs regular API keys for the LLM services it calls.

## Quick Setup

### Option 1: Environment Variables (Recommended)

Set these in your terminal before running:

```bash
# Required: Ideator (Gemini)
export GEMINI_API_KEY="your-gemini-key-here"

# Required: Reviewer (OpenAI)
export OPENAI_API_KEY="sk-your-openai-key-here"

# Optional: Claude (used for Ideator fallback + Falsifier Stage 2) — fallback works without
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
export IDEATOR_FALLBACK_ANTHROPIC_MODEL="claude-3-5-haiku-latest"
```

### Option 2: Cursor Settings

In Cursor, you can set environment variables that persist:

1. **Open Cursor Settings** (Cmd/Ctrl + ,)
2. Go to **Features** → **AI Rules**
3. Add to **Global AI Rules**:

```json
{
  "env": {
    "GEMINI_API_KEY": "your-gemini-key-here",
    "OPENAI_API_KEY": "sk-your-openai-key-here",
    "ANTHROPIC_API_KEY": "sk-ant-your-key-here"
  }
}
```

### Option 3: .env File

Create `.env` in project root:

```bash
# .env
GEMINI_API_KEY=your-gemini-key-here
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Then load it:
```bash
set -a && source .env && set +a
```

### Option 4: Shell Profile (Permanent)

Add to `~/.zshrc` or `~/.bashrc`:

```bash
# AutoResearch Loop API Keys
export GEMINI_API_KEY="your-gemini-key-here"
export OPENAI_API_KEY="sk-your-openai-key-here"

# Optional: Stage 2 LLM (Claude)
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

Then reload:
```bash
source ~/.zshrc  # or ~/.bashrc
```

## Verify Setup

Test the keys are working:

```bash
# Test Gemini
python3 -c "
import os
key = os.environ.get('GEMINI_API_KEY')
if key:
    print('✓ GEMINI_API_KEY set')
else:
    print('✗ GEMINI_API_KEY not set')
"

# Test OpenAI
python3 -c "
import os
key = os.environ.get('OPENAI_API_KEY')
if key:
    print(f'✓ OPENAI_API_KEY set: {key[:20]}...')
else:
    print('✗ OPENAI_API_KEY not set')
"
```

## Usage Examples

### 1. Gemini for Ideator + OpenAI for Reviewer (Recommended)

```bash
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="sk-..."

python3 -m ideator idea \
    --parent-train-gpt parameter-golf/train_gpt.py \
    --knowledge-dir knowledge_graph
```

### 2. Falsifier Stage 2 with Anthropic (Optional)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# Stage 2 will try Claude if the key is set; otherwise it uses a local fallback.
python3 -m falsifier.main --idea-id <idea_id> --knowledge-dir knowledge_graph
```

### 3. Falsifier Only (No APIs Needed)

```bash
python3 -m falsifier.main \
    --candidate-json /path/to/candidate.json \
    --output-json result.json
```

## Where Each Key is Used

| Component | Environment Variable | Purpose | Required? |
|-----------|---------------------|---------|-----------|
| **Ideator** | `GEMINI_API_KEY` | Generate architecture ideas | ✅ Required |
| **Ideator fallback** | `ANTHROPIC_API_KEY` | Fallback idea generation if Gemini errors | ⚠️ Optional (auto fallback) |
| **Reviewer** | `OPENAI_API_KEY` | Evaluate idea novelty | ✅ Required |
| **Falsifier Stage 1** | None | Runs locally in Cursor | ❌ No key needed |
| **Falsifier Stage 2** | `ANTHROPIC_API_KEY` | Generate kill hypotheses | ⚠️ Optional (has fallback) |
| **Cursor IDE** | None | Just your development environment | ❌ No key needed |

### What "Cursor for Falsifier" Actually Means

When the falsifier runs "in Cursor", it means:
1. You open a terminal in Cursor (Cmd/Ctrl + `)
2. You run: `python3 -m falsifier.main ...`
3. The Python code executes on YOUR machine (using MLX/PyTorch on Apple Silicon)
4. Only Stage 2 optionally calls Anthropic API for hypothesis generation

**No Cursor API key exists or is needed.**

## Getting API Keys

### Anthropic
1. Go to https://console.anthropic.com/
2. Create account
3. Generate API key
4. Copy key starting with `sk-ant-api03-`

### OpenAI
1. Go to https://platform.openai.com/
2. Create account
3. Go to API Keys
4. Create new key starting with `sk-`

### Gemini (required for ideator)
1. Go to https://ai.google.dev/
2. Get API key
3. Set as `GEMINI_API_KEY`

## Troubleshooting

### Key Not Recognized
```bash
# Check if key is set
echo $GEMINI_API_KEY
echo $OPENAI_API_KEY

# If empty, you need to export it
export GEMINI_API_KEY="your-key"
export OPENAI_API_KEY="sk-your-key"
```

### Key Expired
```bash
# Gemini: verify your key + model access
python3 -m ideator list-models
```

### Cursor Agent Can't See Keys
In Cursor, environment variables need to be set in the terminal panel:
1. Open terminal in Cursor (Cmd/Ctrl + `)
2. Run export commands
3. Then run your commands

## FAQ: "Where's My Cursor API Key?"

**Q: I want to use Cursor for the falsifier, where's the Cursor API key?**

**A:** There is no "Cursor API key". Cursor is just an IDE (like VS Code or PyCharm). Here's the breakdown:

```
┌─────────────────────────────────────────────────────────────┐
│  Cursor IDE (just runs your code)                           │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Your Terminal in Cursor                               │ │
│  │  $ python3 -m falsifier.main --idea-id xyz            │ │
│  │                                                        │ │
│  │  This runs Python code locally on YOUR machine:       │ │
│  │  • Stage 1 gates (T2-T7) run locally using MLX        │ │
│  │  • Stage 2 optionally calls Anthropic API (if set)    │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**What you actually need:**
1. `GEMINI_API_KEY` - for ideator generating ideas
2. `OPENAI_API_KEY` - for novelty reviewer
3. `ANTHROPIC_API_KEY` - for Stage 2 kill hypotheses (optional)

**What you DON'T need:**
- ❌ Cursor API key (doesn't exist)
- ❌ Special "falsifier key" (runs locally)

## Security Note

⚠️ **Never commit API keys to git!**

Add to `.gitignore`:
```
.env
*.key
```

Use environment variables or a secrets manager in production.
