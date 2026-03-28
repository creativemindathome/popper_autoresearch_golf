# API Key Setup Guide

## Understanding the Architecture

When you say "Cursor for the falsifier", here's what that actually means:

| Component | What It Is | Needs API Key? |
|-----------|------------|----------------|
| **Ideator** | LLM service (Anthropic Claude) | ✅ Yes - `ANTHROPIC_API_KEY` |
| **Falsifier** | Your local Python code running in Cursor | ❌ No - just runs on your CPU |
| **Falsifier Stage 2** | Optional LLM for kill hypotheses | ✅ Optional - `ANTHROPIC_API_KEY` |
| **Reviewer** | LLM that approves ideas | ✅ Yes - `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` |

**Key Point**: Cursor is just your IDE (like VS Code). The falsifier runs Python code locally on your machine. It doesn't need a "Cursor API key" - it just needs regular API keys for the LLM services it calls.

## Quick Setup

### Option 1: Environment Variables (Recommended)

Set these in your terminal before running:

```bash
# For Ideator (using Anthropic instead of Gemini)
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"

# For Falsifier Stage 2 (optional - fallback works without)
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"  # Same key, reused

# Alternative: Use OpenAI for reviewer
export OPENAI_API_KEY="sk-your-openai-key-here"

# Original Gemini (if you want to compare)
export GEMINI_API_KEY="your-gemini-key-here"
```

### Option 2: Cursor Settings

In Cursor, you can set environment variables that persist:

1. **Open Cursor Settings** (Cmd/Ctrl + ,)
2. Go to **Features** → **AI Rules**
3. Add to **Global AI Rules**:

```json
{
  "env": {
    "ANTHROPIC_API_KEY": "sk-ant-api03-your-key-here",
    "OPENAI_API_KEY": "sk-your-openai-key-here"
  }
}
```

### Option 3: .env File

Create `.env` in project root:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
OPENAI_API_KEY=sk-your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here
```

Then load it:
```bash
set -a && source .env && set +a
```

### Option 4: Shell Profile (Permanent)

Add to `~/.zshrc` or `~/.bashrc`:

```bash
# AutoResearch Loop API Keys
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
export OPENAI_API_KEY="sk-your-openai-key-here"
```

Then reload:
```bash
source ~/.zshrc  # or ~/.bashrc
```

## Verify Setup

Test the keys are working:

```bash
# Test Anthropic
python3 -c "
import os
key = os.environ.get('ANTHROPIC_API_KEY')
if key:
    print(f'✓ ANTHROPIC_API_KEY set: {key[:20]}...')
else:
    print('✗ ANTHROPIC_API_KEY not set')
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

### 1. Anthropic for Ideator + Reviewer

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Run ideator with Anthropic
python3 -m ideator idea \
    --parent-train-gpt parameter-golf/train_gpt.py \
    --api-key $ANTHROPIC_API_KEY \
    --model claude-sonnet-4-20250514
```

### 2. Anthropic for Ideator, OpenAI for Reviewer

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
export OPENAI_API_KEY="sk-..."

python3 -m ideator idea \
    --parent-train-gpt parameter-golf/train_gpt.py \
    --api-key $ANTHROPIC_API_KEY \
    --reviewer-api-key $OPENAI_API_KEY
```

### 3. Full Loop with Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Generate idea
python3 -m ideator idea \
    --parent-train-gpt parameter-golf/train_gpt.py \
    --api-key $ANTHROPIC_API_KEY

# Falsifier will use ANTHROPIC_API_KEY for Stage 2 if available
python3 -m falsifier.main \
    --idea-id $(ls knowledge_graph/inbox/approved/*.json | head -1 | xargs basename -s .json)
```

## Where Each Key is Used

| Component | Environment Variable | Purpose | Required? |
|-----------|---------------------|---------|-----------|
| **Ideator** | `ANTHROPIC_API_KEY` | Generate architecture ideas | ✅ Required |
| **Reviewer** | `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` | Evaluate idea novelty | ✅ Required |
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

### Gemini (optional)
1. Go to https://ai.google.dev/
2. Get API key
3. Set as `GEMINI_API_KEY`

## Troubleshooting

### Key Not Recognized
```bash
# Check if key is set
echo $ANTHROPIC_API_KEY

# If empty, you need to export it
export ANTHROPIC_API_KEY="your-key"
```

### Key Expired
```bash
# Test connectivity
python3 ideator/anthropic_client.py
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
1. `ANTHROPIC_API_KEY` - for ideator generating ideas
2. `ANTHROPIC_API_KEY` (same key) - for Stage 2 kill hypotheses (optional)
3. `OPENAI_API_KEY` - alternative for reviewer (optional)

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
