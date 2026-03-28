# Run Autoresearch with Cursor Cloud Agents API

This guide explains how to run the autoresearch pipeline using **Cursor's Cloud Agents API**, which executes experiments on Cursor's cloud infrastructure rather than your local machine.

## What is Cursor Cloud Agents?

[Cursor Cloud Agents](https://cursor.com/docs/cloud-agent/api/endpoints) are remote AI agents that can:
- Execute code in cloud environments
- Run long-running experiments
- Access your GitHub repositories
- Work autonomously on tasks

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  YOUR MACHINE (Cursor IDE)                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Launch script: run_with_cursor_cloud.py          │   │
│  │  ├── Authenticates with Cursor API                 │   │
│  │  └── Launches cloud agent                          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼ API call
┌─────────────────────────────────────────────────────────┐
│  CURSOR CLOUD                                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Cloud Agent                                      │   │
│  │  ├── Clones your repository                       │   │
│  │  ├── Loads .env (ANTHROPIC_API_KEY)               │   │
│  │  ├── Runs: python3 run_full_live_experiment.py    │   │
│  │  ├── Generates 10 hypotheses                      │   │
│  │  ├── Runs Stage 1 & 2 falsifier                   │   │
│  │  └── Creates visualizations                       │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼ Results
┌─────────────────────────────────────────────────────────┐
│  YOUR REPOSITORY (new branch)                           │
│  └── experiments/ten_hypothesis_run/live_run_*/         │
│      ├── summary.json                                   │
│      ├── logs/                                          │
│      └── visualization/                                 │
└─────────────────────────────────────────────────────────┘
```

## Setup

### 1. Get Cursor API Key

1. Go to [cursor.com/settings](https://cursor.com/settings)
2. Navigate to API Keys
3. Create a new API key
4. Copy the key (starts with something like `cur_...`)

### 2. Set Environment Variable

```bash
export CURSOR_API_KEY="cur_your_api_key_here"
```

Add to your shell profile for persistence:
```bash
echo 'export CURSOR_API_KEY="cur_your_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

### 3. Ensure Anthropic Key is in .env

The cloud agent needs access to Anthropic API:

```bash
cd /Users/curiousmind/Desktop/null_fellow_hackathon
cat > .env << 'EOF'
ANTHROPIC_API_KEY="<your-anthropic-api-key>"
EOF
```

**Important**: The `.env` file must be committed to your repo (or use GitHub secrets if your repo is private).

### 4. Push Current Code to GitHub

```bash
git add experiments/ten_hypothesis_run/
git commit -m "Add autoresearch experiment runner"
git push origin Leonard
```

## Usage

### Quick Start

```bash
cd /Users/curiousmind/Desktop/null_fellow_hackathon/experiments/ten_hypothesis_run

# List available models first
python3 run_with_cursor_cloud.py --list-models

# Run 10 hypotheses with specific model
python3 run_with_cursor_cloud.py --model claude-4-sonnet-thinking --num-hypotheses 10
```

### Available Commands

#### 1. List Available Models
```bash
python3 run_with_cursor_cloud.py --list-models
```
Shows all Cursor Cloud models you can use:
- `claude-4-sonnet-thinking`
- `claude-4.5-sonnet-thinking`
- `gpt-5.2`
- etc.

#### 2. Manage Your Agents

**List all agents:**
```bash
python3 run_with_cursor_cloud.py --list-agents
```

**Filter by status:**
```bash
python3 run_with_cursor_cloud.py --list-agents --filter running
python3 run_with_cursor_cloud.py --list-agents --filter finished
```

**Check specific agent:**
```bash
python3 run_with_cursor_cloud.py --status bc_abc123
```

**View conversation:**
```bash
python3 run_with_cursor_cloud.py --conversation bc_abc123
```

**Add follow-up to running agent:**
```bash
python3 run_with_cursor_cloud.py --followup bc_abc123 --prompt "Generate 5 more hypotheses"
```

**Stop agent:**
```bash
python3 run_with_cursor_cloud.py --stop bc_abc123
```

**Delete agent:**
```bash
python3 run_with_cursor_cloud.py --delete bc_abc123
```

#### 3. Run Experiments

**Basic run:**
```bash
python3 run_with_cursor_cloud.py --num-hypotheses 10
```

**Specify which model to use:**
```bash
python3 run_with_cursor_cloud.py --model claude-4-sonnet-thinking --num-hypotheses 5
```

**Run without monitoring (background):**
```bash
python3 run_with_cursor_cloud.py --no-monitor --num-hypotheses 10
```

**Custom polling interval:**
```bash
python3 run_with_cursor_cloud.py --poll-interval 60  # Check every minute
```

## How It Works

1. **Launch**: Script sends experiment prompt to Cursor Cloud
2. **Agent Creation**: Cursor creates a cloud agent on a new branch
3. **Execution**: Agent runs `run_full_live_experiment.py` in cloud
4. **Monitoring**: Script polls agent status every 30 seconds
5. **Completion**: Results saved to new branch, agent reports status
6. **Results**: Pull the branch to get experiment outputs

## Cost

### Cursor Cloud
- **API calls**: Free tier available, paid plans for heavy usage
- **Agent runtime**: Billed based on compute time
- **10 hypothesis experiment**: ~$2-5 (depending on duration)

### Anthropic API
- **Idea generation**: ~$0.20 per hypothesis
- **Stage 2 falsifier**: ~$0.10 per survivor (if any)
- **10 hypothesis run**: ~$2-4 total

### Total Cost Estimate
- **10 hypotheses**: ~$4-9 USD
- **3 hypotheses**: ~$2-4 USD

## Monitoring

While running, you'll see:

```
🚀 Launching autoresearch experiment in Cursor Cloud...
   Repository: https://github.com/yourusername/null_fellow_hackathon
   Hypotheses: 10
   Model: claude-4-sonnet-thinking

✓ Agent launched!
   Agent ID: bc_abc123
   Status: CREATING
   URL: https://cursor.com/agents?id=bc_abc123

📊 Monitoring agent bc_abc123...
   Polling every 30 seconds
   (Press Ctrl+C to stop monitoring - agent will continue running)

[16:30:15] Status: CREATING
[16:30:45] Status: RUNNING
[16:31:15] Status: RUNNING
[16:31:45] Status: RUNNING
         Summary: Generated H1: gradient-sculpted-attention...
...
[17:15:30] Status: FINISHED

✓ Agent finished with status: FINISHED
```

## Getting Results

After completion:

```bash
# Pull the new branch
git fetch origin
git checkout autoresearch-run-1234567890

# View results
cd experiments/ten_hypothesis_run/live_run_*
cat summary.json
open knowledge_graph_evolution.html
```

## Troubleshooting

### "CURSOR_API_KEY not found"
```bash
export CURSOR_API_KEY="cur_your_key"
```

### "Authentication failed"
- Check your API key is correct
- Verify key hasn't expired in cursor.com/settings

### Agent fails to start
- Ensure `.env` file exists with ANTHROPIC_API_KEY
- Ensure code is pushed to GitHub
- Check repository URL is correct

### Agent runs but no results
- Check agent conversation: `python3 run_with_cursor_cloud.py --status AGENT_ID`
- Look for error messages in conversation output

### High costs
- Reduce hypotheses: `--num-hypotheses 3`
- Disable Stage 2: Modify prompt in script to add `--disable-stage2`

## Comparison: Local vs Cloud

| Aspect | Local | Cursor Cloud |
|--------|-------|--------------|
| **Compute** | Your machine | Cursor's servers |
| **Duration** | Limited by your availability | Runs unattended |
| **Monitoring** | Must keep terminal open | Runs in background |
| **Cost** | Free (your electricity) | $4-9 per 10-hyp run |
| **Setup** | Just need API keys | Need Cursor API + GitHub repo |
| **Speed** | Depends on your hardware | Consistent cloud hardware |
| **Best for** | Development, testing | Production runs, overnight experiments |

## Advanced Usage

### Custom Experiment Prompt

Edit `create_experiment_prompt()` in `run_with_cursor_cloud.py` to:
- Change experiment parameters
- Add custom validation
- Modify output format

### Selecting Which Model/Agent to Use

**List available models:**
```bash
python3 run_with_cursor_cloud.py --list-models
```

**Use specific model:**
```bash
python3 run_with_cursor_cloud.py --model claude-4-sonnet-thinking
python3 run_with_cursor_cloud.py --model gpt-5.2
python3 run_with_cursor_cloud.py --model default  # Use your default
```

### Multiple Concurrent Runs

Launch multiple experiments in parallel with different models:

```bash
# Terminal 1 - Claude
python3 run_with_cursor_cloud.py --model claude-4-sonnet-thinking --num-hypotheses 5 --no-monitor

# Terminal 2 - GPT
python3 run_with_cursor_cloud.py --model gpt-5.2 --num-hypotheses 5 --no-monitor

# Terminal 3 - Check all running
python3 run_with_cursor_cloud.py --list-agents --filter running
```

### Interactive Agent Management

```bash
# List all your agents with pretty formatting
python3 run_with_cursor_cloud.py --list-agents

# Output:
# 📋 Your Cursor Cloud Agents (3 total):
# --------------------------------------------------------------------------------
# ID                   Name                           Status       Created
# --------------------------------------------------------------------------------
# bc_abc123            Autoresearch Run 1             ✅ FINISHED   2026-03-28T16:30
# bc_def456            Autoresearch Run 2             🔄 RUNNING    2026-03-28T17:00
# bc_ghi789            Autoresearch Run 3             🆕 CREATING   2026-03-28T17:15
```

### Scheduled Runs

Use cron to run experiments on schedule:

```bash
# Edit crontab
crontab -e

# Run every day at 2 AM
0 2 * * * cd /Users/curiousmind/Desktop/null_fellow_hackathon/experiments/ten_hypothesis_run && /usr/bin/python3 run_with_cursor_cloud.py --num-hypotheses 5 --no-monitor >> /tmp/autoresearch.log 2>&1
```

## API Reference

See full [Cursor Cloud Agents API docs](https://cursor.com/docs/cloud-agent/api/endpoints):

- `POST /v0/agents` - Launch agent
- `GET /v0/agents/{id}` - Check status
- `GET /v0/agents/{id}/conversation` - Get full output
- `POST /v0/agents/{id}/stop` - Stop agent

## Support

- Cursor Cloud issues: support@cursor.com
- Autoresearch issues: Check experiment logs in `live_run_*/logs/`
