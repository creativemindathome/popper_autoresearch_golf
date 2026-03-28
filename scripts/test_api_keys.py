#!/usr/bin/env python3
"""Test API key configuration."""

import os
import sys


def test_key(name, prefix=None):
    """Test if an API key is set."""
    value = os.environ.get(name)

    if not value:
        return False, "Not set"

    if prefix and not value.startswith(prefix):
        return False, f"Invalid format (should start with {prefix})"

    return True, f"{value[:20]}..."


def main():
    print("=" * 60)
    print("API Key Configuration Test")
    print("=" * 60)
    print()
    print("NOTE: There is NO 'Cursor API key'")
    print("Cursor is just your IDE. The falsifier runs locally on your machine.")
    print("You only need API keys for the LLM services it calls (Gemini/OpenAI; Anthropic optional for Stage 2).")
    print()
    print("-" * 60)

    # Test each key
    keys_to_test = [
        ("GEMINI_API_KEY", None, "Ideator (Gemini)"),
        ("OPENAI_API_KEY", "sk-", "Reviewer (OpenAI)"),
        ("ANTHROPIC_API_KEY", "sk-ant-", "Optional: Falsifier Stage 2 kill hypotheses (Claude)"),
    ]

    required = {"GEMINI_API_KEY", "OPENAI_API_KEY"}
    all_required_good = True

    for key_name, prefix, purpose in keys_to_test:
        print(f"\n{key_name}")
        print(f"  Purpose: {purpose}")

        success, message = test_key(key_name, prefix)

        if success:
            print(f"  Status: ✓ SET")
            print(f"  Value: {message}")
        else:
            print(f"  Status: ✗ {message}")
            if key_name in required:
                all_required_good = False

    print()
    print("=" * 60)

    if all_required_good:
        print("✓ Required API keys configured!")
        print()
        print("Architecture Overview:")
        print("""
    ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
    │   IDEATOR       │────▶│   REVIEWER       │────▶│   FALSIFIER     │
    │   (Gemini)      │     │   (OpenAI)       │     │   (Runs local)  │
    │   Needs API key │     │   Needs API key  │     │   No API key    │
    └─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                              │
                    ┌──────────────────────────────────────────┘
                    │
        ┌───────────┴────────────┐
        │   FALSIFIER PIPELINE   │
        │   (Local execution)    │
        ├────────────────────────┤
        │ • Stage 1 (T2-T7): MLX │◄── Runs on YOUR machine
        │ • Stage 2 (Kill hyp):  │    (NO "Cursor key" needed!)
        │   Optional Anthropic   │◄── ANTHROPIC_API_KEY (optional)
        └────────────────────────┘
        """)
        print()
        print("You can now run:")
        print("  python3 -m ideator idea --parent-train-gpt parameter-golf/train_gpt.py")
        print()
        return 0
    else:
        print("✗ Some required API keys are missing")
        print()
        print("To set them:")
        print("  export GEMINI_API_KEY='your-key'")
        print("  export OPENAI_API_KEY='sk-your-key'")
        print()
        print("Or create a .env file and run: set -a && source .env && set +a")
        print()
        print("NOTE: There is NO separate 'Cursor API key' - Cursor just runs the code!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
