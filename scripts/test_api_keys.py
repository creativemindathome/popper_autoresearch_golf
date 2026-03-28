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
        return False, f"Invalid format (expected prefix {prefix!r})"

    return True, "set (value hidden)"


def main():
    print("=" * 60)
    print("API Key Configuration Test")
    print("=" * 60)
    print()
    print("NOTE: There is NO 'Cursor API key'")
    print("Cursor is just your IDE. The falsifier runs locally on your machine.")
    print("You only need API keys for the LLM services (Anthropic/OpenAI).")
    print()
    print("-" * 60)

    # Test each key
    keys_to_test = [
        ("ANTHROPIC_API_KEY", "sk-ant", "Ideator + Stage 2 Falsifier"),
        ("OPENAI_API_KEY", "sk-", "Reviewer (alternative to Anthropic)"),
        ("GEMINI_API_KEY", None, "Ideator (if using Gemini instead)"),
    ]

    all_good = True

    for key_name, prefix, purpose in keys_to_test:
        print(f"\n{key_name}")
        print(f"  Purpose: {purpose}")

        success, message = test_key(key_name, prefix)

        if success:
            print(f"  Status: ✓ SET")
            print(f"  Value: {message}")
        else:
            print(f"  Status: ✗ {message}")
            all_good = False

    print()
    print("=" * 60)

    if all_good:
        print("✓ All API keys configured!")
        print()
        print("Architecture Overview:")
        print("""
    ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
    │   IDEATOR       │────▶│   REVIEWER       │────▶│   FALSIFIER     │
    │   (Anthropic)   │     │   (Anthropic/    │     │   (Runs in      │
    │   Needs API key │     │    OpenAI)       │     │    Cursor IDE)  │
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
        │   Optional Anthropic   │
        └────────────────────────┘
        """)
        print()
        print("You can now run:")
        print("  python3 -m ideator.idea --parent-train-gpt parameter-golf/train_gpt.py")
        print()
        return 0
    else:
        print("✗ Some API keys are missing")
        print()
        print("To set them:")
        print("  export ANTHROPIC_API_KEY='<paste from Anthropic console>'")
        print("  export OPENAI_API_KEY='<paste from OpenAI console>'")
        print()
        print("Or create a .env file and run: set -a && source .env && set +a")
        print()
        print("NOTE: There is NO separate 'Cursor API key' - Cursor just runs the code!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
