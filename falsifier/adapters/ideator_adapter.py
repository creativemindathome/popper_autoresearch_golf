"""Adapter to convert Masha's Ideator output to Leonard's FalsifierInput format.

This module provides functions to load ideator output (ideator.idea.v1 schema)
and adapt it to the FalsifierInput format for processing by the falsifier pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from falsifier.types import FalsifierInput, ParentRef, ComponentSpec


def load_ideator_idea(idea_path: Path) -> dict:
    """Load an ideator output JSON file.

    Args:
        idea_path: Path to the ideator output JSON file.

    Returns:
        Dict containing the ideator output (ideator.idea.v1 schema).

    Raises:
        FileNotFoundError: If the idea file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    idea_path = Path(idea_path)
    if not idea_path.exists():
        raise FileNotFoundError(f"Ideator output file not found: {idea_path}")

    with open(idea_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_config_delta(implementation_steps: list[dict]) -> dict[str, Any]:
    """Extract configuration deltas from implementation steps.

    Attempts to parse implementation steps that modify configuration
    parameters into a config_delta dictionary.

    Args:
        implementation_steps: List of implementation step dicts from ideator output.

    Returns:
        Dictionary of config key-value pairs, or empty dict if none found.
    """
    config_delta: dict[str, Any] = {}

    for step in implementation_steps:
        change = step.get("change", "")
        locate = step.get("locate", "")

        # Look for config parameter patterns in the change description
        # Pattern: "set X to Y" or "change X from A to B" or "X = Y"
        if "=" in change:
            # Try to extract key=value patterns
            parts = change.split("=")
            if len(parts) >= 2:
                key_part = parts[0].strip()
                value_part = parts[1].strip().split()[0].strip(",.;\"'")

                # Try to parse value as number if possible
                try:
                    if "." in value_part:
                        value: Any = float(value_part)
                    else:
                        value = int(value_part)
                except ValueError:
                    value = value_part

                # Clean up key (remove common prefixes)
                key = key_part.split()[-1].strip("\"'")
                if key:
                    config_delta[key] = value

    return config_delta


def _extract_new_components(implementation_steps: list[dict]) -> list[ComponentSpec]:
    """Extract new component specifications from implementation steps.

    Identifies steps that add new components (new classes, modules, etc.)
    that cannot be expressed as simple config deltas.

    Args:
        implementation_steps: List of implementation step dicts from ideator output.

    Returns:
        List of ComponentSpec objects for novel components, or empty list if none.
    """
    new_components: list[ComponentSpec] = []

    for step in implementation_steps:
        change = step.get("change", "")
        locate = step.get("locate", "")
        file = step.get("file", "")

        # Detect new component additions based on change description patterns
        change_lower = change.lower()
        is_new_component = any(
            keyword in change_lower
            for keyword in [
                "add new class",
                "create new",
                "implement new",
                "define new",
                "add module",
                "new layer",
                "new attention",
                "new mlp",
            ]
        )

        if is_new_component:
            # Infer component name from file or change description
            name = file.split("/")[-1].replace(".py", "") if file else "new_component"

            # Infer injection point from locate field
            injection_point = "after_attention"  # default
            locate_lower = locate.lower()
            if "mlp" in locate_lower or "feed" in locate_lower:
                injection_point = "after_mlp"
            elif "attention" in locate_lower or "attn" in locate_lower:
                injection_point = "after_attention"
            elif "embed" in locate_lower:
                injection_point = "after_embedding"
            elif "before" in locate_lower:
                injection_point = "before_mlp"

            component = ComponentSpec(
                name=name,
                code="",  # Code will be in the generated train_gpt.py file
                injection_point=injection_point,
                init_gate=0.0,  # Default: gated to zero at init
            )
            new_components.append(component)

    return new_components


def _build_parents(
    parent_implementation: dict, idea_id: str
) -> list[ParentRef]:
    """Build ParentRef list from parent_implementation info.

    Args:
        parent_implementation: Dict with repo_url, primary_file, etc.
        idea_id: The current idea's ID for describing what changed.

    Returns:
        List containing one ParentRef pointing to the seed/parent.
    """
    parents: list[ParentRef] = []

    # Extract parent identifier from repo_url or use a default
    repo_url = parent_implementation.get("repo_url", "")
    primary_file = parent_implementation.get("primary_file", "")

    # Use primary_file basename or repo name as parent node_id
    if primary_file:
        parent_id = Path(primary_file).stem
    elif repo_url:
        # Extract repo name from URL
        parent_id = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    else:
        parent_id = "seed"

    parent = ParentRef(
        node_id=parent_id,
        relationship="builds_on",
        what_changed=f"Derived from ideator idea {idea_id}",
    )
    parents.append(parent)

    return parents


def adapt_ideator_to_falsifier(
    idea: dict, knowledge_dir: Path
) -> FalsifierInput:
    """Convert ideator output to FalsifierInput format.

    Args:
        idea: Dict containing ideator output (ideator.idea.v1 schema).
        knowledge_dir: Path to the knowledge directory where generated
                      train_gpt.py files are stored.

    Returns:
        FalsifierInput populated from the ideator output.

    Raises:
        KeyError: If required fields are missing from the idea dict.
        FileNotFoundError: If the generated train_gpt.py file cannot be found.
    """
    # Extract required fields
    idea_id = idea.get("idea_id", "")
    if not idea_id:
        raise KeyError("idea_id is required in ideator output")

    novelty_summary = idea.get("novelty_summary", "")
    if not novelty_summary:
        novelty_summary = idea.get("title", "")

    # Get implementation steps for parsing
    implementation_steps = idea.get("implementation_steps", [])

    # Extract config delta from implementation steps
    config_delta = _extract_config_delta(implementation_steps)

    # Extract new components from implementation steps
    new_components = _extract_new_components(implementation_steps)

    # Build parents list from parent_implementation
    parent_implementation = idea.get("parent_implementation", {})
    parents = _build_parents(parent_implementation, idea_id)

    # Get train_gpt_path from parent_implementation
    train_gpt_path = parent_implementation.get("primary_file", "")

    # Load proposed_train_gpt from generated file
    knowledge_dir = Path(knowledge_dir)
    train_gpt_file = knowledge_dir / "outbox" / "ideator" / f"{idea_id}_train_gpt.py"

    proposed_train_gpt = ""
    if train_gpt_file.exists():
        proposed_train_gpt = train_gpt_file.read_text(encoding="utf-8")

    # Determine theory_type (default to architectural)
    theory_type = "architectural"  # ideator.idea.v1 doesn't specify, use default

    # Construct FalsifierInput
    return FalsifierInput(
        theory_id=idea_id,
        what_and_why=novelty_summary,
        config_delta=config_delta if config_delta else None,
        new_components=new_components if new_components else None,
        parents=parents,
        theory_type=theory_type,  # type: ignore[arg-type]
        proposed_train_gpt=proposed_train_gpt,
        train_gpt_path=train_gpt_path,
    )


def load_and_adapt_ideator_idea(
    idea_path: Path, knowledge_dir: Path
) -> FalsifierInput:
    """Convenience function to load and adapt an ideator output in one step.

    Args:
        idea_path: Path to the ideator output JSON file.
        knowledge_dir: Path to the knowledge directory.

    Returns:
        FalsifierInput populated from the ideator output.
    """
    idea = load_ideator_idea(idea_path)
    return adapt_ideator_to_falsifier(idea, knowledge_dir)
