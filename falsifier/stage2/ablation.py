"""Build ablation source code (theory with one change reverted)."""

from __future__ import annotations

from typing import Any


def build_ablation_source(
    theory_src: str,
    parent_src: str,
    config_delta: dict[str, Any] | None,
    change_to_revert: str,
) -> str:
    """Build theory source with ONE change reverted.
    
    Start from parent (SOTA), apply all config_delta EXCEPT the target.
    """
    if config_delta is None:
        return parent_src
    
    # Remove the target change
    reduced = {k: v for k, v in config_delta.items() if k != change_to_revert}
    
    # Apply remaining changes to parent source
    return apply_config_delta(parent_src, reduced)


def apply_config_delta(source: str, delta: dict[str, Any]) -> str:
    """Apply config delta to source code.
    
    This is a simple string-based replacement. For each key in delta,
    we look for patterns like `key = value` in the Hyperparameters class
    and replace with the new value.
    """
    lines = source.splitlines()
    result_lines = []
    
    in_hyperparams = False
    hyperparams_indent = None
    
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        
        # Detect Hyperparameters class start
        if stripped.startswith("class Hyperparameters"):
            in_hyperparams = True
            hyperparams_indent = None
            result_lines.append(line)
            continue
        
        # Detect end of Hyperparameters (next class or def at same or lower indent)
        if in_hyperparams:
            if stripped and not stripped.startswith("#"):
                if stripped.startswith("class ") or stripped.startswith("def "):
                    if hyperparams_indent is not None and indent <= hyperparams_indent:
                        in_hyperparams = False
            
            # Track first real line indent
            if hyperparams_indent is None and stripped and not stripped.startswith("#"):
                hyperparams_indent = indent
        
        # Try to match and replace delta keys
        replaced = False
        if in_hyperparams and stripped:
            for key, value in delta.items():
                # Match patterns like: key = value, key: type = value, key = int(...)
                import re
                pattern = rf'^([ \t]*)({re.escape(key)})\s*[=:]\s*[^#\n]+(.*)$'
                match = re.match(pattern, line)
                if match:
                    indent_str = match.group(1)
                    rest = match.group(3)
                    # Format value appropriately
                    if isinstance(value, str):
                        new_line = f'{indent_str}{key} = "{value}"{rest}'
                    elif isinstance(value, bool):
                        new_line = f'{indent_str}{key} = {int(value)}{rest}'
                    else:
                        new_line = f'{indent_str}{key} = {value}{rest}'
                    result_lines.append(new_line)
                    replaced = True
                    break
        
        if not replaced:
            result_lines.append(line)
    
    return "\n".join(result_lines)
