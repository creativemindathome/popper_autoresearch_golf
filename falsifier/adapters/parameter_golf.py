from __future__ import annotations

import importlib.util
import os
import sys
import types
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from ..types import ModelSignature, SmokeDiagnostics


class _RecursiveStubModule(types.ModuleType):
    """A stub module that returns itself for any attribute access.
    
    This handles submodules like `mlx.optimizers` where code does:
        import mlx.optimizers
    Python tries to access mlx.optimizers, so the mlx stub must support 
    arbitrary attribute access.
    
    Also supports being treated as a package (has __path__) and used as
    a base class (provides __mro_entries__).
    """
    
    def __init__(self, name: str):
        super().__init__(name)
        # Make it look like a package to support submodule imports
        self.__path__ = []  # Empty list means it's a namespace package
    
    def __getattr__(self, name: str) -> Any:
        # Skip special attributes
        if name in ('__name__', '__doc__', '__package__', '__spec__', '__file__', 
                    '__path__', '__cached__', '__loader__', '__mro_entries__'):
            return object.__getattribute__(self, name)
        
        # Return a new recursive stub for any attribute access
        submodule_name = f"{self.__name__}.{name}"
        if submodule_name in sys.modules:
            return sys.modules[submodule_name]
        # Create and cache the submodule stub
        stub = _RecursiveStubModule(submodule_name)
        sys.modules[submodule_name] = stub
        return stub
    
    def __call__(self, *args, **kwargs):
        # Support being called as a function/class
        return None
    
    def __iter__(self):
        # Support iteration (empty iterator)
        return iter([])
    
    def __getitem__(self, key):
        # Support indexing
        return None
    
    @classmethod
    def __mro_entries__(cls, bases):
        # When used as a base class, return an empty tuple to exclude from MRO
        return ()


def _pre_stub_modules(names: list[str]) -> None:
    """Pre-create stub modules in sys.modules before they are imported.
    
    For packages like 'mlx', also pre-stubs common submodules to support
    import patterns like 'import mlx.optimizers'.
    """
    # Map of base packages to their common submodules
    submodule_map: dict[str, list[str]] = {
        "mlx": ["mlx.optimizers", "mlx.core", "mlx.nn", "mlx.utils"],
        "torch": ["torch.nn", "torch.optim"],
    }
    
    all_names = list(names)
    for name in names:
        if name in submodule_map:
            all_names.extend(submodule_map[name])
    
    for name in all_names:
        if name not in sys.modules:
            # Use recursive stub for packages that may have submodules
            base_name = name.split(".")[0]
            if base_name in ("mlx", "torch"):
                stub = _RecursiveStubModule(name)
            else:
                stub = types.ModuleType(name)
                stub.__dict__["__spec__"] = None
                stub.__dict__["__file__"] = None
            
            # For sentencepiece specifically, add common attributes
            if "sentencepiece" in name or name == "spm":
                class _SentencePieceProcessor:
                    def __init__(self, *args, **kwargs):
                        pass
                    def Load(self, *args, **kwargs):
                        pass
                    def Encode(self, *args, **kwargs):
                        return []
                    def Decode(self, *args, **kwargs):
                        return ""
                    def __getattr__(self, name):
                        return lambda *args, **kwargs: None
                
                stub.SentencePieceProcessor = _SentencePieceProcessor
                stub.SentencePieceTrainer = type("SentencePieceTrainer", (), {
                    "Train": staticmethod(lambda *args, **kwargs: None),
                    "__getattr__": lambda self, name: lambda *args, **kwargs: None,
                })
            
            sys.modules[name] = stub


def _cleanup_stub_modules(names: list[str]) -> None:
    """Remove stub modules from sys.modules.
    
    Also removes any nested submodules that were created (e.g., mlx.optimizers).
    """
    # Find all modules that start with any of the names (for nested stubs)
    to_remove = set()
    for name in names:
        if name in sys.modules:
            to_remove.add(name)
        # Also remove submodules
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith(name + "."):
                to_remove.add(mod_name)
    
    for name in to_remove:
        if name in sys.modules:
            del sys.modules[name]


MINIMAL_TRAIN_GPT_ENV = {
    "NUM_LAYERS": "2",
    "MODEL_DIM": "32",
    "NUM_HEADS": "4",
    "NUM_KV_HEADS": "2",
    "MLP_MULT": "2",
    "VOCAB_SIZE": "64",
    "TIE_EMBEDDINGS": "1",
    "TRAIN_SEQ_LEN": "8",
    "VAL_BATCH_SIZE": "8",
    "TRAIN_BATCH_TOKENS": "8",
    "ITERATIONS": "1",
    "MAX_WALLCLOCK_SECONDS": "0",
    "VAL_LOSS_EVERY": "0",
    "TRAIN_LOG_EVERY": "0",
}


@contextmanager
def _patched_env(overrides: dict[str, str] | None):
    merged = dict(MINIMAL_TRAIN_GPT_ENV)
    if overrides:
        merged.update({key: str(value) for key, value in overrides.items()})

    previous: dict[str, str | None] = {}
    for key, value in merged.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_train_gpt_module(
    train_gpt_path: str | Path,
    env_overrides: dict[str, str] | None = None,
    block_imports: list[str] | None = None,
) -> ModuleType:
    """Load train_gpt.py module with optional import blocking.
    
    Args:
        train_gpt_path: Path to train_gpt.py
        env_overrides: Environment variable overrides
        block_imports: List of module names to stub (e.g., ["sentencepiece"])
    """
    path = Path(train_gpt_path).resolve()
    module_name = f"falsifier_train_gpt_{uuid.uuid4().hex}"
    
    # Determine which imports to stub
    to_stub = block_imports or []
    if os.environ.get("BLOCK_NON_IMPORTANT_IMPORTS", "false").lower() == "true":
        to_stub.extend(["sentencepiece", "spm"])
    
    # Pre-stub modules to prevent import errors
    _pre_stub_modules(to_stub)
    
    try:
        with _patched_env(env_overrides):
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"unable to load train_gpt.py from {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
    finally:
        # Clean up stub modules
        _cleanup_stub_modules(to_stub)


def _detect_file_framework(train_gpt_path: str | Path) -> str:
    """Detect whether a train_gpt file is PyTorch or MLX based.

    Args:
        train_gpt_path: Path to the train_gpt file

    Returns:
        "pytorch" or "mlx"
    """
    path = Path(train_gpt_path)
    if not path.exists():
        return "pytorch"  # Default fallback

    content = path.read_text()

    # Check for MLX-specific imports and patterns
    if "import mlx" in content or "from mlx" in content:
        return "mlx"
    if "train_gpt_mlx" in path.name:
        return "mlx"

    # Default to PyTorch
    return "pytorch"


def instantiate_minimal_model(
    train_gpt_path: str | Path,
    env_overrides: dict[str, str] | None = None,
    block_imports: list[str] | None = None,
) -> tuple[ModuleType, Any]:
    """Instantiate a minimal model from train_gpt.py or train_gpt_mlx.py.

    Args:
        train_gpt_path: Path to train_gpt.py or train_gpt_mlx.py
        env_overrides: Environment variable overrides
        block_imports: List of module names to stub (defaults to sentencepiece only)
            Note: MLX is now supported, so it's not in default block_imports

    Returns:
        Tuple of (module, model) - model type depends on the file (PyTorch or MLX)
    """
    # Determine file framework
    file_framework = _detect_file_framework(train_gpt_path)

    # Build block_imports - only block sentencepiece by default (MLX is now supported)
    to_block = block_imports or ["sentencepiece", "spm"]

    module = load_train_gpt_module(
        train_gpt_path,
        env_overrides=env_overrides,
        block_imports=to_block,
    )
    args = module.Hyperparameters()

    # Dynamically inspect GPT.__init__ signature to determine correct parameter names
    import inspect
    sig = inspect.signature(module.GPT.__init__)
    gpt_params = list(sig.parameters.keys())
    # Remove 'self' from the list
    gpt_params = [p for p in gpt_params if p != 'self']
    
    # Build kwargs by matching Hyperparameters attributes to GPT parameter names
    kwargs: dict[str, Any] = {}
    
    # Mapping of Hyperparameters attr -> possible GPT param names
    # The first match found in gpt_params is used
    param_candidates: dict[str, list[str]] = {
        "vocab_size": ["vocab_size"],
        "num_layers": ["num_layers"],
        "model_dim": ["model_dim", "dim"],  # Try model_dim first, then dim
        "num_heads": ["num_heads"],
        "num_kv_heads": ["num_kv_heads"],
        "mlp_mult": ["mlp_mult"],
        "tie_embeddings": ["tie_embeddings"],
        "tied_embed_init_std": ["tied_embed_init_std"],
        "logit_softcap": ["logit_softcap"],
        "rope_base": ["rope_base"],
        "qk_gain_init": ["qk_gain_init", "qk_gain"],
        "logit_chunk_tokens": ["logit_chunk_tokens"],
    }
    
    for hp_attr, candidate_names in param_candidates.items():
        if not hasattr(args, hp_attr):
            continue
        # Find the first candidate that exists in gpt_params
        for gpt_param in candidate_names:
            if gpt_param in gpt_params and gpt_param not in kwargs:
                kwargs[gpt_param] = getattr(args, hp_attr)
                break

    model = module.GPT(**kwargs)
    return module, model


def model_signature(model: Any, smoke_loss: float | None = None) -> ModelSignature:
    """Get model signature, supporting both PyTorch and MLX models."""
    from falsifier.utils.framework_adapter import detect_framework, get_model_info, get_param_count

    framework = detect_framework(model)
    model_info = get_model_info(model)

    # Get trainable param count (MLX doesn't have requires_grad concept)
    if framework == "pytorch":
        trainable_count = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
        tok_emb_dim = int(model.tok_emb.embedding_dim)
        tie_embeddings = bool(model.tie_embeddings)
    else:  # mlx
        trainable_count = get_param_count(model)
        tok_emb_dim = model_info.get("embedding_dim", 0)
        tie_embeddings = getattr(model, "tie_embeddings", False)

    return ModelSignature(
        param_count=get_param_count(model),
        trainable_param_count=trainable_count,
        num_layers=model_info.get("num_layers", 0),
        model_dim=tok_emb_dim,
        num_heads=model_info.get("num_heads", 0),
        num_kv_heads=model_info.get("num_kv_heads", 0),
        tie_embeddings=tie_embeddings,
        smoke_loss=smoke_loss,
    )


def smoke_test_train_gpt(
    train_gpt_path: str | Path,
    env_overrides: dict[str, str] | None = None,
) -> ModelSignature:
    signature, _ = run_smoke_diagnostics(train_gpt_path, env_overrides=env_overrides)
    return signature


def run_smoke_diagnostics(
    train_gpt_path: str | Path,
    env_overrides: dict[str, str] | None = None,
) -> tuple[ModelSignature, SmokeDiagnostics]:
    """Run smoke diagnostics on a model, supporting both PyTorch and MLX."""
    from falsifier.utils.framework_adapter import (
        check_finite,
        create_random_input,
        create_rolled_targets,
        detect_framework,
        get_model_info,
        get_named_parameters,
        model_forward,
    )

    _, model = instantiate_minimal_model(train_gpt_path, env_overrides=env_overrides)
    framework = detect_framework(model)
    model_info = get_model_info(model)

    seq_len = 8
    vocab_size = model_info.get("vocab_size", 64)
    batch_size = 1

    # Create input using framework-agnostic functions
    input_ids = create_random_input(vocab_size, batch_size, seq_len, framework=framework)
    target_ids = create_rolled_targets(input_ids)

    # Forward pass
    loss = model_forward(model, input_ids, target_ids)

    if not check_finite(loss):
        raise ValueError("train_gpt smoke test produced a non-finite loss")

    # Backward pass and gradient check (PyTorch only - MLX uses different pattern)
    params_without_grad: list[str] = []
    if framework == "pytorch":
        import torch

        loss_mean = loss.mean()
        loss_mean.backward()
        params_without_grad = [
            name
            for name, param in get_named_parameters(model)
            if hasattr(param, "requires_grad") and param.requires_grad and param.grad is None
        ]
    # For MLX, backward is handled via value_and_grad in the training loop

    diagnostics = SmokeDiagnostics(
        forward_ok=True,
        backward_ok=len(params_without_grad) == 0,
        loss_is_finite=True,
        params_without_grad=params_without_grad,
    )
    return model_signature(model, smoke_loss=float(loss.item())), diagnostics
