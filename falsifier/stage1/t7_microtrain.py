"""T7: Micro-Training Gate (100 Steps) - Enhanced with Learning Curve Analysis.

Validates theories via 100-step training with:
- MLX on Apple Silicon (preferred)
- PyTorch minimal-env fallback

FATAL conditions:
- Loss[99] > Loss[0] (diverging)
- Learning ratio < calibration.baseline_100.learning_ratio_kill
- Max gradient norm > 100 (explosion)
- Projected full run > 720s

TAGs issued:
- T7_slow_learning: learning_ratio < learning_ratio_tag
- T7_gradient_instability: CV of gradient norms > 0.5
- T7_component_speed_imbalance: max(comp_grads)/min(comp_grads) > 10
- T7_entropy_collapse: entropy drops by >50%
- T7_low_throughput: throughput < baseline/2
- T7_learning_plateau: curve_shape == "plateau"
- T7_divergent_training: curve_shape == "divergent"
- T7_high_variance: curve_shape == "noisy"
- T7_component_imbalance: one component learns much slower than others
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import torch

from falsifier.adapters.mlx_adapter import (
    mlx_available,
    run_mlx_micro_train_summary,
)
from falsifier.adapters.parameter_golf import instantiate_minimal_model
from falsifier.types import (
    Baseline100,
    Calibration,
    FalsifierInput,
    T7Result,
    Tag,
    TestStatus,
)
from falsifier.utils.metrics import classify_component

T7_TEST_ID = "T7"


def _run_pytorch_microtrain(
    train_gpt_path: Path,
    config_overrides: dict[str, Any] | None,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    """Run PyTorch micro-training as fallback with per-step component tracking."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    module, model = instantiate_minimal_model(train_gpt_path, env_overrides=config_overrides)
    device = torch.device("cpu")
    model = model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    seq_len = 8
    vocab = int(model.tok_emb.num_embeddings)

    loss_trajectory: list[float] = []
    gradient_norms: list[float] = []
    entropy_trajectory: list[float] = []
    component_speeds: dict[str, list[float]] = {k: [] for k in ["attn", "mlp", "embed", "skip"]}
    
    # Per-step component gradient tracking for learning order analysis
    component_grad_norms_per_step: dict[str, list[float]] = {
        "attn": [], "mlp": [], "embed": [], "norm": [], "other": []
    }

    t0 = time.perf_counter()

    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        # Synthetic random data
        x = torch.randint(0, vocab, (1, seq_len), device=device)
        y = torch.roll(x, shifts=-1, dims=1)

        # Forward
        loss = model(x, y)
        loss_value = float(loss.detach().item())

        # Backward
        loss.backward()

        # Compute gradient norm
        total_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_sq += float((param.grad * param.grad).sum().item())
        grad_norm = total_sq ** 0.5
        gradient_norms.append(grad_norm)

        # Track component speeds (aggregate)
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_sq = float((param.grad * param.grad).sum().item())
                if "attn" in name:
                    component_speeds["attn"].append(grad_sq ** 0.5)
                elif "mlp" in name:
                    component_speeds["mlp"].append(grad_sq ** 0.5)
                elif "tok_emb" in name:
                    component_speeds["embed"].append(grad_sq ** 0.5)
                elif "skip" in name:
                    component_speeds["skip"].append(grad_sq ** 0.5)
        
        # Track per-component gradient norms per step for learning order analysis
        step_component_norms: dict[str, float] = {"attn": 0.0, "mlp": 0.0, "embed": 0.0, "norm": 0.0, "other": 0.0}
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = float(param.grad.norm(2).item())
                comp_type = classify_component(name)
                step_component_norms[comp_type] += param_norm
        
        for comp_type, norm_val in step_component_norms.items():
            component_grad_norms_per_step[comp_type].append(norm_val)

        # Calculate entropy every 10 steps
        if step % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                mean_entropy = float(entropy.mean().item())
                entropy_trajectory.append(mean_entropy)

        opt.step()
        loss_trajectory.append(loss_value)

    elapsed = time.perf_counter() - t0

    # Aggregate component speeds
    agg_component_speeds: dict[str, float] = {}
    for k, v in component_speeds.items():
        if v:
            agg_component_speeds[k] = sum(v) / len(v)
        else:
            agg_component_speeds[k] = 0.0

    # Calculate throughput (tokens/sec)
    total_tokens = steps * seq_len
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "loss_trajectory": loss_trajectory,
        "loss_first": loss_trajectory[0] if loss_trajectory else 0.0,
        "loss_last": loss_trajectory[-1] if loss_trajectory else 0.0,
        "loss_drop": (loss_trajectory[0] - loss_trajectory[-1]) if loss_trajectory else 0.0,
        "throughput": throughput,
        "gradient_norms": gradient_norms,
        "entropy_trajectory": entropy_trajectory,
        "component_speeds": agg_component_speeds,
        "component_grad_norms_per_step": component_grad_norms_per_step,
        "elapsed_seconds": elapsed,
    }


def _compute_gradient_stability(gradient_norms: list[float]) -> float:
    """Compute coefficient of variation (CV) of gradient norms."""
    if len(gradient_norms) < 2:
        return 0.0
    mean_val = statistics.mean(gradient_norms)
    if mean_val == 0:
        return 0.0
    try:
        std_val = statistics.stdev(gradient_norms)
        return std_val / mean_val
    except statistics.StatisticsError:
        return 0.0


def _compute_component_speed_ratio(component_speeds: dict[str, float]) -> float:
    """Compute max/min ratio of component speeds."""
    values = [v for v in component_speeds.values() if v > 0]
    if len(values) < 2:
        return 1.0
    return max(values) / min(values)


def _check_entropy_collapse(entropy_trajectory: list[float]) -> bool:
    """Check if entropy dropped by >50%."""
    if len(entropy_trajectory) < 2:
        return False
    first = entropy_trajectory[0]
    last = entropy_trajectory[-1]
    if first <= 0:
        return False
    return (first - last) / first > 0.5


def _analyze_learning_curve_shape(loss_trajectory: list[float]) -> str:
    """Analyze the shape of the learning curve.
    
    Returns one of: "monotonic", "plateau", "sawtooth", "divergent", "noisy"
    """
    if len(loss_trajectory) < 10:
        return "unknown"
    
    # Check for divergence (loss increasing overall)
    first_half_mean = statistics.mean(loss_trajectory[:len(loss_trajectory)//2])
    second_half_mean = statistics.mean(loss_trajectory[len(loss_trajectory)//2:])
    
    if second_half_mean > first_half_mean * 1.05:  # 5% tolerance
        return "divergent"
    
    # Check for monotonic (consistently decreasing)
    increases = sum(1 for i in range(1, len(loss_trajectory)) if loss_trajectory[i] > loss_trajectory[i-1])
    decrease_ratio = 1 - (increases / (len(loss_trajectory) - 1))
    
    if decrease_ratio > 0.95:
        return "monotonic"
    
    # Check for plateau (flat for >20 steps)
    window_size = 20
    for i in range(len(loss_trajectory) - window_size):
        window = loss_trajectory[i:i+window_size]
        window_std = statistics.stdev(window) if len(window) > 1 else 0
        window_range = max(window) - min(window)
        if window_range < 0.1:  # Less than 0.1 loss change in 20 steps
            return "plateau"
    
    # Check for sawtooth (alternating up/down pattern)
    direction_changes = 0
    last_direction = 0  # 1 = up, -1 = down
    for i in range(1, len(loss_trajectory)):
        current_direction = 1 if loss_trajectory[i] > loss_trajectory[i-1] else -1
        if current_direction != last_direction and last_direction != 0:
            direction_changes += 1
        last_direction = current_direction
    
    if direction_changes > len(loss_trajectory) * 0.3:  # >30% direction changes
        return "sawtooth"
    
    # Check for high variance/noisy
    try:
        loss_std = statistics.stdev(loss_trajectory)
        loss_mean = statistics.mean(loss_trajectory)
        cv = loss_std / loss_mean if loss_mean > 0 else 0
        if cv > 0.3:  # Coefficient of variation > 30%
            return "noisy"
    except statistics.StatisticsError:
        pass
    
    return "monotonic"  # Default fallback


def _analyze_convergence_trajectory(
    loss_trajectory: list[float], 
    gradient_norms: list[float]
) -> str:
    """Analyze convergence trajectory behavior.
    
    Returns one of: "stable", "oscillating", "unstable", "improving", "degrading"
    """
    if len(loss_trajectory) < 10:
        return "unknown"
    
    # Split into early and late phases
    mid = len(loss_trajectory) // 2
    early_loss = statistics.mean(loss_trajectory[:mid])
    late_loss = statistics.mean(loss_trajectory[mid:])
    
    # Check gradient stability in late phase
    late_grads = gradient_norms[mid:] if len(gradient_norms) >= mid else gradient_norms
    grad_cv = _compute_gradient_stability(late_grads) if late_grads else 0
    
    # Check loss variance in late phase
    try:
        late_variance = statistics.variance(loss_trajectory[mid:]) if len(loss_trajectory[mid:]) > 1 else 0
    except statistics.StatisticsError:
        late_variance = 0
    
    if grad_cv > 1.0 and late_variance > 1.0:
        return "unstable"
    elif grad_cv > 0.5:
        return "oscillating"
    elif late_loss < early_loss * 0.8:  # Significant improvement
        return "improving"
    elif late_loss > early_loss * 1.05:
        return "degrading"
    else:
        return "stable"


def _compute_component_learning_order(
    component_grad_norms_per_step: dict[str, list[float]],
    threshold_ratio: float = 0.5
) -> dict[str, int]:
    """Compute which component learned first (first significant gradient drop).
    
    Returns dict mapping component name to first step where its gradient norm
    dropped below threshold_ratio of initial value.
    """
    learning_order: dict[str, int] = {}
    
    for comp_type, norms in component_grad_norms_per_step.items():
        if not norms or len(norms) < 2:
            continue
        
        initial_norm = norms[0]
        if initial_norm < 1e-6:  # Skip near-zero gradients
            continue
            
        threshold = initial_norm * threshold_ratio
        
        # Find first step where norm drops below threshold
        for step, norm in enumerate(norms):
            if norm < threshold:
                learning_order[comp_type] = step
                break
        else:
            # Never dropped below threshold
            learning_order[comp_type] = -1
    
    return learning_order


def _detect_stepwise_instability(gradient_norms: list[float], threshold: float = 3.0) -> list[int]:
    """Detect steps where gradient spiked > threshold x mean.
    
    Returns list of step indices where instability occurred.
    """
    if len(gradient_norms) < 5:
        return []
    
    mean_grad = statistics.mean(gradient_norms)
    if mean_grad < 1e-6:
        return []
    
    unstable_steps: list[int] = []
    for step, norm in enumerate(gradient_norms):
        if norm > mean_grad * threshold:
            unstable_steps.append(step)
    
    return unstable_steps


def _compute_projected_convergence(
    loss_trajectory: list[float],
    target_loss: float,
    baseline_loss_at_100: float
) -> tuple[int | None, float | None]:
    """Extrapolate when loss would reach baseline and final projected loss.
    
    Uses linear extrapolation of late-phase learning rate.
    
    Returns:
        (projected_convergence_step, projected_final_loss)
    """
    if len(loss_trajectory) < 20:
        return None, None
    
    current_loss = loss_trajectory[-1]
    
    # If already at or below target
    if current_loss <= target_loss:
        return 100, current_loss
    
    # Compute recent learning rate (steps 80-100)
    late_phase = loss_trajectory[80:]
    if len(late_phase) < 2:
        return None, None
    
    # Average loss drop per step in late phase
    late_drops = [late_phase[i-1] - late_phase[i] for i in range(1, len(late_phase))]
    avg_drop_per_step = statistics.mean(late_drops) if late_drops else 0
    
    if avg_drop_per_step <= 1e-6:
        # Not converging
        return None, current_loss
    
    # Project steps needed to reach target
    remaining_drop = current_loss - target_loss
    steps_needed = int(remaining_drop / avg_drop_per_step)
    projected_step = 100 + steps_needed
    
    # Project final loss assuming exponential decay tails off
    # Use heuristic: 90% of remaining drop achieved by 2x the steps
    projected_final = target_loss + (remaining_drop * 0.1)
    
    return projected_step, projected_final


def _compute_component_imbalance_score(
    component_learning_order: dict[str, int]
) -> tuple[float, str | None]:
    """Compute component imbalance score and identify slowest component.
    
    Returns (ratio, slowest_component_name) or (1.0, None) if balanced.
    """
    valid_entries = {k: v for k, v in component_learning_order.items() if v >= 0}
    
    if len(valid_entries) < 2:
        return 1.0, None
    
    # Filter to main components only
    main_components = {k: v for k, v in valid_entries.items() if k in ["attn", "mlp", "embed"]}
    if len(main_components) < 2:
        return 1.0, None
    
    steps_list = list(main_components.values())
    max_steps = max(steps_list)
    min_steps = min(steps_list)
    
    if min_steps == 0:
        min_steps = 1  # Avoid division by zero
    
    ratio = max_steps / min_steps
    
    # Find the slowest component
    slowest = max(main_components.keys(), key=lambda k: main_components[k])
    
    return ratio, slowest


def run_t7(
    inp: FalsifierInput,
    steps: int = 100,
    seed: int = 42,
) -> T7Result:
    """Execute T7 micro-training gate with enhanced learning curve analysis.

    Args:
        inp: FalsifierInput containing theory, calibration, and paths
        steps: Number of training steps (default 100)
        seed: Random seed for reproducibility

    Returns:
        T7Result with status, metrics, learning curve analysis, and tags
    """
    t0 = time.perf_counter()

    # Resolve paths
    repo_root = Path(__file__).resolve().parents[2]
    train_gpt_path = Path(inp.train_gpt_path) if inp.train_gpt_path else repo_root / "train_gpt.py"
    train_gpt_mlx_path = repo_root / "train_gpt_mlx.py"

    # Resolve val_data_path
    val_data_path = None
    if inp.val_data_path:
        val_data_path = Path(inp.val_data_path)
        if not val_data_path.exists():
            val_data_path = None

    # Get calibration
    calibration = inp.calibration
    baseline_100 = calibration.baseline_100 if calibration else Baseline100()

    # Determine backend: MLX first, fallback to PyTorch
    use_mlx = False
    train_result: dict[str, Any] | None = None

    if mlx_available() and train_gpt_mlx_path.exists():
        try:
            # Build config overrides from config_delta
            config_overrides: dict[str, Any] = {}
            if inp.config_delta:
                for key, value in inp.config_delta.items():
                    # Map common config keys to env var format
                    env_key = key.upper()
                    config_overrides[env_key] = str(value)

            train_result = run_mlx_micro_train_summary(
                repo_root,
                config_overrides=config_overrides,
                steps=steps,
                seed=seed,
            )
            use_mlx = True
        except Exception as e:
            # MLX failed, will fallback to PyTorch
            train_result = None

    # Fallback to PyTorch
    if train_result is None:
        try:
            config_overrides: dict[str, Any] = {}
            if inp.config_delta:
                config_overrides = {k.upper(): str(v) for k, v in inp.config_delta.items()}

            train_result = _run_pytorch_microtrain(
                train_gpt_path,
                config_overrides,
                steps,
                seed,
            )
        except Exception as e:
            # Training failed completely
            return T7Result(
                status="FAIL_FATAL",
                test_id=T7_TEST_ID,
                kill_reason=f"Training failed: {e}",
                wall_seconds=time.perf_counter() - t0,
            )

    # Extract metrics
    loss_trajectory = train_result.get("loss_trajectory", [])
    loss_first = float(train_result.get("loss_first", 0.0))
    loss_last = float(train_result.get("loss_last", 0.0))
    loss_drop = float(train_result.get("loss_drop", 0.0))
    throughput = float(train_result.get("throughput", 0.0))
    gradient_norms = train_result.get("gradient_norms", [])
    entropy_trajectory = train_result.get("entropy_trajectory", [])
    component_speeds = train_result.get("component_speeds", {})
    component_grad_norms_per_step = train_result.get("component_grad_norms_per_step", {})
    elapsed = float(train_result.get("elapsed_seconds", 0.0))

    # Calculate derived metrics
    learning_ratio = 0.0
    if baseline_100.loss_drop_mean > 0:
        learning_ratio = loss_drop / baseline_100.loss_drop_mean

    gradient_stability = _compute_gradient_stability(gradient_norms)
    component_speed_ratio = _compute_component_speed_ratio(component_speeds)

    # Project full run time (assuming 20_000 steps)
    full_run_steps = 20_000
    projected_full_run_seconds = (elapsed / steps) * full_run_steps

    # ═══════════════════════════════════════════════════════════════════════════
    # Enhanced Learning Curve Analysis
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Learning curve shape
    curve_shape = _analyze_learning_curve_shape(loss_trajectory)
    
    # Convergence trajectory
    conv_trajectory = _analyze_convergence_trajectory(loss_trajectory, gradient_norms)
    
    # Component learning order
    component_learning_order = _compute_component_learning_order(component_grad_norms_per_step)
    
    # Stepwise instability detection
    instability_flags = _detect_stepwise_instability(gradient_norms, threshold=3.0)
    
    # Projected convergence metrics
    target_loss = baseline_100.loss_at_100_mean if baseline_100.loss_at_100_mean > 0 else loss_first * 0.5
    proj_conv_step, proj_final_loss = _compute_projected_convergence(
        loss_trajectory, target_loss, baseline_100.loss_at_100_mean
    )
    
    # Component imbalance detection
    imbalance_ratio, slowest_component = _compute_component_imbalance_score(component_learning_order)

    # ═══════════════════════════════════════════════════════════════════════════
    # FATAL Checks with Enhanced Kill Reasons
    # ═══════════════════════════════════════════════════════════════════════════
    
    kill_reason: str | None = None
    status: TestStatus = "PASS"
    tags: list[Tag] = []

    # FATAL: Diverging loss (using curve shape for more context)
    if loss_last > loss_first:
        status = "FAIL_FATAL"
        diverging_step = None
        for i in range(1, len(loss_trajectory)):
            if loss_trajectory[i] > loss_first * 1.1:  # 10% above initial
                diverging_step = i
                break
        
        if diverging_step:
            kill_reason = f"Training diverged after step {diverging_step} (loss: {loss_first:.2f} → {loss_last:.2f})"
        else:
            kill_reason = f"Loss diverging: loss[99]={loss_last:.4f} > loss[0]={loss_first:.4f}"

    # FATAL: Learning ratio below kill threshold
    elif learning_ratio < baseline_100.learning_ratio_kill:
        status = "FAIL_FATAL"
        kill_reason = (
            f"Learning ratio {learning_ratio:.3f} < kill threshold {baseline_100.learning_ratio_kill:.3f} "
            f"({loss_drop:.3f} vs baseline {baseline_100.loss_drop_mean:.3f} drop)"
        )

    # FATAL: Gradient explosion
    elif gradient_norms and max(gradient_norms) > 100.0:
        max_grad = max(gradient_norms)
        max_grad_step = gradient_norms.index(max_grad)
        status = "FAIL_FATAL"
        kill_reason = f"Gradient explosion at step {max_grad_step}: max norm {max_grad:.2f} > 100"

    # FATAL: Projected run too slow
    elif projected_full_run_seconds > 720.0:
        status = "FAIL_FATAL"
        kill_reason = f"Projected full run too slow: {projected_full_run_seconds:.0f}s > 720s limit"

    # ═══════════════════════════════════════════════════════════════════════════
    # TAG Checks (only if not already fatal)
    # ═══════════════════════════════════════════════════════════════════════════
    
    if status == "PASS":
        # TAG: Slow learning
        if learning_ratio < baseline_100.learning_ratio_tag:
            tags.append(
                Tag(
                    tag_id="T7_slow_learning",
                    test_id=T7_TEST_ID,
                    category="speed_pathology",
                    description=f"Learning ratio {learning_ratio:.3f} below tag threshold {baseline_100.learning_ratio_tag:.3f}",
                )
            )

        # TAG: Gradient instability (CV > 0.5)
        if gradient_stability > 0.5:
            tags.append(
                Tag(
                    tag_id="T7_gradient_instability",
                    test_id=T7_TEST_ID,
                    category="gradient_pathology",
                    description=f"Gradient instability (CV={gradient_stability:.3f}) > 0.5",
                )
            )

        # TAG: Component speed imbalance (ratio > 10)
        if component_speed_ratio > 10.0:
            tags.append(
                Tag(
                    tag_id="T7_component_speed_imbalance",
                    test_id=T7_TEST_ID,
                    category="capacity_pathology",
                    description=f"Component speed imbalance: max/min ratio {component_speed_ratio:.2f} > 10",
                )
            )

        # TAG: Entropy collapse (>50% drop)
        if _check_entropy_collapse(entropy_trajectory):
            if entropy_trajectory:
                drop_pct = 100 * (entropy_trajectory[0] - entropy_trajectory[-1]) / entropy_trajectory[0]
                tags.append(
                    Tag(
                        tag_id="T7_entropy_collapse",
                        test_id=T7_TEST_ID,
                        category="entropy_pathology",
                        description=f"Entropy collapsed by {drop_pct:.1f}%",
                    )
                )

        # TAG: Low throughput (< baseline/2)
        if baseline_100.tokens_per_second_mean > 0 and throughput < baseline_100.tokens_per_second_mean / 2:
            tags.append(
                Tag(
                    tag_id="T7_low_throughput",
                    test_id=T7_TEST_ID,
                    category="speed_pathology",
                    description=f"Throughput {throughput:.0f} tokens/s < half baseline {baseline_100.tokens_per_second_mean / 2:.0f}",
                )
            )
        
        # ═════════════════════════════════════════════════════════════════════════
        # New Enhanced TAGs
        # ═════════════════════════════════════════════════════════════════════════
        
        # TAG: Learning plateau
        if curve_shape == "plateau":
            # Find plateau start
            plateau_start = None
            for i in range(len(loss_trajectory) - 20):
                window = loss_trajectory[i:i+20]
                if max(window) - min(window) < 0.1:
                    plateau_start = i
                    break
            
            plateau_msg = f" at step {plateau_start}" if plateau_start else ""
            tags.append(
                Tag(
                    tag_id="T7_learning_plateau",
                    test_id=T7_TEST_ID,
                    category="speed_pathology",
                    description=f"Learning plateaued{plateau_msg} (loss: {loss_last:.2f} for 20+ steps)",
                )
            )
        
        # TAG: Divergent training (curve shape detected divergent pattern)
        if curve_shape == "divergent":
            diverging_step = None
            for i in range(1, len(loss_trajectory)):
                if loss_trajectory[i] > loss_first * 1.05:
                    diverging_step = i
                    break
            
            diverge_msg = f" after step {diverging_step}" if diverging_step else ""
            tags.append(
                Tag(
                    tag_id="T7_divergent_training",
                    test_id=T7_TEST_ID,
                    category="stability_pathology",
                    description=f"Training shows divergent pattern{diverge_msg} (loss: {loss_first:.2f} → {loss_last:.2f})",
                )
            )
        
        # TAG: High variance / noisy training
        if curve_shape == "noisy":
            try:
                loss_std = statistics.stdev(loss_trajectory)
                loss_mean = statistics.mean(loss_trajectory)
                cv = loss_std / loss_mean if loss_mean > 0 else 0
            except:
                cv = 0
            
            tags.append(
                Tag(
                    tag_id="T7_high_variance",
                    test_id=T7_TEST_ID,
                    category="stability_pathology",
                    description=f"High variance training (CV={cv:.2f}) with noisy loss trajectory",
                )
            )
        
        # TAG: Component imbalance (one component learns much slower)
        if imbalance_ratio > 5.0 and slowest_component:
            slow_step = component_learning_order.get(slowest_component, -1)
            other_steps = [v for k, v in component_learning_order.items() 
                          if k != slowest_component and k in ["attn", "mlp", "embed"] and v >= 0]
            avg_other = statistics.mean(other_steps) if other_steps else 0
            
            if slow_step > 0 and avg_other > 0:
                ratio = slow_step / avg_other
                tags.append(
                    Tag(
                        tag_id="T7_component_imbalance",
                        test_id=T7_TEST_ID,
                        category="capacity_pathology",
                        description=f"{slowest_component.upper()} gradients {ratio:.1f}x slower than others (step {slow_step} vs avg {avg_other:.0f})",
                    )
                )

    wall_seconds = time.perf_counter() - t0

    # Determine framework used
    framework = "mlx" if use_mlx else "pytorch"

    return T7Result(
        status=status,
        test_id=T7_TEST_ID,
        framework=framework,
        loss_trajectory=loss_trajectory,
        loss_at_1=loss_first,
        loss_at_100=loss_last,
        loss_drop=loss_drop,
        learning_ratio=learning_ratio,
        gradient_norms=gradient_norms,
        gradient_mean=statistics.mean(gradient_norms) if gradient_norms else 0.0,
        gradient_stability=gradient_stability,
        entropy_trajectory=entropy_trajectory,
        component_speeds=component_speeds,
        component_speed_ratio=component_speed_ratio,
        tokens_per_second=throughput,
        projected_full_run_seconds=projected_full_run_seconds,
        kill_reason=kill_reason,
        tags=tags,
        wall_seconds=wall_seconds,
        # Enhanced fields
        learning_curve_shape=curve_shape,
        convergence_trajectory=conv_trajectory,
        component_learning_order=component_learning_order,
        stepwise_instability_flags=instability_flags,
        projected_convergence_step=proj_conv_step,
        projected_final_loss=proj_final_loss,
        component_grad_norms_per_step=component_grad_norms_per_step,
    )
