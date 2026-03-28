from pathlib import Path

from falsifier.thresholds import load_stage1_thresholds


def test_load_thresholds_defaults_for_missing_profile(tmp_path):
    t = load_stage1_thresholds(tmp_path)
    assert t.source == "defaults"
    assert t.artifact_limit_bytes > 0


def test_load_thresholds_from_repo_when_profile_valid():
    root = Path(__file__).resolve().parents[2]
    profile = root / "research" / "profiles" / "latest_baseline_profile.json"
    if not profile.is_file():
        return
    t = load_stage1_thresholds(root)
    assert t.baseline_config["num_layers"] >= 1
