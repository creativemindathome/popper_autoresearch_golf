from pathlib import Path

from falsifier.mechanism_probes import run_readonly_mechanism_probes


def test_readonly_mechanism_probe_bundle():
    root = Path(__file__).resolve().parents[2]
    out = run_readonly_mechanism_probes(root)
    assert out["readonly"] is True
    assert out["schema_version"] == "1"
    assert out["param_tensor_count"] > 0
