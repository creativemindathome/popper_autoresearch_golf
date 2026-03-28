from falsifier.utils.diff_utils import classify_diff_changes, compute_unified_diff


def test_classify_hyperparameter_change():
    old = "class Hyperparameters:\n    num_layers = 9\n"
    new = "class Hyperparameters:\n    num_layers = 10\n"
    diff = compute_unified_diff(old, new)
    kinds = classify_diff_changes(diff)
    assert "hyperparameter" in kinds


def test_classify_architecture_change():
    old = "def build():\n    return 1\n"
    new = "class NewBlock:\n    pass\n\ndef build():\n    return 1\n"
    diff = compute_unified_diff(old, new)
    kinds = classify_diff_changes(diff)
    assert "architecture" in kinds

