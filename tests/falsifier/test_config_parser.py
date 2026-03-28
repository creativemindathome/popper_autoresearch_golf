from falsifier.utils.config_parser import (
    count_parameters,
    estimate_artifact_bytes,
    extract_model_config,
)


SOURCE = """
class Hyperparameters:
    vocab_size = int(__import__("os").environ.get("VOCAB_SIZE", 1024))
    num_layers = int(__import__("os").environ.get("NUM_LAYERS", 9))
    model_dim = int(__import__("os").environ.get("MODEL_DIM", 512))
    num_heads = int(__import__("os").environ.get("NUM_HEADS", 8))
    num_kv_heads = int(__import__("os").environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(__import__("os").environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(__import__("os").environ.get("TIE_EMBEDDINGS", "1")))
    iterations = int(__import__("os").environ.get("ITERATIONS", 20000))
    train_batch_tokens = int(__import__("os").environ.get("TRAIN_BATCH_TOKENS", 524288))
    train_seq_len = int(__import__("os").environ.get("TRAIN_SEQ_LEN", 1024))
"""


def test_extract_model_config():
    config = extract_model_config(SOURCE)
    assert config["num_layers"] == 9
    assert config["model_dim"] == 512
    assert config["tie_embeddings"] is True


def test_count_parameters_has_expected_groups():
    config = extract_model_config(SOURCE)
    counts = count_parameters(config)
    assert counts["attention"] > 0
    assert counts["mlp"] > 0


def test_estimate_artifact_bytes_is_positive():
    total, remaining = estimate_artifact_bytes(SOURCE, {})
    assert total > 0
    assert remaining < 16_777_216

