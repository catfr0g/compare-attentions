"""Общие маленькие конфиги kv_retrieval для тестов."""

from compare_attention.data.config import KVRetrievalConfig

_BASE = {
    "vocab_size": 32,
    "input_seq_len": 16,
    "num_kv_pairs": 2,
    "power_a": 0.01,
    "seed": 0,
    "n_train": 4,
    "n_val": 3,
    "n_test": 2,
}


def config(**overrides: object) -> KVRetrievalConfig:
    params = {**_BASE, **overrides}
    return KVRetrievalConfig(**params)
