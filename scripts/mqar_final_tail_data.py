"""Exact authenticated immutable MQAR data pins; no arrays regenerated."""

MQAR_DATA = {
    "source_ref": "zimike/ca-kv-1c660aa5f54d88ea9e0c1ce4/1",
    "fingerprint": "1c660aa5f54d88ea9e0c1ce4eba0c25084eee164c55ab6c1d80d5db14d3e59fb",
    "spec": {
        "vocab_size": 8192,
        "input_seq_len": 512,
        "num_kv_pairs": 64,
        "power_a": 0.01,
        "n_train": 100_000,
        "n_val": 1_000,
        "n_test": 1_000,
        "seed": 1337,
    },
    "files": {
        "data-manifest.json": {
            "bytes": 1_060,
            "sha256": "dd01b61b5f219c3992c0ffe025e30ea2713d871af31c92a85d0a2c2b64cab7fe",
        },
        "meta.json": {
            "bytes": 340,
            "sha256": "30e86af308326f871097e6eb31f5d6ade6629dc5b3f4b19b078110cbe46ebfb2",
        },
        "train_inputs.npy": {
            "bytes": 409_600_128,
            "sha256": "8bc23e47e6f005ada50ebcf4eae36518045f2ba7aeb971c05c2c1cfc82e5881c",
        },
        "train_labels.npy": {
            "bytes": 409_600_128,
            "sha256": "50f189f846c7acfe2dc4bc825f573d87a126e613f0c941fbc72e1d8b60882609",
        },
        "val_inputs.npy": {
            "bytes": 4_096_128,
            "sha256": "e7b7d1e4a89f921ca8ae1728c9588dd79507ab977bbf267e639d752155e6f876",
        },
        "val_labels.npy": {
            "bytes": 4_096_128,
            "sha256": "ad2f06cdce49813c612f4c3c813cdea49a4cb539f649168596186be38e2debfa",
        },
        "test_inputs.npy": {
            "bytes": 4_096_128,
            "sha256": "b5a12ac96a6ad295db47916365925d3ce15badcd865980ca1ee47bb0595283d5",
        },
        "test_labels.npy": {
            "bytes": 4_096_128,
            "sha256": "12d7e1e6f8a0f0651a896d6861880f43dd6aa0615ca7bca8866048a6fb26d123",
        },
    },
}


def verify_files(root):
    """Read/hash existing input bytes only; never regenerate or copy arrays."""
    from pathlib import Path

    from mqar_final_tail_native import digest, require

    root = Path(root)
    for name, pin in MQAR_DATA["files"].items():
        path = root / name
        require(
            path.is_file()
            and path.stat().st_size == pin["bytes"]
            and digest(path) == pin["sha256"],
            "ALL_IMMUTABLE_MQAR_DATA_BEFORE_CREDENTIALS",
        )
    audit_val_bank(root)
    return MQAR_DATA


def audit_val_bank(root):
    """Validate every fixed VAL label from its designated KV prefix; not predictions."""
    from pathlib import Path

    import numpy as np
    from mqar_final_tail_native import require

    root = Path(root)
    inputs = np.load(root / "val_inputs.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(root / "val_labels.npy", mmap_mode="r", allow_pickle=False)
    spec = MQAR_DATA["spec"]
    require(
        inputs.dtype == labels.dtype == np.int64
        and inputs.shape == labels.shape == (spec["n_val"], spec["input_seq_len"]),
        "FIXED_VAL_SHAPE",
    )
    pairs = spec["num_kv_pairs"]
    for x, y in zip(inputs, labels, strict=True):
        lookup = dict(zip(x[: 2 * pairs : 2].tolist(), x[1 : 2 * pairs : 2].tolist(), strict=True))
        positions = np.flatnonzero(y != -1)
        require(
            len(lookup) == len(positions) == pairs
            and bool((positions >= 2 * pairs).all())
            and len(set(x[positions].tolist())) == pairs
            and all(lookup.get(int(x[p])) == int(y[p]) for p in positions),
            "ALL_VAL_TARGETS_DESIGNATED_PREFIX_ORACLE",
        )
    return {
        "rows": len(inputs),
        "queries": len(inputs) * pairs,
        "model_forwards": 0,
        "scope": "LABEL_INTEGRITY_ONLY_NOT_MODEL_PREDICTIONS",
    }
