"""Retain native FULL predicates while replacing host scientific constructors."""

import json
from pathlib import Path


def integrate(package, original):
    from build_seed202_mqar_continuation import replace_once, sha

    if sha(original) != "e2289ced84240f1f81dfcaa254741457a7863ae9a40db6ab68f9cc021e904542":
        raise ValueError("ORIGINAL_SCHEMA_AUTHENTICATION")
    import torch

    payload = torch.load(original, weights_only=True, map_location="cpu")
    blueprint = {
        "model": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in payload["model"].items()
        },
        "model_structure": payload["model_structure"],
        "optimizer_parameters": payload["optimizer_parameters"],
        "optimizer_group": payload["optimizer"]["param_groups"][0],
    }
    package = Path(package)
    (package / "mqar-tensor-schema.json").write_text(
        json.dumps(blueprint, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    helper = Path(__file__).with_name("seed202_mqar_tensor_schema.py").read_bytes()
    (package / "mqar_tensor_schema.py").write_bytes(helper)
    source = (package / "cloud_native.py").read_text(encoding="utf-8")
    start = source.index('    with torch.random.fork_rng(devices=[]), torch.device("cpu"):')
    end = source.index("    if target:\n        validate_recipe", start)
    source = (
        source[:start]
        + (
            "    from seed202_pilot.mqar_tensor_schema import validate_model_optimizer\n\n"
            "    final = validate_model_optimizer(payload)\n"
        )
        + source[end:]
    )
    source = replace_once(
        source,
        '    model.load_state_dict(payload["model"], strict=True)\n'
        "    final = parameter_fingerprint(model)\n",
        "",
    )
    for line in (
        "    from compare_attention.models.config import GPT2Config\n",
        "    from compare_attention.models.gpt2 import GPT2\n",
        "    from compare_attention.train import checkpoint as cp\n",
        "    from compare_attention.train.checkpoint_optimizer import _validate_optimizer\n",
        "    from compare_attention.train.identity import parameter_fingerprint\n",
    ):
        # verify_shared still needs its own unchanged cp import.
        source = source.replace(line, "", 1)
    source = source.replace("    import torch\n", "", 1)
    compile(source, "mqar_host_native.py", "exec")
    (package / "mqar_host_native.py").write_text(source, encoding="utf-8", newline="\n")
    for name, before, after in (
        (
            "science",
            "        from seed202_pilot.cloud_native import bind_native",
            "        from seed202_pilot.cloud_native import bind_native\n"
            '        if case["id"] == "mqar-k64-classic-seed202":\n'
            "            from seed202_pilot.mqar_host_native import bind_native",
        ),
        (
            "periodic",
            "    from seed202_pilot.cloud_native import validate_full",
            "    from seed202_pilot.cloud_native import validate_full\n"
            '    if case["id"] == "mqar-k64-classic-seed202":\n'
            "        from seed202_pilot.mqar_host_native import validate_full",
        ),
    ):
        path = package / (name + ".py")
        text = replace_once(path.read_text(encoding="utf-8"), before, after)
        path.write_text(text, encoding="utf-8", newline="\n")
