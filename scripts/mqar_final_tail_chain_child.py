"""Selected UV metadata/HTTP fixture; never genuine inference or real authority."""

import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace


def metadata(function, *, bad=False, **overrides):
    import builtins

    torch = SimpleNamespace(
        __version__="2.0.0" if bad else "2.11.0+cu130",
        version=SimpleNamespace(cuda="13.0"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 91900)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_name=lambda _i: "Tesla T4",
            get_device_capability=lambda _i: (7, 5),
        ),
    )
    original = builtins.__import__

    def imported(name, *args, **kwargs):
        return torch if name == "torch" else original(name, *args, **kwargs)

    function_globals = {
        **function.__globals__,
        **overrides,
        "__builtins__": {**vars(builtins), "__import__": imported},
    }
    result = FunctionType(
        function.__code__, function_globals, function.__name__, function.__defaults__
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def metadata_records_only(source, model_config, data, output, start_task):
    """Oracle-shaped interface fixture only. No model construction/load/forward occurs."""
    import numpy as np
    from mqar_final_tail_native import FINAL, digest
    from mqar_final_tail_runner import create_json

    from compare_attention.cli import evaluate  # noqa: F401 -- source alias proof, not scoring
    from compare_attention.data.store import load_config, load_split

    output = Path(output)
    output.mkdir(exist_ok=False)
    create_json(output / "evaluation-intent.json", {"scope": "ENGINEERING_ONLY_NOT_INFERENCE"})
    identifier = start_task()
    create_json(output / "inference-task.json", {"task_id": identifier})
    config = load_config(data)
    arrays = load_split(data, "val", mmap=True)
    records = [
        {
            "row": i,
            "positions": np.flatnonzero(y != -1).tolist(),
            "targets": y[y != -1].tolist(),
            "predictions": y[y != -1].tolist(),
            "nll": [0.0] * int((y != -1).sum()),
        }
        for i, y in enumerate(arrays.labels)
    ]
    runtime = json.loads(
        (Path(os.environ["FIXTURE_INPUT"]) / "original/out/metrics.jsonl")
        .read_text()
        .splitlines()[-1]
    )["provenance"]["runtime"]
    create_json(
        output / "mqar-val.json",
        {
            "schema_version": 1,
            "loss": 0.0,
            "accuracy": 1.0,
            "accuracy_percent": 100.0,
            "correct": 64000,
            "supervised_count": 64000,
            "rows": 1000,
            "split": "val",
            "batch_size": 32,
            "device": "cuda",
            "inference_dtype": "float16",
            "parameter_dtype": "float32",
            "model_config": model_config,
            "runtime": runtime,
            "checkpoint": {
                "sha256": digest(source),
                "parameters_sha256": FINAL,
                "provenance": None,
                "format": "state_dict",
                "step": None,
            },
            "data": {
                "config": config.model_dump(mode="json"),
                "sha256": {
                    n: digest(Path(data) / n)
                    for n in ("meta.json", "val_inputs.npy", "val_labels.npy")
                },
            },
        },
    )
    create_json(output / "mqar-val-records.json", records)
    create_json(
        output / "no-optimization.json",
        {
            **dict.fromkeys(
                (
                    "optimizer_constructions",
                    "optimizer_updates",
                    "backward_calls",
                    "checkpoint_rng_restores",
                    "checkpoint_loads",
                    "training_calls",
                    "training_flops",
                ),
                0,
            ),
            "scorer_invocations": 1,
            "models_loaded": 1,
            "loaded_parameters_before": [FINAL],
            "loaded_parameters_after": [FINAL],
            "source_model_sha256": digest(source),
            "standalone_provenance": None,
        },
    )


def main():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    args = sys.argv[2:]
    release = Path(args[args.index("--release") + 1])
    work = Path(args[args.index("--work") + 1])
    work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(release / "src"))
    import mqar_final_tail_cli as worker
    import mqar_final_tail_runner as runner
    import mqar_final_tail_runtime as runtime
    from mqar_final_tail_http_fixture import FixtureHTTP
    from pytest import MonkeyPatch

    patch = MonkeyPatch()
    http = FixtureHTTP(patch, work)
    http.install()
    environment = SimpleNamespace(environ={}, path=os.path, fsync=os.fsync)
    platform = SimpleNamespace(
        system=lambda: "Linux",
        python_version=lambda: "3.13.16",
        python_implementation=lambda: "CPython",
    )
    inspect = metadata(runtime.inspect_hardware, os=environment, platform=platform)
    patch.setattr(
        runtime,
        "guard",
        metadata(
            runtime.guard,
            os=environment,
            platform=platform,
            driver_directory=lambda: "/usr/local/nvidia/lib64",
            inspect_hardware=inspect,
        ),
    )
    import torch

    def forbidden(*_args, **_kwargs):
        raise AssertionError("GENUINE_MODEL_OPTIMIZER_RNG_EXECUTION_FORBIDDEN_IN_METADATA_FIXTURE")

    patch.setattr(torch.nn.Module, "__init__", forbidden)
    patch.setattr(torch.nn.Module, "load_state_dict", forbidden)
    patch.setattr(torch.optim.Optimizer, "load_state_dict", forbidden)
    patch.setattr(torch, "set_rng_state", forbidden)
    patch.setattr(torch.cuda, "set_rng_state_all", forbidden)
    patch.setattr(runner, "execute_once", metadata_records_only)
    with contextlib.redirect_stdout(io.StringIO()):
        code = worker.main(args)
    if code:
        print((work / "failure.json").read_text())
        return code
    (work / "fixture-service-state.json").write_text(
        json.dumps(
            {
                "tasks": http.tasks,
                "files": {k: base64.b64encode(v).decode() for k, v in http.files.items()},
                "head": http.head,
            }
        )
    )
    http.save()
    print(
        json.dumps(
            {
                "scope": "ENGINEERING_METADATA_REAL_SDK_HTTP_NOT_SCIENTIFIC_INFERENCE",
                "sdk_calls": len(http.calls),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
