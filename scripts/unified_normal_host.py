"""Read-only NORMAL host consumers; no provider writes, optimizer or model restore."""

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
import zipfile
from pathlib import Path, PurePosixPath


def require(condition, code):
    if not condition:
        raise ValueError(code)


def read(path):
    return json.loads(Path(path).read_bytes())


@contextlib.contextmanager
def sources(package):
    """Bind the reviewed wheel and helper roots in a fresh observer process."""
    package = Path(package).resolve()
    require(
        not any(
            n == "compare_attention" or n.startswith("compare_attention.") for n in sys.modules
        ),
        "UNIFIED_HOST_FRESH_IMPORT_REQUIRED",
    )
    sys.path[:0] = [
        str(package / "src"),
        str(package / "inherited/operator"),
        str(package / "inherited/operator/inherited"),
        str(package / "inherited/tools"),
        str(package / "inherited-source"),
    ]
    from normal_training_plan import source_identity

    identity = source_identity(package)
    root = Path(os.environ["TMPDIR"]) / ("normal-host-wheel-" + uuid.uuid4().hex)
    with zipfile.ZipFile(package / "compare_attention-0.1.0-py3-none-any.whl") as saved:
        names = saved.namelist()
        require(len(names) == len(set(names)), "UNIFIED_HOST_UNIQUE_WHEEL")
        for name in names:
            part = PurePosixPath(name)
            require(
                not part.is_absolute()
                and ".." not in part.parts
                and "\\" not in name
                and ":" not in name,
                "UNIFIED_HOST_SAFE_WHEEL",
            )
        saved.extractall(root)
    sys.path.insert(0, str(root))
    try:
        yield identity
    finally:
        shutil.rmtree(root)


def declared_data_domains(case, nonce=None):
    """Expected mount domains come from the saved main, never a downloaded path."""
    if case["engineering_only"]:
        return (case["config"]["data"],)
    if case["task"] in {"tiny_shakespeare_char", "ruler_variable_tracking"}:
        domain = "tiny" if case["task"] == "tiny_shakespeare_char" else "ruler"
        require(
            type(nonce) is str and re.fullmatch(r"[0-9a-f]{32}", nonce) is not None,
            "UNIFIED_HOST_" + domain.upper() + "_NONCE",
        )
        return (f"/kaggle/temp/normal-{nonce}/data/{domain}",)
    if case["task"] == "sorting":
        return ("/kaggle/temp/sorting-science-fcff1b5daf574b70/data/sorting",)
    require(case["task"] == "kv_retrieval", "UNIFIED_HOST_TASK")
    return (
        "/kaggle/input/datasets/zimike/ca-kv-1c660aa5f54d88ea9e0c1ce4",
        "/kaggle/input/ca-kv-1c660aa5f54d88ea9e0c1ce4",
    )


def task_quality(package, root, case, data):
    """Use the unchanged task validators, never a downloaded evidence-policy choice."""
    from normal_training_plan import canonical

    root = Path(root)
    if case.get("task") == "tiny_shakespeare_char" and not case["engineering_only"]:
        from normal_training_lm import accept_cached

        return accept_cached(package, root, case, data)
    if case.get("task") == "ruler_variable_tracking" and not case["engineering_only"]:
        from normal_training_tracking import accept_cached

        return accept_cached(package, root, case, data)
    if case["engineering_only"]:
        note = read(root / "task-engineering.json")
        require(
            note["case"] == case
            and note["scientific_acceptance"] is False
            and note["task_evaluation_executed"] is False,
            "UNIFIED_ENGINEERING_TASK_SCOPE",
        )
        return {"scope": "ENGINEERING_NOT_TASK_QUALITY", "next_launch_authorized": False}
    import numpy as np
    from seed202_pilot.contract import require_authorized_case
    from seed202_pilot.metrics import verify_mqar_report, verify_sorting_report

    spec = require_authorized_case(case["id"])
    report = read(root / "task-val.json")
    identity = read(root / "out/run_identity.json")
    require(spec.family == case["family"] and spec.seed == case["seed"], "UNIFIED_TASK_CASE")
    if case["task"] == "kv_retrieval":
        labels = np.load(Path(data) / "val_labels.npy", mmap_mode="r")
        verify_mqar_report(
            report, spec, identity, required_scope="TARGET_GPU_RESULT", labels=labels
        )
    else:
        verify_sorting_report(report, spec, identity, data, required_scope="TARGET_GPU_RESULT")
    source = read(root / "task-source.json")
    wheel = Path(package) / "compare_attention-0.1.0-py3-none-any.whl"
    require(
        source["no_training_method_replacement"] is True
        and source["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "UNIFIED_TASK_SOURCE",
    )
    require(
        set(source) == {"wheel_sha256", "imported_modules", "no_training_method_replacement"}
        and {
            "compare_attention.evaluation_checkpoint",
            "compare_attention.train.loop",
            "compare_attention.train.distributed_checkpoint",
            "compare_attention.train.lifecycle",
            "compare_attention.models." + case["family"],
        }
        <= set(source["imported_modules"]),
        "UNIFIED_TASK_SOURCE_SCHEMA",
    )
    with zipfile.ZipFile(wheel) as saved:
        for name, row in source["imported_modules"].items():
            base = name.replace(".", "/")
            require(
                (name == "compare_attention" or name.startswith("compare_attention."))
                and row["member"] in {base + ".py", base + "/__init__.py"}
                and row["sha256"] == hashlib.sha256(saved.read(row["member"])).hexdigest(),
                "UNIFIED_TASK_IMPORTED_MODULE",
            )
    return {
        "status": "ACCEPTED",
        "case": case,
        "final_sha256": identity["final_parameters_sha256"],
        "report_sha256": hashlib.sha256(canonical(report).encode()).hexdigest(),
        "next_launch_authorized": False,
    }


def inference_readback(root, case, values):
    """Get the existing inference task and artifact; never create or update a task."""
    if case["engineering_only"]:
        return {"scope": "ENGINEERING_NO_TASK_INFERENCE", "remote_mutations": 0}
    if case.get("task") == "tiny_shakespeare_char":
        return {"scope": "NATIVE_VALIDATION_NO_ADDITIONAL_INFERENCE", "remote_mutations": 0}
    if case.get("task") == "ruler_variable_tracking":
        from normal_training_tracking import service_scope

        return service_scope()
    from normal_training_plan import canonical
    from quality_inherit import leaf
    from seed202_pilot.cloud_services import _scalars

    root = Path(root)
    envelope = read(root / "task-clearml-readback.json")
    require(
        set(envelope) == {"task_id", "status", "artifact_sha256", "fresh_sdk_readback"}
        and envelope["status"] == "completed"
        and envelope["fresh_sdk_readback"] is True
        and type(envelope["task_id"]) is str
        and bool(envelope["task_id"]),
        "UNIFIED_INFERENCE_ENVELOPE",
    )
    report = read(root / "task-val.json")
    identity = read(root / "out/run_identity.json")
    services = leaf("services")

    def operation():
        from clearml import Task

        with services.service_env(values, "track"):
            observed = Task.get_task(task_id=envelope["task_id"])
            observed.reload()
            require(
                str(observed.get_status()).lower().rsplit(".", 1)[-1] == "completed",
                "UNIFIED_INFERENCE_COMPLETE",
            )
            binding = {
                "final_parameters_sha256": identity["final_parameters_sha256"],
                "config_sha256": hashlib.sha256(canonical(case["config"]).encode()).hexdigest(),
                "task": report["task"],
                "split": "val",
            }
            require(
                observed.get_parameters_as_dict()["seed202-binding"]
                == {key: str(value) for key, value in binding.items()},
                "UNIFIED_INFERENCE_BINDING",
            )
            metrics = observed.get_last_scalar_metrics()
            expected = _scalars(report)
            for name, value in expected.items():
                require(metrics["quality"][name]["last"] == value, "UNIFIED_INFERENCE_SCALARS")
            from clearml.backend_api.services import events as api_events

            response = (
                Task._get_default_session()
                .send(  # pylint: disable=protected-access
                    api_events.GetTaskEventsRequest(
                        task=envelope["task_id"],
                        order="asc",
                        scroll_id=None,
                        batch_size=10000,
                        event_type="training_stats_scalar",
                    )
                )
                .response_data
            )
            require(
                type(response) is dict
                and type(response.get("total")) is int
                and type(response.get("events")) is list
                and response["total"] == len(response["events"]) == len(expected),
                "UNIFIED_INFERENCE_UNSAMPLED_TOTAL",
            )
            actual = sorted(
                (row["metric"], row["variant"], row["iter"], row["value"])
                for row in response["events"]
            )
            require(
                actual == sorted(("quality", name, 0, value) for name, value in expected.items()),
                "UNIFIED_INFERENCE_UNSAMPLED_VALUES",
            )
            path = Path(
                observed.artifacts["task-val"].get_local_copy(
                    extract_archive=False, raise_on_error=True, force_download=True
                )
            )
            raw = path.read_bytes()
            require(raw == (root / "task-val.json").read_bytes(), "UNIFIED_INFERENCE_ARTIFACT")
            require(
                hashlib.sha256(raw).hexdigest() == envelope["artifact_sha256"],
                "UNIFIED_INFERENCE_STORED_RECEIPT",
            )
            observation = {
                **envelope,
                "metrics": expected,
                "binding": binding,
                "raw": response,
                "fresh_sdk_readback": True,
                "remote_mutations": 0,
            }
            from unified_normal_ordinary import inference_observation

            return inference_observation(root, case, observation)

    return services.safe("normal_host_inference_readback", operation)


def consume(package, root, case, values, data, *, snapshot, accept=False):
    """Join actual source/native/training/task/inference consumers with one recipe."""
    from normal_training_acceptance import training_readback, verify_sources
    from normal_training_native import verify_native
    from normal_training_plan import canonical
    from training_diagnostics import observe
    from unified_normal_host_view import lifecycle as assemble_lifecycle
    from unified_normal_host_view import reader_identity
    from unified_normal_publications import publication_readback

    source = verify_sources(package, root, case)
    native = verify_native(Path(root) / "out", case)
    training = observe(
        "normal_host_training_readback",
        lambda: training_readback(root, case, values),
        {"receipt": None, "raw": None, "fresh_sdk_readback": False},
    )
    task = task_quality(package, root, case, data)
    inference = observe(
        "normal_host_inference_readback", lambda: inference_readback(root, case, values)
    )
    publication = publication_readback(root, case, values, snapshot)
    result = {
        "schema": "unified-normal-host-observation-v1",
        "case": case,
        "scope": "ENGINEERING_CPU_SDK_FIXTURE_ONLY"
        if case["engineering_only"]
        else "TARGET_NORMAL_FULL_TASK_LIFECYCLE",
        "source": source,
        "native": native,
        "training": training,
        "task": task,
        "inference": inference,
        "publication": publication,
        "remote_mutations": 0,
        "next_launch_authorized": False,
    }
    if accept:
        require(native == read(Path(root) / "native-proof.json"), "NORMAL_HOST_NATIVE_DRIFT")
        # Fresh diagnostics are retained verbatim, never equated with an older SDK snapshot.
        lifecycle = assemble_lifecycle(case, source, native, training, publication["publications"])
        result.update(
            schema="unified-normal-whole-acceptance-v1",
            status="ACCEPTED_UNIFIED_NORMAL",
            lifecycle=lifecycle,
            host_reader=reader_identity(),
        )
        require(
            task.get("final_sha256", native["final_sha256"]) == native["final_sha256"],
            "UNIFIED_NORMAL_TASK_NATIVE_JOIN",
        )
        result["whole_lifecycle_sha256"] = hashlib.sha256(canonical(lifecycle).encode()).hexdigest()
    return result


def validate_arguments(args):
    """Public read-only routes reject incomplete source/data arguments before auth."""
    if args.command not in {"services", "accept"}:
        return
    for name in ("destination", "index", "output", "private", "service_snapshot"):
        require(getattr(args, name, None) is not None, "UNIFIED_HOST_REQUIRED_ARGUMENT")
    require(args.task_data is not None or args.mqar_data is not None, "UNIFIED_HOST_REQUIRED_DATA")
    require(
        all(
            not Path(args.output).resolve().is_relative_to(Path(p).resolve())
            for p in (args.destination, args.service_snapshot)
        ),
        "UNIFIED_HOST_IMMUTABLE_OUTPUT",
    )


def run_public(args, approval, provider, metadata, source):
    """The audited public HOST routes use the new saved NORMAL result layout."""
    import subprocess

    from seed202_pilot.fresh_mode import binding
    from seed202_pilot.provider import kernel_readback
    from seed202_pilot.storage import inventory

    validate_arguments(args)
    plan = approval["plan"]
    candidate = Path(plan["candidate_root"])
    bound = binding(candidate, plan["deployment"])
    require(bound == plan["source_authority"]["fresh"], "UNIFIED_HOST_SOURCE_AUTHORITY")
    index = read(args.index)
    before = kernel_readback(provider, metadata, source)
    require(
        before == index["provider_before"] and before["typed_status"] == "COMPLETE",
        "UNIFIED_HOST_PROVIDER_BEFORE",
    )
    require(inventory(args.destination) == index["inventory"], "UNIFIED_HOST_DOWNLOAD_INVENTORY")
    root = Path(args.destination) / ("normal-" + plan["nonce"])
    payload = candidate / "payload"
    script = Path(__file__).resolve()
    command = [
        shutil.which("uv"),
        "run",
        "--offline",
        "--no-project",
        "--no-sync",
        "--python",
        sys.executable,
        "python",
        "-B",
        str(script),
    ]
    fields = {
        "package": payload / "normal-package",
        "plan": payload / "unified/normal-plan.json",
        "recipe": payload / "configs" / (plan["case_id"] + ".yaml"),
        "data": args.task_data or args.mqar_data,
        "private": args.private,
        "result": root,
        "output": args.output,
        "case": plan["case_id"],
        "request": payload / "unified/request.json",
        "phase": args.command,
        "snapshot": args.service_snapshot,
        "nonce": plan["nonce"],
    }
    command += [token for name, value in fields.items() for token in ("--" + name, str(value))]
    allowed = set(
        (
            "PATH PATHEXT SYSTEMROOT WINDIR COMSPEC OS HOME USERPROFILE USERNAME USER "
            "LOGNAME APPDATA LOCALAPPDATA TMPDIR TMP TEMP UV_CACHE_DIR "
            "NUMBER_OF_PROCESSORS PROCESSOR_ARCHITECTURE"
        ).split()
    )
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    env.update(
        CUDA_VISIBLE_DEVICES="-1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHON_DOTENV_DISABLED="1",
        CLEARML_OFFLINE_MODE="0",
        HF_HUB_OFFLINE="0",
    )
    env["CLEARML_CONFIG_FILE"] = os.devnull
    cache = Path(os.environ["TMPDIR"]) / ("normal-host-caches-" + uuid.uuid4().hex)
    for key in (
        "HF_HOME",
        "CLEARML_CACHE_DIR",
        "TORCH_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
        "UV_CACHE_DIR",
    ):
        env[key] = str(cache / key.lower())
    env.pop("PYTHONPATH", None)
    child = subprocess.run(
        command, env=env, cwd=payload, capture_output=True, timeout=1800, check=False
    )
    require(child.returncode == 0, "UNIFIED_NORMAL_HOST_REJECTED")
    require(kernel_readback(provider, metadata, source) == before, "UNIFIED_HOST_PROVIDER_AFTER")
    require(inventory(args.destination) == index["inventory"], "UNIFIED_HOST_DOWNLOAD_AFTER")
    return read(args.output)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("package", "plan", "recipe", "data", "private", "result", "output", "snapshot"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--nonce")
    parser.add_argument("--phase", choices=("services", "accept"), required=True)
    args = parser.parse_args(argv)
    require(
        all(
            not args.output.resolve().is_relative_to(p.resolve())
            for p in (args.result.parent, args.snapshot, args.package)
        ),
        "UNIFIED_HOST_IMMUTABLE_OUTPUT",
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "UNIFIED_HOST_CPU_ENV",
    )
    with sources(args.package):
        from normal_training_plan import create_json, derive, digest
        from quality_inherit import leaf

        case = derive(args.package, args.plan, args.case, args.recipe, args.data)
        if not case["engineering_only"]:
            require(args.request is not None, "UNIFIED_HOST_DECLARED_REQUEST")
            request = read(args.request)
            expected = request["run"]
            require(
                case["id"] == expected["id"]
                and case["seed"] == expected["seed"]
                and case["family"] == expected["family"]
                and digest(args.plan) == expected["plan_sha256"],
                "UNIFIED_HOST_REQUEST_SCOPE",
            )
            domains = declared_data_domains(case, args.nonce)
            from normal_training_native import verify_native

            for index, domain in enumerate(domains):
                selected = {**case, "config": {**case["config"], "data": domain}}
                try:
                    verify_native(args.result / "out", selected)
                except ValueError as error:
                    if error.args != ("NORMAL_NATIVE_CONFIG",) or index == len(domains) - 1:
                        raise
                else:
                    case = selected
                    break
        values = leaf("services").credentials(args.private, rank=0)
        result = consume(
            args.package,
            args.result,
            case,
            values,
            args.data,
            snapshot=args.snapshot,
            accept=args.phase == "accept",
        )
        create_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
