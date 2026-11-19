"""Published mount -> saved main -> selected UV metadata worker -> real SDK destinations."""

import builtins
import contextlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from types import FunctionType
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_host_saved_main_collector_fresh_sdk_observer_whole_composite(tmp_path, monkeypatch):
    import uuid

    from mqar_final_tail_chain_child import metadata
    from mqar_final_tail_test_layout import ART, DATA, INPUT, SCIENCE
    from mqar_final_tail_test_support import World

    runtime = ART / "ENGINEERING-ONLY-runtime" / ("r-" + uuid.uuid4().hex[:12])
    runtime.mkdir(parents=True, exist_ok=False)
    tmp_path = runtime  # Heavy fixture outputs stay on E; source/tests remain on C.
    world = World(tmp_path, monkeypatch)
    for action in ("publish", "submit"):
        receipt, approval = world.prepare(action)
        assert world.command(action, action, receipt=receipt, approval=approval) == 0, world.errors
    raw = runpy.run_path(str(world.source / "kernel/kernel.py"))["main"].__globals__
    mount = tmp_path / "mount"
    published = world.datasets[world.plan["dataset_target"][:-2]]
    slug = raw["BINDING"]["slug"]
    mounted = mount / "datasets/zimike" / slug
    mounted.mkdir(parents=True)
    for name, value in published.items():
        (mounted / name).write_bytes(value)
    science = mount / "sorting-quality-input-fcff1b5daf574b70"
    science.mkdir()
    shutil.copyfile(world.science, science / "scientific.zip.bin")
    private = mount / "k64-private-20261109a1b2"
    private.mkdir()
    shutil.copyfile(world.private, private / "private.env")
    events = []
    original_import, real_run = builtins.__import__, subprocess.run
    scripts = Path(__file__).resolve().parents[2] / "scripts"

    def setup(_science, work, _deadline):
        events.append("SETUP_BEFORE_SELECTED_UV")
        work.mkdir(parents=True)
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            in {
                "SYSTEMROOT",
                "PATH",
                "HOME",
                "USERPROFILE",
                "USERNAME",
                "USER",
                "LOGNAME",
                "TMPDIR",
                "TMP",
                "TEMP",
                "UV_CACHE_DIR",
            }
        }
        env.update(
            CUDA_VISIBLE_DEVICES="-1",
            PYTHONDONTWRITEBYTECODE="1",
            CLEARML_CONFIG_FILE=str(tmp_path / "NO_REAL_CONFIG"),
        )
        return shutil.which("uv"), sys.executable, env, {}

    def selected(command, env, cwd, timeout):
        events.append("ACTUAL_SELECTED_UV_CHILD")
        at = command.index("-B") + 1
        assert Path(command[at]).name == "mqar_final_tail_cli.py"
        command = command[:at] + [str(scripts / "mqar_final_tail_chain_child.py")] + command[at:]
        env.update(PYTHONPATH=str(scripts), FIXTURE_INPUT=str(INPUT), CUDA_VISIBLE_DEVICES="-1")
        result = real_run(command, env=env, cwd=cwd, capture_output=True, text=True, timeout=180)
        (world.root / "worker.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ENGINEERING_METADATA_REAL_SDK_HTTP_NOT_SCIENTIFIC_INFERENCE" in result.stdout

    def imported(name, *args, **kwargs):
        if name == "session":
            return NS(setup=setup)
        if name == "process_control":
            return NS(bounded_run=selected)
        return original_import(name, *args, **kwargs)

    def resolve(root, owner, slug_, name):
        if slug_ == "ca-kv-1c660aa5f54d88ea9e0c1ce4":
            return DATA / name  # Declared fixture path relocation; no array copying.
        return raw["resolve_mount"](root, owner, slug_, name)

    environment = NS(environ={}, pathsep=os.pathsep)
    platform = NS(system=lambda: "Linux", python_version=lambda: "3.13.16")
    globs = {
        **raw,
        "os": environment,
        "platform": platform,
        "resolve_mount": resolve,
        "__builtins__": {**vars(builtins), "__import__": imported},
        "driver_directory": lambda: "/usr/local/nvidia/lib64",
    }
    globs["inspect_hardware"] = metadata(
        raw["inspect_hardware"], os=environment, platform=platform, bad=True
    )
    main = FunctionType(raw["main"].__code__, globs, "main", raw["main"].__defaults__)
    main(mount, tmp_path / "working", tmp_path / "package")
    assert events == ["SETUP_BEFORE_SELECTED_UV", "ACTUAL_SELECTED_UV_CHILD"]
    work = tmp_path / "working" / ("mqar202-final10780-tail-" + world.plan["nonce"])
    world.kernels[world.plan["kernel_target"]].status = "COMPLETE"
    world.outputs = {
        p.relative_to(work).as_posix(): p.read_bytes() for p in work.rglob("*") if p.is_file()
    }
    import requests

    def get(_session, url, **_kwargs):
        value = world.outputs[url.removeprefix("https://host-fixture.invalid/")]
        return contextlib.nullcontext(
            NS(raise_for_status=lambda: None, iter_content=lambda _: iter([value]))
        )

    monkeypatch.setattr(requests.Session, "get", get)
    destination, index = tmp_path / "download", tmp_path / "download-index.json"
    assert world.command("collect", destination=destination, index=index) == 0, world.errors

    def observer(command, **kwargs):
        at = command.index("-B") + 1
        assert Path(command[at]).name == "mqar_final_tail_service_cli.py"
        command[at] = str(scripts / "mqar_final_tail_observer_child.py")
        kwargs["env"].update(
            FIXTURE_SERVICE_STATE=str(work / "fixture-service-state.json"),
            PYTHONPATH=str(scripts),
            CUDA_VISIBLE_DEVICES="-1",
        )
        kwargs.pop("stdout", None)
        kwargs.pop("stderr", None)
        result = real_run(command, capture_output=True, text=True, **kwargs)
        (world.root / "observer.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    monkeypatch.setattr(subprocess, "run", observer)
    services = tmp_path / "fresh-services.json"
    assert (
        world.command(
            "services",
            input=INPUT,
            work=destination,
            science_dir=SCIENCE,
            data=DATA,
            output=services,
        )
        == 0
    ), world.errors
    acceptance = tmp_path / "ENGINEERING_ONLY_NOT_GENUINE_ACCEPTANCE.json"
    assert (
        world.command(
            "accept",
            destination=destination,
            index=index,
            result=destination / "result",
            native=destination / "native-audit.json",
            service_proof=services,
            policy_review=world.plan["review_path"],
            data=DATA,
            output=acceptance,
        )
        == 0
    ), world.errors
    value = json.loads(acceptance.read_bytes())
    assert value["later_pilots_authorized"] is False
    assert [r["numeric_id"] for r in value["original_provider_errors"]] == [145778849, 145850896]
    assert all(r["typed_status"] == "ERROR" for r in value["original_provider_errors"])
    assert value["services"]["clearml"]["unsampled"]["total"] == 5
    from mqar_final_tail_native import digest
    from mqar_final_tail_next import SORTING
    from mqar_final_tail_test_layout import BASE

    status = tmp_path / "status.json"
    assert world.command("status", output=status) == 0
    spec = {
        "release": str(world.source / "payload"),
        "result": str(destination / "result"),
        "native": str(destination / "native-audit.json"),
        "services": str(services),
        "provider": str(status),
        "review": world.plan["review_path"],
        "plan": str(world.source / "plan.json"),
        "data": str(DATA),
        "input": str(INPUT),
        "output": str(acceptance),
    }
    spec_path = tmp_path / "FIXTURE-replay.json"
    spec_path.write_text(json.dumps(spec))
    transition = tmp_path / "FIXTURE-transition-review.json"
    transition.write_text(
        json.dumps(
            {
                "status": "PASS",
                "independent": True,
                "scope": "MQAR202_QUALIFIED_CLASSIC_TO_DELTA_WITNESS",
                "composite_sha256": digest(acceptance),
                "release_sha256": digest(world.source / "payload/manifest.json"),
                "permission": "STANDING_CONDITIONAL_MQAR_DELTA_ONLY_NO_SUBMISSION",
                "sorting_predecessors": SORTING,
            }
        )
    )
    next_output = tmp_path / "FIXTURE-not-launch-authority.json"
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
        str(scripts / "mqar_final_tail_next.py"),
    ]
    for key, arg in {
        "spec": spec_path,
        "review": transition,
        "science": SCIENCE,
        "work": tmp_path / "next-work",
        "output": next_output,
        "sorting-classic": BASE
        / "full-budget-tail-recovery-v1/execution-final02-20261117/qualified-acceptance.json",
        "sorting-delta": BASE
        / "delta-inference-tail-recovery-v1/execution-final01/qualified-acceptance.json",
    }.items():
        command += ["--" + key, str(arg)]
    result = real_run(command, env=dict(os.environ), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    tagged = json.loads(next_output.read_bytes())
    assert tagged["next_case"] == "mqar-k64-deltanet-seed202" and tagged["submitted"] is False
    for mutate in (
        lambda r: r.update(loss=r["loss"] + 1),
        lambda r: r.update(correct=r["correct"] - 1),
        lambda r: r["model_config"].update(n_layer=8),
        lambda r: r["data"]["config"].update(num_kv_pairs=32),
        lambda r: r["checkpoint"].update(provenance={"status": "completed"}),
    ):
        import copy

        import pytest
        from mqar_final_tail_report import verify_report

        report = json.loads((destination / "result/mqar-val.json").read_bytes())
        records = json.loads((destination / "result/mqar-val-records.json").read_bytes())
        broken = copy.deepcopy(report)
        mutate(broken)
        with pytest.raises(ValueError):
            verify_report(broken, DATA, records)
    before = acceptance.read_bytes()
    bad = {**value, "extra_acceptance_key": True}
    acceptance.write_text(json.dumps(bad))
    assert (
        world.command(
            "accept",
            destination=destination,
            index=index,
            result=destination / "result",
            native=destination / "native-audit.json",
            service_proof=services,
            policy_review=world.plan["review_path"],
            data=DATA,
            output=acceptance,
        )
        == 1
    )
    acceptance.write_bytes(before)
    assert tagged["ordinary_training_acceptance_fabricated"] is False
    hf_only = tmp_path / "FIXTURE-hf-only.json"
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
        str(scripts / "mqar_final_tail_observer_child.py"),
    ]
    for key, arg in {
        "release": world.source / "payload",
        "science": SCIENCE,
        "input": INPUT,
        "work": destination,
        "private": world.private,
        "nonce": world.plan["nonce"],
        "data": DATA,
        "output": hf_only,
        "hf-revision": value["services"]["hf"]["revision"],
    }.items():
        command += ["--" + key, str(arg)]
    command += ["--hf-reconcile-only"]
    env = dict(os.environ)
    env.update(
        HF_HUB_OFFLINE="0",
        CLEARML_OFFLINE_MODE="0",
        CUDA_VISIBLE_DEVICES="-1",
        FIXTURE_SERVICE_STATE=str(work / "fixture-service-state.json"),
        PYTHONPATH=str(scripts),
    )
    original_journal = (destination / "result-services/hf.jsonl").read_bytes()
    readback = real_run(command, env=env, capture_output=True, text=True, timeout=120)
    assert readback.returncode == 0, readback.stdout + readback.stderr
    hf_value = json.loads(hf_only.read_bytes())
    assert hf_value["remote_mutations"] == 0 and hf_value["scientific_acceptance"] is False
    assert hf_value["hf"] == value["services"]["hf"]
    assert (destination / "result-services/hf.jsonl").read_bytes() == original_journal
    command[command.index("--hf-revision") + 1] = "f" * 40
    command[command.index("--output") + 1] = str(tmp_path / "FIXTURE-hf-wrong-revision.json")
    rejected = real_run(command, env=env, capture_output=True, text=True, timeout=120)
    assert rejected.returncode == 1 and "FINAL_VERIFIED_REVISION_DRIFT" in rejected.stderr
    assert (destination / "result-services/hf.jsonl").read_bytes() == original_journal
