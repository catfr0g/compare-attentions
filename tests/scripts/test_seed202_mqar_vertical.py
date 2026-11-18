"""ENGINEERING ONLY full native/periodic/service/composite path without training."""

import importlib
import json
import os
import sys
import uuid
from pathlib import Path

import pytest
from test_seed202_mqar_builder import HISTORY, WORK


def test_full_ordinary_native_eleven_publications_and_required_continuation(tmp_path, monkeypatch):
    from build_seed202_mqar_continuation import stage_science_helpers, stage_source

    source = stage_source(WORK / "parent", HISTORY, tmp_path / "source")
    helpers = stage_science_helpers(WORK / "parent", tmp_path / "science")
    old = {key: value for key, value in sys.modules.items() if key.startswith("seed202_pilot")}
    for name in old:
        del sys.modules[name]
    paths = [str(source), str(helpers / "operator"), str(helpers / "operator/inherited")]
    sys.path[:0] = paths
    previous_join = sys.modules.get("quality_runtime_join")
    try:
        import torch
        from seed202_mqar_binding_fixture import engineering_candidate
        from seed202_mqar_scientific_fixture import produce
        from seed202_pilot.inherit import read_parent_member
        from seed202_pilot.science import _JOIN, _NATIVE, _module
        from seed202_pilot.storage import create_json

        def forbidden(*_args, **_kwargs):
            raise AssertionError("HOST_MODEL_OPTIMIZER_OR_RNG_RESTORE_FORBIDDEN")

        monkeypatch.setattr(torch.optim, "AdamW", forbidden)
        monkeypatch.setattr(torch.random, "fork_rng", forbidden)
        native = importlib.import_module("seed202_pilot.mqar_host_native")
        staged = Path(os.environ["MQAR_TEST_CANDIDATE"])
        work = staged.parent / ("ENGINEERING_ONLY_vertical_" + uuid.uuid4().hex)
        candidate = engineering_candidate(staged, work / "candidate")
        deployment = next(
            row
            for row in json.loads((candidate / "PLAN-UNAPPROVED.json").read_bytes())["deployments"]
            if row["case_id"] == "mqar-k64-classic-seed202"
        )
        nonce = deployment["nonce"]
        downloaded = work / "download"
        run = downloaded / ("seed202-pilot-" + nonce)
        root = run / deployment["case_id"]
        case, identity, quality, snapshot, rows, config, fixture = produce(root, monkeypatch)
        from seed202_mqar_worker_fixture import worker_return

        worker_args = worker_return(root, case, identity, quality, config, candidate, monkeypatch)
        assert worker_args.wheel_destination.is_dir()
        parent = WORK / "parent/inherited/parent-candidate03.zip.bin"
        sys.modules["quality_runtime_join"] = _module(
            "quality_runtime_join", read_parent_member(parent, _JOIN)
        )
        original_native = _module("quality_native_v2", read_parent_member(parent, _NATIVE))
        proof = native.bind_native(vars(original_native)["verify_native"])(root / "out", case)
        assert proof["updates"] == 10780 and proof["shared_fields_verified"] == 9
        create_json(root / "native-proof.json", proof)
        from seed202_pilot.periodic import replay_periodic

        periodic = replay_periodic(root, case, rows, snapshot, tmp_path / "periodic-replay")
        assert periodic["revision_count"] == 11 and periodic["full_count"] == 21
        from seed202_mqar_acceptance_fixture import close

        data, services, labels, _clearml_fixture = close(
            root,
            case,
            identity,
            quality,
            snapshot,
            rows,
            config,
            fixture,
            candidate,
            worker_args,
            monkeypatch,
        )
        from seed202_pilot.contract import require_authorized_case
        from seed202_pilot.mqar_composite import consume

        spec = require_authorized_case(case["id"])
        config_path = candidate / "payload/configs" / (case["id"] + ".yaml")
        assert consume(root, case, config_path, spec)["continuation"]["remaining_updates"] == 9800
        assert (
            json.loads((root / "resume-provenance.json").read_bytes())["original_status"] == "ERROR"
        )
        for name in ("resume-provenance.json", "restore-equality-rank-1.json"):
            raw = (root / name).read_bytes()
            (root / name).unlink()
            with pytest.raises(FileNotFoundError):
                consume(root, case, config_path, spec)
            (root / name).write_bytes(raw)
        create_json(
            run / "intent.json",
            {
                "schema_version": 1,
                "case_id": case["id"],
                "nonce": nonce,
                "case_deadline_seconds": spec.case_deadline_seconds,
                "policy": "STOPALL_ANYFAIL",
            },
        )
        create_json(
            run / "finish-returned.json",
            {
                "case_id": case["id"],
                "status": "AWAITING_PROVIDER_DOWNLOAD_AND_HOST_WHOLE_ACCEPTANCE",
            },
        )
        from seed202_pilot.storage import inventory

        index = work / "ENGINEERING_ONLY_index.json"
        create_json(index, {"inventory": inventory(downloaded)})
        provider = {
            "typed_status": "COMPLETE",
            "head": 1,
            "version": 1,
            "private": True,
            "numeric_id": 123,
            "target": deployment["kernel_target"] + "/1",
            "source_sha256": deployment["kernel_source_sha256"],
        }
        bridge = json.loads((candidate / "payload/qualified-launch/bridge.json").read_bytes())
        # Whole host replay must import a fresh authenticated wheel, as in production.
        for name in list(sys.modules):
            if name == "compare_attention" or name.startswith("compare_attention."):
                del sys.modules[name]
        from seed202_pilot.acceptance import complete

        output = work / "ENGINEERING_ONLY_acceptance.json"
        accepted = complete(
            spec,
            nonce,
            downloaded,
            index,
            provider,
            provider,
            services,
            task_labels=labels,
            task_data=data,
            wheel=candidate / "payload/inherited/training-wheel.whl",
            parent_archive=parent,
            science_dir=Path(bridge["science"]),
            config_path=config_path,
            periodic_destination=work / "acceptance-periodic",
            output=output,
        )
        assert accepted["scientific"]["continuation"]["original_status_preserved"] == "ERROR"
        from seed202_pilot.qualified_bridge import next_qualified_case

        following = next_qualified_case(
            candidate,
            candidate / "payload/predecessor/qualified-acceptance.json",
            [Path(bridge["cases"]["sorting-deltanet-seed202"]["spec"]["output"]), output],
        )
        assert following.case_id == "mqar-k64-deltanet-seed202"
    finally:
        if previous_join is None:
            sys.modules.pop("quality_runtime_join", None)
        else:
            sys.modules["quality_runtime_join"] = previous_join
        for path in paths:
            sys.path.remove(path)
        for name in list(sys.modules):
            if name.startswith("seed202_pilot"):
                del sys.modules[name]
        sys.modules.update(old)
