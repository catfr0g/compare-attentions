"""Engineering-only public host chain with genuine qualified Sorting witnesses."""

import contextlib
import io
import json
import sys
import threading
import uuid
from pathlib import Path

import pytest
from test_seed202_mqar_builder import HISTORY, WORK

BASE = Path("E:/HermesArtifacts/compare-attention/sorting-mqar-seed202-pilot-v1")


def test_public_both_inputs_unknown_ack_get_only_with_genuine_original_error(monkeypatch):
    from seed202_mqar_package import json_bytes, stage_candidate

    root = BASE / (
        "mqar-classic-resume-repair-v1/continuation-development-03/"
        "engineering-shadow-" + uuid.uuid4().hex
    )
    stage_candidate(
        WORK / "parent",
        BASE
        / "remaining-three-after-tail-v1/qualified-launch-bridge-v1/development-07"
        / "sorting-mqar-seed202-pilot-v1.zip",
        HISTORY,
        root,
    )
    descriptor = json.loads((root / "candidate.json").read_bytes())
    descriptor.update(
        status="COMPLETE_FROZEN_CANDIDATE_REVIEW_REQUIRED",
        engineering_fixture="SYNTHETIC_SDK_ONLY_NOT_PRODUCTION_CANDIDATE",
    )
    (root / "candidate.json").write_bytes(json_bytes(descriptor))
    from seed202_mqar_binding_fixture import engineering_bind

    root = engineering_bind(root, BASE / ("mqar-engineering-bound-" + uuid.uuid4().hex[:16]))
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "seed202_pilot"
        or name.startswith("seed202_pilot.")
        or name == "compare_attention"
        or name.startswith("compare_attention.")
    }
    for name in previous_modules:
        del sys.modules[name]
    paths = [
        str(root / "payload/source/src"),
        str(root / "payload/source/tests/seed202_pilot"),
        str(root / "payload/source/tests/qualified_launch"),
    ]
    sys.path[:0] = paths
    try:
        from fresh_sdk_fixture import FreshTransport
        from seed202_pilot.mqar_host_resume import original
        from seed202_qualified_http_fixture import HttpFixture
        from test_fresh_public_chain_integration06 import _approve, _prepare, _probe, _review

        transport = FreshTransport(root, monkeypatch)
        server_call = transport.call

        def all_files_server(service, method, request, response_type):
            if method == "CreateDataset" and len(request.files) > 1:
                import types

                ref = request.owner_slug + "/" + request.slug
                received = dict(transport.uploads[row.token] for row in request.files)
                assert len(received) == len(request.files) == 5
                transport.add_dataset(ref, received, 2000 + len(transport.datasets))
                transport.mutations.append(("CreateDataset", ref))
                if transport.lost_ack == transport.current_action:
                    raise TimeoutError("SYNTHETIC_LOST_ACK")
                return types.SimpleNamespace(error=None, ref=ref)
            return server_call(service, method, request, response_type)

        monkeypatch.setattr(transport, "call", all_files_server)
        original_command = transport.command
        plan_path = transport.root / "FIXTURE-ONLY-plan.json"
        review = transport.root / "FIXTURE-ONLY-independent-decision.json"

        def quiet_command(command, case_id, *extra, journal=None):
            extra = list(extra)
            if command != "make-plan":
                if "--plan" not in extra:
                    extra += ["--plan", plan_path]
                if "--review" not in extra:
                    extra += ["--review", review]
            with contextlib.redirect_stdout(io.StringIO()):
                return original_command(command, case_id, *extra, journal=journal)

        monkeypatch.setattr(transport, "command", quiet_command)
        metadata, source, worker = original(root)
        transport.add_dataset(
            metadata["dataset_sources"][0].rsplit("/", 1)[0],
            {"seed202-pilot.zip.bin": worker},
            13454634,
        )
        transport.add_kernel(metadata, source, 145778849, "ERROR")
        transport.enable_mutations()
        wire = HttpFixture(transport, monkeypatch)
        # The production reader owns two SDK clients. The legacy in-memory HTTP
        # server has one pending-request slot, so serialize only this network seam.
        wire_lock = threading.RLock()

        def serialized_call(*args):
            with wire_lock:
                return wire.call(*args)

        monkeypatch.setattr(transport, "call", serialized_call)
        from seed202_pilot import provider as provider_module

        create_dataset = provider_module.create_private_dataset
        mutation_errors = []

        def traced_create(*args, **kwargs):
            import traceback

            try:
                return create_dataset(*args, **kwargs)
            except Exception:
                mutation_errors.append(traceback.format_exc())
                raise

        monkeypatch.setattr(provider_module, "create_private_dataset", traced_create)
        case = "mqar-k64-classic-seed202"
        bridge = json.loads((root / "payload/qualified-launch/bridge.json").read_bytes())
        delta = bridge["cases"]["sorting-deltanet-seed202"]["spec"]["output"]
        assert (
            transport.command(
                "make-plan",
                case,
                "--acceptance",
                delta,
                "--output",
                plan_path,
                journal=transport.root / "journals",
            )
            == 0
        ), transport.last_trace
        plan = json.loads(plan_path.read_bytes())
        assert plan["source_authority"]["continuation"]["original_error"]["numeric_id"] == 145778849
        review.write_bytes(json_bytes(_review(plan)))
        from seed202_pilot import cli

        original_provider = cli._provider

        def leased_provider(args):
            before = review.read_bytes()
            with pytest.raises(PermissionError):
                with review.open("r+b"):
                    pass
            assert review.read_bytes() == before
            if args.approval:
                with pytest.raises(PermissionError):
                    with args.approval.open("r+b"):
                        pass
            return original_provider(args)

        monkeypatch.setattr(cli, "_provider", leased_provider)
        for action in ("publish", "publish-resume", "submit"):
            preparation = _prepare(transport, case, plan, plan_path, review, action)
            proof = _probe(transport, case, plan, plan_path, review, action, preparation)
            observed = json.loads(proof.read_bytes())
            assert [r["replay_scope"] for r in observed["predecessor_evidence"]] == [
                "QUALIFIED_SORTING_CLASSIC_TAIL_V1",
                "QUALIFIED_SORTING_DELTA_TAIL_V1",
            ]
            approval = _approve(transport, case, plan, plan_path, review, proof, action)
            journal = Path(plan["journals"][action])
            transport.lost_ack = action
            assert (
                transport.command(
                    action,
                    case,
                    "--approval",
                    approval,
                    "--review",
                    review,
                    "--preparation",
                    preparation,
                    journal=journal,
                )
                == 1
            )
            count = len(transport.mutations)
            assert any(
                method == ("SaveKernel" if action == "submit" else "CreateDataset")
                for method, _ref in transport.mutations
            ), mutation_errors
            from seed202_pilot import cli

            provider = cli._provider
            authenticated = []

            def counted_provider(args, observed=authenticated, factory=provider):
                observed.append(True)
                return factory(args)

            with monkeypatch.context() as boundary:
                boundary.setattr(cli, "_provider", counted_provider)
                assert (
                    transport.command("status", case, "--approval", approval, journal=journal) == 1
                )
                assert not authenticated
            assert (
                transport.command(
                    action,
                    case,
                    "--approval",
                    approval,
                    "--review",
                    review,
                    "--preparation",
                    preparation,
                    journal=journal,
                )
                == 1
            )
            assert len(transport.mutations) == count
            assert (
                transport.command(
                    "reconcile", case, "--approval", approval, "--action", action, journal=journal
                )
                == 0
            ), transport.last_trace
            assert len(transport.mutations) == count
        assert sum(method == "CreateDataset" for method, _fields in wire.serialized) == 2
        assert sum(method == "SaveKernel" for method, _fields in wire.serialized) == 1
        resume_target = plan["deployment"]["resume_input_target"].rsplit("/", 1)[0]
        from seed202_mqar_authority import pin
        from seed202_pilot.mqar_publisher import local_bundle

        published = transport.datasets[resume_target]
        assert {name: pin(raw) for name, raw in published["files"].items()} == local_bundle(
            root / "payload/deploy" / case, plan["deployment"]
        )
        target = plan["deployment"]["kernel_target"]
        transport.kernels[target].status = "COMPLETE"
        wire.outputs = {
            "ENGINEERING_ONLY_NO_OPTIMIZER.txt": b"Synthetic SDK collection, no scientific result\n"
        }
        assert transport.command("status", case, "--approval", approval, journal=journal) == 0, (
            transport.last_trace
        )
        download = transport.root / "FIXTURE-ONLY-download"
        index = transport.root / "FIXTURE-ONLY-download-index.json"
        assert (
            transport.command(
                "collect",
                case,
                "--approval",
                approval,
                "--destination",
                download,
                "--index",
                index,
                journal=Path(plan["journals"]["collect"]),
            )
            == 0
        ), transport.last_trace
        collected = json.loads(index.read_bytes())
        assert collected["file_count"] == 1
        assert collected["provider_before"]["typed_status"] == "COMPLETE"
        assert (
            transport.command(
                "next",
                case,
                "--qualified-predecessor",
                root / "payload/predecessor/qualified-acceptance.json",
                "--acceptance",
                delta,
                "--acceptance",
                transport.root / "ABSENT_REAL_MQAR_ACCEPTANCE.json",
            )
            == 1
        )
    finally:
        for path in paths:
            sys.path.remove(path)
        for name in list(sys.modules):
            if (
                name == "seed202_pilot"
                or name.startswith("seed202_pilot.")
                or name == "compare_attention"
                or name.startswith("compare_attention.")
            ):
                del sys.modules[name]
        sys.modules.update(previous_modules)
