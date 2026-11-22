"""CPU contracts for the isolated LR matrix and fail-closed rollout."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "local/experiments/fp16-lr-pilots"
FAMILIES = ("classic", "deltanet")


@pytest.fixture(name="comparison")
# Cohesive on-disk provenance fixture; one setup shared by adversarial tests.
# pylint: disable-next=too-many-locals
def fixture_comparison(tmp_path, monkeypatch):
    return load_test_support().fixture_comparison(tmp_path, monkeypatch)


@pytest.mark.parametrize("index", range(6))
def test_lr_comparison_accepts_only_intended_differences(comparison, index):
    operator, manifest = comparison
    assert operator.verify_lr_comparison(manifest, names("1e-3")[index])["matched"] is True


@pytest.mark.parametrize(
    "mutation",
    ["runtime", "initial", "model", "cadence", "optim", "data", "frozen", "baseline", "scheduler"],
)
def test_lr_comparison_rejects_mutations(comparison, mutation):
    operator, manifest = comparison
    name = names("1e-3")[0]
    state = json.loads(manifest.read_text())
    path = operator.result_dir(manifest, name) / "run_identity.json"
    identity = json.loads(path.read_text())
    if mutation == "runtime":
        identity["runtime"]["torch"] = "different"
    elif mutation == "initial":
        identity["initial_parameters_sha256"] = "c" * 64
    elif mutation == "model":
        identity["config"]["model"]["dropout"] = 0.1
    elif mutation == "cadence":
        identity["config"]["schedule"]["val_every_steps"] = 1
    elif mutation == "optim":
        identity["config"]["optim"]["weight_decay"] = 0.2
    elif mutation == "scheduler":
        state["identity"]["configs"][name]["optim"]["lr_schedule"] = "warmup_cosine"
    elif mutation == "data":
        state["identity"]["data"]["spec"]["seed"] = 5
    elif mutation == "frozen":
        state["identity"]["configs"][name]["kv"]["dataset_seed"] = 4
    else:
        old = json.loads(operator.BASELINE.read_text())
        slug = next(iter(old["runs"].values()))["slug"]
        (
            operator.BASELINE.parent / "runs" / slug / "output" / slug / "run_identity.json"
        ).write_text("{}")
    path.write_text(json.dumps(identity))
    manifest.write_text(json.dumps(state))
    with pytest.raises(
        (ValueError, RuntimeError), match="(?i)(runtime|initial|config|data|checksum|baseline)"
    ):
        operator.verify_lr_comparison(manifest, name)


@pytest.mark.parametrize(
    "failure", [None, "scientific", "comparison", "first_runtime", "scheduled_history"]
)
# Explicit ordering journal oracle retains distinct submit, LR, science and comparison events.
# pylint: disable-next=too-many-locals
def test_singletons_full_gate_order_failfast_and_partial_reports(comparison, monkeypatch, failure):
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel

    operator, manifest = comparison
    events = []

    def submit(_manifest, selected, **kwargs):
        assert len(selected) == 1 and kwargs["accelerator"] == "NvidiaTeslaT4"
        events.append(("submit", selected[0]))

    def scientific(_manifest, name, *steps):
        events.append(("scientific", name))
        if failure == "scientific" and name == names("1e-3")[1]:
            raise ValueError("scientific rejection")
        assert steps == (() if "/full/" in name else (1 if "/exact1/" in name else 32,))
        return {"verified": True}

    real_compare = operator.verify_lr_comparison

    def compare(_manifest, name):
        events.append(("comparison", name))
        if (failure == "comparison" and name == names("1e-3")[1]) or (
            failure == "first_runtime" and name == names("1e-3")[0]
        ):
            path = operator.result_dir(manifest, name) / "run_identity.json"
            identity = json.loads(path.read_text())
            identity["runtime"]["torch"] = "changed"
            path.write_text(json.dumps(identity))
        return real_compare(_manifest, name)

    def telemetry(_api, _manifest, name):
        events.append(("telemetry", name))
        if failure == "scheduled_history" and name == scheduled_names()[0]:
            operator.protocol.verify_history(
                [], {"step": 1, "flops": 10}, {"lr_schedule": "warmup_cosine"}, 10, 10
            )
        return {"verified": True}

    monkeypatch.setattr(operator.protocol, "verify_artifact", telemetry)
    monkeypatch.setattr(operator, "run_selected", submit)
    monkeypatch.setattr(
        operator,
        "load_helper",
        lambda: SimpleNamespace(verify_stage=scientific, verify_pilot=scientific),
    )
    monkeypatch.setattr(operator, "verify_lr_comparison", compare)
    prior = {"accepted": [], "status": "rejected", "error": "prior rejection preserved"}
    operator.write_json(manifest.parent / "lr-results.json", prior)
    prior_sha = operator.checksum(manifest.parent / "lr-results.json")
    if failure:
        with pytest.raises(ValueError):
            operator.run_plan(manifest, "owner/credentials")
    else:
        operator.run_plan(manifest, "owner/credentials")
    submitted = [name for kind, name in events if kind == "submit"]
    stop = {None: 12, "scientific": 2, "comparison": 2, "first_runtime": 1, "scheduled_history": 7}[
        failure
    ]
    assert submitted == (names("1e-3") + scheduled_names())[:stop]
    for index, name in enumerate(submitted):
        if not (failure == "scheduled_history" and name == submitted[-1]):
            assert events.index(("submit", name)) < events.index(("scientific", name))
        if index:
            assert events.index(("comparison", submitted[index - 1])) < events.index(
                ("submit", name)
            )
    assert operator.read_json(manifest.parent / f"lr-results.previous-{prior_sha}.json") == prior
    report = json.loads((manifest.parent / "lr-results.json").read_text())
    assert len(report["accepted"]) == (stop - 1 if failure else 12)
    assert report["status"] == ("rejected" if failure else "complete_stop_lower_lr_not_submitted")
    assert all(
        entry["state"] == "staged" for entry in json.loads(manifest.read_text())["runs"].values()
    )


def test_real_reviewed_helper_globals_unchanged(comparison):
    operator, manifest = comparison
    helper = operator.load_helper()
    before = {
        key: deepcopy(getattr(helper, key))
        for key in ("ROOT", "CACHE", "PREVIOUS", "SMOKES", "BURNINS", "PILOTS", "REMAINING")
    }
    assert helper.verify_stage.__globals__ is helper.__dict__
    globals_before = dict(helper.__dict__)
    name = names("1e-3")[0]
    assert helper.expected_config(manifest, name) == operator.expected_config(
        operator.read_json(manifest), name
    )
    for verifier, args in (
        (helper.verify_stage, (manifest, name, 1)),
        (helper.verify_pilot, (manifest, name)),
    ):
        with pytest.raises(FileNotFoundError):
            verifier(*args)
    assert globals_before == helper.__dict__
    assert before == {key: getattr(helper, key) for key in before}
    assert helper.ROOT == ROOT
    assert helper.PILOTS[0].startswith("experiments/fp16_long_controls/")


def test_cli_check_and_launch_approval_boundary(tmp_path, monkeypatch):
    operator = load_operator()
    monkeypatch.setattr(operator, "HERE", tmp_path)
    calls = []
    monkeypatch.setattr(operator, "authorize", lambda: calls.append("credentials"))
    monkeypatch.setattr(operator, "publish_series", lambda _path: calls.append("publish"))
    monkeypatch.setattr(operator, "reuse_private_data", lambda *_args: calls.append("reuse"))
    monkeypatch.setattr(operator, "run_plan", lambda *_args: calls.append("run"))
    monkeypatch.setattr(
        operator,
        "bindings",
        lambda _manifest: {"source_sha256": "current", "operator_sha256": "operator"},
    )
    manifest = tmp_path / "series.json"
    manifest.write_text("{}")
    monkeypatch.setattr(operator, "MANIFEST", manifest)
    (tmp_path / "candidate.json").write_text(json.dumps({"manifest": str(manifest)}))
    for argv in ([], ["--check"], ["--launch"]):
        with pytest.raises((RuntimeError, FileNotFoundError)):
            operator.main(argv)
        assert not calls
    bindings = operator.bindings(manifest)
    review = {"approved": True, "bindings": bindings}
    (tmp_path / "lr-review.json").write_text(json.dumps(review))
    from compare_attention.kaggle_artifacts import checksum  # pylint: disable=import-outside-toplevel

    approval = {
        "approved": True,
        "manifest": str(manifest.resolve()),
        **bindings,
        "lr_review_sha256": checksum(tmp_path / "lr-review.json"),
    }
    (tmp_path / "lr-cloud-approval.json").write_text(json.dumps(approval))
    operator.main(["--check"])
    assert not calls
    approval["operator_sha256"] = "changed"
    (tmp_path / "lr-cloud-approval.json").write_text(json.dumps(approval))
    with pytest.raises(RuntimeError, match="operator_sha256"):
        operator.main(["--launch"])
    assert not calls


def test_source_pin_fails_before_any_credentials(monkeypatch):
    operator = load_operator()
    reference = operator.load_reference()
    monkeypatch.setattr(reference, "SOURCE_SHA256", "wrong-pin")
    with pytest.raises(RuntimeError, match="source"):
        reference.evidence_bindings()


@pytest.fixture(name="frozen_series")
# Complete release fixture exercises real archive/file verifiers.
# pylint: disable-next=too-many-locals
def fixture_frozen_series(comparison, monkeypatch):
    return load_test_support().fixture_frozen_series(comparison, monkeypatch)


@pytest.mark.parametrize(
    "mutation", [None, "source", "config", "data", "archive", "wheel", "lower_lr"]
)
def test_full_frozen_preflight_byte_and_identity_mutations(frozen_series, mutation):
    operator, manifest = frozen_series
    if mutation in {"source", "data", "archive", "wheel"}:
        relative = {
            "source": "source/source.py",
            "data": "data/meta.json",
            "archive": "release/release.bundle",
            "wheel": "bundle/fake.whl",
        }[mutation]
        (manifest.parent / relative).write_bytes(b"mutated")
    if mutation in {"config", "lower_lr"}:
        state = operator.read_json(manifest)
        if mutation == "config":
            state["identity"]["configs"][names("1e-3")[0]]["optim"]["learning_rate"] = 0.03
        else:
            state["runs"][names("1e-4")[0]]["state"] = "submitted"
        manifest.write_text(json.dumps(state))
    if mutation:
        with pytest.raises((ValueError, RuntimeError)):
            operator.series_bindings(manifest)
    else:
        assert (
            operator.series_bindings(manifest)["authorized_names"]
            == names("1e-3") + scheduled_names()
        )


def test_legacy_constant_optim_defaults_are_effectively_equal(comparison):
    operator, manifest = comparison
    baseline = operator.read_json(operator.BASELINE)
    for name, config in baseline["identity"]["configs"].items():
        for key in ("lr_schedule", "warmup_fraction", "min_lr_ratio"):
            config["optim"].pop(key, None)
        directory = operator.result_dir(operator.BASELINE, name)
        identity_path = directory / "run_identity.json"
        identity = operator.read_json(identity_path)
        identity["config"]["optim"] = config["optim"]
        identity_path.write_text(json.dumps(identity))
        from compare_attention.kaggle_artifacts import file_manifest  # pylint: disable=import-outside-toplevel

        baseline["runs"][name]["result_files"] = file_manifest(directory.parent, [identity_path])
    operator.BASELINE.write_text(json.dumps(baseline))
    assert operator.verify_lr_comparison(manifest, names("1e-3")[0])["matched"] is True


def test_unstaged_candidate_fails_closed_without_credentials(tmp_path, monkeypatch):
    operator = load_operator()
    monkeypatch.setattr(operator, "HERE", tmp_path)
    with pytest.raises(RuntimeError, match="not staged"):
        operator.main(["--check"])


@pytest.mark.parametrize("index", range(6))
def test_scheduler_matches_same_peak_control(comparison, monkeypatch, index):
    operator, manifest = comparison
    monkeypatch.setattr(
        operator,
        "load_helper",
        lambda: operator.SimpleNamespace(verify_pilot=lambda *_args: {"passed": True}),
    )
    monkeypatch.setattr(operator.protocol, "verify_artifact", lambda *_args: {"verified": True})
    name = scheduled_names()[index]
    control = f"experiments/fp16_lr_pilots/1e-3/full/{Path(name).parent.name}/seed_42.yaml"
    operator.write_json(
        manifest.parent / "lr-results.json",
        {
            "accepted": [
                {
                    "config": control,
                    "scientific": {"verified": True},
                    "comparison": {"matched": True},
                    "result_files": {
                        "run_identity.json": operator.file_manifest(
                            operator.result_dir(manifest, control),
                            [operator.result_dir(manifest, control) / "run_identity.json"],
                        )["run_identity.json"]
                    },
                }
            ]
        },
    )
    assert operator.verify_lr_comparison(manifest, name)["same_peak_constant_config"] == control


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "missing",
        "duplicate",
        "flops",
        "nonfinite",
        "lr",
        "optimizer",
        "constant_forgery",
        "order",
    ],
)
def test_all_lr_events_and_optimizer_are_replayed(mutation):
    operator = load_operator()
    optim = {
        "learning_rate": 0.001,
        "lr_schedule": "warmup_cosine",
        "warmup_fraction": 0.05,
        "min_lr_ratio": 0.1,
    }
    rows = [
        {"type": "learning_rate", "flops": 10, "learning_rate": 0.0005871603277361663},
        {"type": "learning_rate", "flops": 20, "learning_rate": 0.0001},
        {"type": "result", "steps": 2, "train_flops": 20},
    ]
    # Use the production stateless function for the exact platform float, not decimal approximation.
    from compare_attention.train.config import Optim  # pylint: disable=import-outside-toplevel
    from compare_attention.train.lr_schedule import next_learning_rate  # pylint: disable=import-outside-toplevel

    for i in range(2):
        rows[i]["learning_rate"] = next_learning_rate(Optim(**optim), 19, i * 10, 10)
    payload = {
        "step": 2,
        "flops": 20,
        "optimizer": {"param_groups": [{"lr": rows[1]["learning_rate"]}]},
    }
    if mutation == "missing":
        rows.pop(0)
    elif mutation == "duplicate":
        rows.insert(0, deepcopy(rows[0]))
    elif mutation == "flops":
        rows[0]["flops"] = 11
    elif mutation == "nonfinite":
        rows[0]["learning_rate"] = float("nan")
    elif mutation == "lr":
        rows[0]["learning_rate"] += 1e-12
    elif mutation == "optimizer":
        payload["optimizer"]["param_groups"][0]["lr"] = 0.001
    elif mutation == "constant_forgery":
        optim["lr_schedule"] = "constant"
    elif mutation == "order":
        rows = [rows[-1], *rows[:-1]]
    rows = [{"type": "step", "flops": 10}, rows[0], {"type": "step", "flops": 20}, *rows[1:]]
    if mutation:
        with pytest.raises(ValueError, match="(?i)(LR|learning.rate)"):
            operator.protocol.verify_history(rows, payload, optim, 19, 10)
    else:
        assert operator.protocol.verify_history(rows, payload, optim, 19, 10)["lr_events"] == 2
        constant = {"learning_rate": 0.001}
        payload["optimizer"]["param_groups"][0]["lr"] = 0.001
        assert (
            operator.protocol.verify_history(rows[-1:], payload, constant, 19, 10)["lr_events"] == 0
        )


@pytest.mark.parametrize("failure", [None, "raw_gate"])
def test_native_helpers_see_original_lr_metrics_without_temporary_views(
    comparison, monkeypatch, failure
):
    load_test_support().check_raw_helper_paths(comparison, monkeypatch, failure)


def test_source_evidence_requires_new_parent_pins(monkeypatch):
    operator = load_operator()
    monkeypatch.setattr(operator, "FULL_TEST_COUNT", None)
    with pytest.raises(RuntimeError, match="(?i)(pending|parent|regression)"):
        operator.evidence_bindings()


def scheduled_names():
    return [name.replace("fp16_lr_pilots", "fp16_lr_scheduler") for name in names("1e-3")]


def test_literal_protocol_order():
    operator = load_operator()
    assert operator.execution_names() == [
        "experiments/fp16_lr_pilots/1e-3/exact1/classic/seed_42.yaml",
        "experiments/fp16_lr_pilots/1e-3/exact1/deltanet/seed_42.yaml",
        "experiments/fp16_lr_pilots/1e-3/exact32/classic/seed_42.yaml",
        "experiments/fp16_lr_pilots/1e-3/exact32/deltanet/seed_42.yaml",
        "experiments/fp16_lr_pilots/1e-3/full/classic/seed_42.yaml",
        "experiments/fp16_lr_pilots/1e-3/full/deltanet/seed_42.yaml",
        "experiments/fp16_lr_scheduler/1e-3/exact1/classic/seed_42.yaml",
        "experiments/fp16_lr_scheduler/1e-3/exact1/deltanet/seed_42.yaml",
        "experiments/fp16_lr_scheduler/1e-3/exact32/classic/seed_42.yaml",
        "experiments/fp16_lr_scheduler/1e-3/exact32/deltanet/seed_42.yaml",
        "experiments/fp16_lr_scheduler/1e-3/full/classic/seed_42.yaml",
        "experiments/fp16_lr_scheduler/1e-3/full/deltanet/seed_42.yaml",
    ]
    assert set(operator.execution_names()).isdisjoint(names("1e-4"))


def names(lr):
    return [
        f"experiments/fp16_lr_pilots/{lr}/{stage}/{family}/seed_42.yaml"
        for stage in ("exact1", "exact32", "full")
        for family in FAMILIES
    ]


@pytest.mark.parametrize(
    "mutation", ["unaccepted", "runtime", "config", "extra_optim", "scientific"]
)
def test_scheduler_rejects_nonmatched_control(comparison, monkeypatch, mutation):
    load_test_support().check_rejected_control(comparison, monkeypatch, mutation)


@pytest.mark.parametrize("mutation", [None, "policy_ulp", "optimizer_ulp", "meaningful", "bool"])
def test_cross_platform_policy_tolerance_keeps_saved_optimizer_exact(mutation):
    load_test_support("native_test_support.py").check_portable_lr(mutation, load_operator())


@pytest.mark.parametrize("mutation", [None, "lr"])
def test_real_cpu_saved_artifact_through_native_gate_and_unchanged_helper(
    tmp_path, monkeypatch, mutation
):
    load_test_support("native_test_support.py").check_real_cpu_artifact(
        tmp_path, monkeypatch, mutation, load_operator()
    )


def load_test_support(filename="operator_test_support.py"):
    spec = importlib.util.spec_from_file_location("lr_test_support", HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_operator():
    return load_test_support().load_operator()


# Keep the full matrix contract in one end-to-end assertion loop.
# pylint: disable-next=too-many-locals
def test_complete_lr_matrix_raw_parsed_cost_and_legacy_bytes(tmp_path):
    return load_test_support().test_complete_lr_matrix_raw_parsed_cost_and_legacy_bytes(tmp_path)
