"""Staged Kaggle series: immutable private inputs and persistent sequential journal."""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from compare_attention.exploration_gate import verify_exploration_run
from compare_attention.kaggle_artifacts import (
    checksum,
    digest,
    file_manifest,
    snapshot,
    verify_files,
)
from compare_attention.kaggle_run import (
    Runner,
    _cli_run,
    build_wheel,
    download_output,
    poll_dataset_ready,
    poll_status,
    push_kernel,
    stage_requirements,
)
from compare_attention.kaggle_series_kernel import render_series_kernel


def _save(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _phase(path: Path, state: dict[str, Any], name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        state.setdefault("timings", []).append(
            {"phase": name, "seconds": time.perf_counter() - start}
        )
        _save(path, state)


def _metadata(folder: Path, identifier: str) -> None:
    (folder / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": identifier,
                "title": identifier.split("/")[1],
                "licenses": [{"name": "CC0-1.0"}],
                "isPrivate": True,
            }
        ),
        encoding="utf-8",
    )


def _freeze_source(root: Path, source: Path, files: dict):
    for name in files:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    verify_files(source, files)


def _bundle_configs(bundle: Path, release: Path, state: dict):
    identity = state["identity"]
    for name, payload in identity["configs"].items():
        payload = dict(payload)
        payload["data"] = f"/kaggle/input/datasets/{state['data_id']}"
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        slug = re.sub(r"[^a-z0-9]+", "-", str(Path(name).with_suffix("")).lower()).strip("-")
        slug = f"{slug[:27].rstrip('-')}-{digest(name)[:6]}-{identity['fingerprint'][:12]}"
        state["runs"][name] = {"slug": slug, "state": "staged"}
    bundle_files = file_manifest(bundle, [item for item in bundle.rglob("*") if item.is_file()])
    (bundle / "release-manifest.json").write_text(json.dumps(bundle_files), encoding="utf-8")
    with zipfile.ZipFile(release / "release.bundle", "w", zipfile.ZIP_DEFLATED) as archive:
        for item in bundle.rglob("*"):
            if item.is_file():
                archive.write(item, item.relative_to(bundle).as_posix())


def _stage_local(root: Path, prepared_data: Path, folder: Path, state: dict, run: Runner | None):
    identity = state["identity"]
    # A partial local build is never mistaken for a complete release.
    source, bundle, release, data = (
        folder / part for part in ("source", "bundle", "release", "data")
    )
    for directory in (source, bundle, release, data):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()
    _freeze_source(root, source, identity["source"])
    runner = run or _cli_run

    def frozen_run(command: list[str]):
        return runner([*command, "--directory", str(source)])

    with _phase(folder / "series.json", state, "build"):
        wheel = build_wheel(folder, run=frozen_run)
        shutil.copy2(wheel, bundle / wheel.name)
    with _phase(folder / "series.json", state, "export"):
        stage_requirements(bundle, run=lambda command: frozen_run([*command, "--locked"]))
    _bundle_configs(bundle, release, state)
    with _phase(folder / "series.json", state, "stage_data"):
        for name in identity["data"]["files"]:
            shutil.copy2(prepared_data / name, data / name)
        verify_files(data, identity["data"]["files"])
        (data / "data-manifest.json").write_text(json.dumps(identity["data"]), encoding="utf-8")
    _metadata(release, state["release_id"])
    _metadata(data, state["data_id"])
    return wheel.name


def stage_series(
    root: Path,
    configs: list[Path],
    prepared_data: Path,
    cache: Path,
    owner: str,
    run: Runner | None = None,
) -> Path:
    """Build/export once from frozen uncommitted content; no credentials or network."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", owner):
        raise ValueError("invalid Kaggle owner")
    root, prepared_data, cache = root.resolve(), prepared_data.resolve(), cache.resolve()
    identity = snapshot(root, configs, prepared_data)
    folder = cache / identity["fingerprint"]
    path = folder / "series.json"
    if path.exists():
        state = _read(path)
        if state["owner"] != owner:
            raise ValueError("cached series belongs to a different owner")
        if state.get("ready"):
            verify_files(folder, state["artifacts"])
            return path
    folder.mkdir(parents=True, exist_ok=True)
    state = {
        "identity": identity,
        "owner": owner,
        "release_id": f"{owner}/ca-release-{identity['fingerprint'][:24]}",
        "data_id": f"{owner}/ca-kv-{identity['data']['fingerprint'][:24]}",
        "published": {},
        "runs": {},
    }
    state["wheel"] = _stage_local(root, prepared_data, folder, state, run)
    release, data = folder / "release", folder / "data"
    state["archive_sha256"] = checksum(release / "release.bundle")
    state["artifacts"] = file_manifest(
        folder, [item for part in (release, data) for item in part.rglob("*") if item.is_file()]
    )
    state["ready"] = True
    _save(path, state)
    return path


def publish_series(path: Path, run: Runner | None = None) -> None:
    """Create-only private datasets; ready readback before journaling publication."""
    state = _read(path)
    if not state.get("ready"):
        raise ValueError("series staging incomplete")
    verify_files(path.parent, state["artifacts"])
    runner = run or _cli_run
    for kind in ("release", "data"):
        identifier = state[f"{kind}_id"]
        if state["published"].get(kind) == "ready":
            continue
        with _phase(path, state, f"upload_{kind}"):
            if state["published"].get(kind) != "created":
                code = (
                    "import sys; from kaggle.api.kaggle_api_extended import KaggleApi; "
                    "api=KaggleApi(); api.authenticate(); "
                    "result=api.dataset_create_new(sys.argv[1], "
                    "public=False, quiet=True, dir_mode='skip'); "
                    "sys.exit(1 if result.error else 0)"
                )
                completed = runner([sys.executable, "-c", code, str(path.parent / kind)])
                if completed.returncode != 0:
                    raise RuntimeError(f"private create failed: {identifier}")
                state["published"][kind] = "created"
                _save(path, state)
            poll_dataset_ready(identifier, poll=runner)
            state["published"][kind] = "ready"


def reuse_private_data(path: Path, source: Path, run: Runner | None = None) -> None:
    """Adopt a previously published private KV input after remote manifest verification.

    Every kernel still verifies all remote KV file checksums before training. No dataset
    is overwritten, and a differing remote manifest fails before any kernel submission.
    """
    state, previous = _read(path), _read(source)
    if not state.get("ready") or previous.get("published", {}).get("data") != "ready":
        raise ValueError("both staged inputs and an already published source are required")
    if any(state[key] != previous[key] for key in ("owner", "data_id")):
        raise ValueError("owner or immutable data identifier differs")
    if state["identity"]["data"] != previous["identity"]["data"]:
        raise ValueError("prepared data manifest differs")
    verify_files(source.parent / "data", previous["identity"]["data"]["files"])
    target = path.parent / "reused-data-proof"
    target.mkdir(exist_ok=True)
    code = "\n".join(
        [
            "import sys; from kaggle.api.kaggle_api_extended import KaggleApi",
            "api=KaggleApi(); api.authenticate(); identifier=sys.argv[1]",
            "items=api.dataset_list(mine=True, search=identifier.split('/')[-1])",
            "matches=[item for item in items if item.ref == identifier]",
            "if len(matches) != 1 or matches[0].is_private is not True:",
            "    raise RuntimeError('expected exact private dataset')",
            "api.dataset_download_file(identifier, 'data-manifest.json', path=sys.argv[2], "
            "force=True, quiet=True, unzip=True)",
        ]
    )
    runner = run or _cli_run
    with _phase(path, state, "verify_reused_data"):
        completed = runner([sys.executable, "-c", code, state["data_id"], str(target)])
        if completed.returncode != 0:
            raise RuntimeError("private data reuse verification failed")
        if _read(target / "data-manifest.json") != state["identity"]["data"]:
            raise ValueError("remote data manifest differs")
        poll_dataset_ready(state["data_id"], poll=runner)
        state["published"]["data"] = "ready"
        state["reused_data"] = {
            "source_series": str(source.resolve()),
            "remote_manifest_sha256": checksum(target / "data-manifest.json"),
        }


def _verify_result(output: Path, state: dict[str, Any], name: str) -> dict[str, Any]:
    # pylint: disable=too-many-locals
    entry = state["runs"][name]
    receipt = _read(output / "result-manifest.json")
    expected = {
        "series": state["identity"]["fingerprint"],
        "config": name,
        "id": f"{state['owner']}/{entry['slug']}",
        "slug": entry["slug"],
    }
    if receipt["identity"] != expected:
        raise ValueError("result identity mismatch; refusing latest output from another kernel")
    verify_files(output, receipt["files"])
    prefix = entry["slug"] + "/"
    for artifact in ("model.pt", "metrics.jsonl"):
        if prefix + artifact not in receipt["files"]:
            raise ValueError(f"missing result artifact: {artifact}")
    records = [
        json.loads(line)
        for line in (output / prefix / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    final = [row for row in records if row.get("type") == "result"]
    if len(final) != 1 or final[0]["steps"] < 1:
        raise ValueError("missing unique final result")
    result = final[0]
    experiment = state["identity"]["configs"][name]
    budget = experiment["schedule"]["train_flops"]
    early = result.get("stop_reason") == "early_stopping"
    scheduled = experiment.get("optim", {}).get("lr_schedule", "constant") != "constant"
    if scheduled or early or result["train_flops"] < budget or result.get("provenance") is not None:
        if early or result["train_flops"] < budget:
            patience = experiment["schedule"].get("early_stopping_patience", 0)
            if (
                not isinstance(patience, int)
                or isinstance(patience, bool)
                or patience <= 0
                or not early
            ):
                raise ValueError("Below-budget results require explicitly requested early stopping")
        if prefix + "run_identity.json" not in receipt["files"]:
            raise ValueError("missing checksum-bound run identity")
        expected_config = {
            "data": f"/kaggle/input/datasets/{state['data_id']}",
            "model": experiment["model"],
            "schedule": experiment["schedule"],
            "optim": experiment["optim"],
            "dataset": state["identity"]["data"]["spec"],
        }
        report = verify_exploration_run(output / prefix, expected_config=expected_config)
        return {
            **receipt,
            **{key: report[key] for key in ("stop_reason", "steps", "train_flops", "quality")},
        }
    if result.get("stop_reason", "budget_exhausted") != "budget_exhausted" or not all(
        math.isfinite(v) for v in result["quality"].values()
    ):
        raise ValueError("invalid final quality or incomplete FLOP budget")
    return {
        **receipt,
        "stop_reason": "budget_exhausted",
        "steps": result["steps"],
        "train_flops": result["train_flops"],
        "quality": result["quality"],
    }


def _push_version(staging: Path, entry: dict, accelerator: str, timeout: int, runner: Runner):
    def capture(command):
        completed = runner(command)
        if completed.returncode == 0:
            match = re.search(r"Kernel version (\d+) successfully pushed", completed.stdout)
            if match is None or "could not be added" in completed.stdout:
                raise RuntimeError("ambiguous kernel push: version or valid inputs not confirmed")
            entry["version"] = int(match[1])
        return completed

    push_kernel(staging, accelerator, timeout, run=capture)


def _write_kernel(staging, kernel):
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "kernel.py").write_text(kernel.code, encoding="utf-8")
    (staging / "kernel-metadata.json").write_text(json.dumps(kernel.metadata), encoding="utf-8")


def run_selected(
    path: Path,
    names: list[str],
    *,
    credentials_dataset: str | None = None,
    accelerator: str = "NvidiaTeslaT4",
    timeout: int = 9 * 3600,
    poll_seconds: int = 60,
    run: Runner | None = None,
) -> None:
    """Run only explicit selected configs; download/verify fully before next push.

    Recalling after a failed download resumes the completed run, without build or push.
    Never resubmit a journaled submission, even if polling previously failed.
    """
    # pylint: disable=too-many-locals
    state = _read(path)
    if (
        not names
        or len(names) != len(set(names))
        or any(name not in state["runs"] for name in names)
    ):
        raise ValueError("select unique config names from the staged series")
    if state["published"] != {"release": "ready", "data": "ready"}:
        raise ValueError("publish both immutable inputs before running")
    runner = run or _cli_run
    for name in names:
        entry = state["runs"][name]
        staging = path.parent / "runs" / entry["slug"]
        output = staging / "output"
        if entry["state"] in {"verified", "early_stopped"}:
            receipt = _verify_result(output, state, name)
            expected_reason = (
                "early_stopping" if entry["state"] == "early_stopped" else "budget_exhausted"
            )
            if receipt["stop_reason"] != expected_reason:
                raise ValueError("terminal journal state differs from verified stop reason")
            continue
        if entry["state"] in {"failed", "submitting"}:
            raise RuntimeError(
                "failed or ambiguous submission; reconcile journal manually, never auto-retrain"
            )
        if entry["state"] == "staged":
            _write_kernel(
                staging,
                render_series_kernel(state, name, credentials_dataset, accelerator=accelerator),
            )
            with _phase(path, state, "push:" + name):
                entry["state"] = "submitting"
                _save(path, state)
                _push_version(staging, entry, accelerator, timeout, runner)
                entry["state"] = "submitted"
        identifier = f"{state['owner']}/{entry['slug']}/{entry['version']}"
        if entry["state"] == "submitted":
            with _phase(path, state, "wait:" + name):
                status = poll_status(identifier, poll_seconds, poll=runner, timeout=float(timeout))
                if status != "complete":
                    entry["state"] = "failed"
                    raise RuntimeError(f"kernel {identifier}: {status}")
                entry["state"] = "completed"
        if entry["state"] == "completed":
            with _phase(path, state, "download:" + name):
                download_output(
                    output, identifier, run=lambda command: runner([*command, "--force"])
                )
            with _phase(path, state, "verify_result:" + name):
                receipt = _verify_result(output, state, name)
                entry["result_files"] = receipt["files"]
                entry.update(
                    {
                        key: receipt[key]
                        for key in ("stop_reason", "steps", "train_flops", "quality")
                    }
                )
                entry["state"] = (
                    "early_stopped" if receipt["stop_reason"] == "early_stopping" else "verified"
                )
