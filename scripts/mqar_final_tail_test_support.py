"""Actual Kaggle SDK builders; only HTTP/blob transport and credentials are synthetic."""

import base64
import contextlib
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS

from mqar_final_tail_fixture_plan import fixture_plan
from mqar_final_tail_sdk_fixture import read_response, sdk_api
from mqar_final_tail_test_layout import (
    DATA,
    MOUNT,
    ORIGINAL_SOURCES,
    SCIENCE,
)


class LazyFiles:
    """Metadata listing without copying immutable arrays; request bytes live only in memory."""

    def __init__(self, root, pins):
        self.root, self.pins = Path(root), pins

    def items(self):
        class Size:
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

        return [(n, Size(p["bytes"])) for n, p in self.pins.items()]

    def __getitem__(self, name):
        return (self.root / name).read_bytes()


class World:
    def __init__(self, root, monkeypatch):
        from mqar_final_tail_host_contract import CONTROL, PRIVATE, SCIENCE, activate, value_sha
        from mqar_final_tail_host_contract import DATA as DATA_REF

        self.root, self.monkeypatch = root, monkeypatch
        self.plan = fixture_plan(root)
        activate(self.plan)
        from kagglesdk.blobs.services.blob_api_service import BlobApiClient
        from kagglesdk.datasets.services.dataset_api_service import DatasetApiClient
        from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
        from seed202_pilot.provider import inherited_provider_class, private_environment_bytes

        self.source = Path(self.plan["candidate_root"])
        self.requests, self.writes, self.uploads, self.fsynced = [], [], {}, set()
        from kagglesdk.kaggle_http_client import KaggleHttpClient

        self.http = KaggleHttpClient(api_token="SYNTHETIC_BEARER_FIXTURE")
        self.http._init_session()
        self.serialized = []
        self.lost_ack, self.fail_fsync, self.drift = None, False, {}
        self.science = MOUNT / "payload/inherited/sorting-science.zip.bin"
        self.parent = MOUNT / "payload/inherited/parent-candidate03.zip.bin"
        self.private = root / "SYNTHETIC.env"
        self.private.write_text(
            "CLEARML_API_HOST=https://fixture-api.invalid\n"
            "CLEARML_WEB_HOST=https://fixture-web.invalid\n"
            "CLEARML_FILES_HOST=https://fixture-files.invalid\n"
            "CLEARML_API_ACCESS_KEY=FIXTURE_ONLY\nCLEARML_API_SECRET_KEY=FIXTURE_ONLY\n"
            "HF_TOKEN=hf_FIXTURE_ONLY\nHF_REPO=fixture/repo\n"
        )
        self.datasets = {
            SCIENCE: {"scientific.zip.bin": self.science.read_bytes()},
            PRIVATE: {"private.env": private_environment_bytes(self.private)},
            DATA_REF: LazyFiles(DATA, self.plan["data_files"]),
        }
        self.kernels = {CONTROL: NS(metadata=NS(ref=CONTROL, is_private=True), status="COMPLETE")}
        for row, source in zip(self.plan["predecessors"], ORIGINAL_SOURCES, strict=True):
            raw = source.read_bytes()
            self.kernels[row["target"]] = NS(
                metadata=NS(
                    ref=row["target"],
                    id=row["numeric_id"],
                    current_version_number=1,
                    is_private=True,
                    dataset_data_sources=list(
                        reversed([s[:-2] for s in row["requested_versioned_inputs"]])
                    ),
                ),
                blob=NS(source=raw.decode()),
                status="ERROR",
            )
        kind = inherited_provider_class(self.parent)
        self.provider = kind.__new__(kind)
        self.provider.owner = "zimike"
        self.api = NS(
            CONFIG_NAME_TOKEN="token",
            config_values={"token": "SYNTHETIC"},
            _introspect_token=lambda _: "zimike",
            build_kaggle_client=self.context,
        )
        self.provider.api = self.api
        self.client = NS(
            datasets=NS(dataset_api_client=DatasetApiClient(self)),
            kernels=NS(kernels_api_client=KernelsApiClient(self)),
            blobs=NS(blob_api_client=BlobApiClient(self)),
            _http_client=self,
        )
        sdk_api(monkeypatch, self)
        self.review = {
            "status": "PASS",
            "independent": True,
            "reviewer": "SYNTHETIC_INDEPENDENT_FIXTURE_NOT_REVIEW",
            "scope": "MQAR202_FINAL10780_TAIL_HOST_TWO_STAGE_V1",
            "plan_sha256": value_sha(self.plan),
            "policy": "RECOVERED_TAIL_WITH_ORIGINAL_TELEMETRY_GAP",
        }
        self.plan_path, self.review_path, self.lifetime = (
            root / "plan.json",
            root / "review.json",
            root / "lifetime.json",
        )
        self.plan_path.write_text(json.dumps(self.plan))
        self.review_path.write_text(json.dumps(self.review))
        self.lifetime.write_text(
            json.dumps(
                {
                    "source": "https://www.kaggle.com/docs/notebooks",
                    "observed_unix_seconds": time.time(),
                    "documentation_text": (
                        "12 hours execution time for CPU and GPU notebook sessions. "
                        "Save & Run All creates a new session with a completely clean state"
                    ),
                }
            )
        )
        import mqar_final_tail_host_cli
        from mqar_final_tail_host_cli import execute

        monkeypatch.setattr(mqar_final_tail_host_cli, "_provider", lambda _: self.provider)
        import seed202_pilot.provider as provider_module

        monkeypatch.setattr(
            provider_module, "inherited_provider_class", lambda _: lambda *_: self.provider
        )
        import mqar_final_tail_host_services

        monkeypatch.setattr(
            mqar_final_tail_host_services,
            "absence",
            lambda *a: {"status": "ABSENT_FRESH_TAIL_SERVICES", "nonce": self.plan["nonce"]},
        )
        monkeypatch.setattr(
            mqar_final_tail_host_cli,
            "_services",
            lambda a, p: {"status": "ABSENT_FRESH_TAIL_SERVICES", "nonce": p["nonce"]},
        )
        self.errors = []

        def observed(args):
            try:
                return execute(args)
            except Exception as error:
                self.errors.append((type(error).__name__, error.args))
                raise

        monkeypatch.setattr(mqar_final_tail_host_cli, "execute", observed)
        fsync = os.fsync

        def synced(fd):
            if self.fail_fsync:
                raise OSError("SYNTHETIC_FSYNC_FAILURE")
            fsync(fd)
            stat = os.fstat(fd)
            self.fsynced.add((stat.st_dev, stat.st_ino))

        monkeypatch.setattr(os, "fsync", synced)
        import socket

        monkeypatch.setattr(
            socket.socket,
            "connect",
            lambda *a: (_ for _ in ()).throw(AssertionError("NETWORK_FORBIDDEN")),
        )

    def context(self):
        return contextlib.nullcontext(self.client)

    @staticmethod
    def http_error(code):
        from requests import HTTPError, Response

        r = Response()
        r.status_code = code
        raise HTTPError("SYNTHETIC_SECRET_NOT_LOGGED", response=r)

    def before_write(self):
        journal = Path(self.plan["journals"][self.action])
        for p in [
            journal.parent / "host-single-writer.lock",
            *(
                journal / n
                for n in (
                    "single-writer.lock",
                    "intent.json",
                    "approval.json",
                    "fresh-under-lock.json",
                )
            ),
        ]:
            stat = p.stat()
            assert (stat.st_dev, stat.st_ino) in self.fsynced

    def upload_complete(self, path, url, quiet, resume=False):
        self.before_write()
        path = Path(path)
        self.uploads["SYNTHETIC-" + path.name] = (path.name, path.read_bytes())
        return self.__dict__["sdk_module"].ResumableUploadResult.COMPLETE

    def call(self, service, method, request, response_type):
        prepared = self.http._prepare_request(service, method, request)
        assert prepared.headers["Authorization"] == "Bearer SYNTHETIC_BEARER_FIXTURE"
        assert json.loads(prepared.body) == type(request).to_dict(request)
        assert prepared.url.startswith("https://api.kaggle.com/v1/")
        self.serialized.append(method)
        self.requests.append((service, method, request))
        if method == "GetDatasetStatus":
            ref = request.owner_slug + "/" + request.dataset_slug
            if ref not in self.datasets:
                self.http_error(403)
        if method in {
            "GetDatasetStatus",
            "ListDatasetFiles",
            "GetKernelSessionStatus",
            "GetAcceleratorQuotaStatistics",
        }:
            return read_response(self, method, request, response_type)
        if method == "ListDatasets":
            return NS(
                datasets=[
                    NS(ref=r, is_private=self.drift.get(r, {}).get("is_private", True))
                    for r in self.datasets
                ]
                if request.page == 1
                else [],
                next_page_token="",
            )
        if method == "ListKernels":
            return NS(
                kernels=[r.metadata for r in self.kernels.values()] if request.page == 1 else [],
                next_page_token="",
            )
        if method == "GetDataset":
            ref = request.owner_slug + "/" + request.dataset_slug
            if ref not in self.datasets:
                self.http_error(403)
            return NS(
                **{
                    **dict(
                        ref=ref,
                        id=1000 + list(self.datasets).index(ref),
                        is_private=True,
                        current_version_number=1,
                    ),
                    **self.drift.get(ref, {}),
                }
            )
        if method == "GetKernel":
            ref = request.user_name + "/" + request.kernel_slug
            if ref not in self.kernels:
                self.http_error(403)
            return self.kernels[ref]
        if method == "DownloadDataset":
            assert request.dataset_version_number == 1
            ref = request.owner_slug + "/" + request.dataset_slug
            raw = self.datasets[ref][request.file_name]
            return NS(
                content=raw,
                url="https://storage.googleapis.com/synthetic/" + request.file_name,
                headers={
                    "x-goog-generation": "123",
                    "Content-Length": str(len(raw)),
                    "x-goog-hash": "md5="
                    + base64.b64encode(hashlib.md5(raw, usedforsecurity=False).digest()).decode(),
                },
                raise_for_status=lambda: None,
                close=lambda: None,
            )
        if method == "StartBlobUpload":
            self.before_write()
            response = response_type()
            response.token, response.create_url = (
                "SYNTHETIC-" + request.name,
                "https://fixture.invalid/" + request.name,
            )
            return response
        if method == "CreateDataset":
            self.before_write()
            ref = request.owner_slug + "/" + request.slug
            assert request.is_private is True
            self.datasets[ref] = dict(self.uploads[f.token] for f in request.files)
            self.writes.append((method, ref))
            if self.lost_ack == self.action:
                raise TimeoutError("SYNTHETIC_SECRET")
            return NS(error=None, ref=ref)
        if method == "ListKernelSessionOutput":
            return NS(
                files=[
                    NS(
                        file_name=n,
                        url="https://host-fixture.invalid/" + n,
                        sha256=hashlib.sha256(raw).hexdigest(),
                    )
                    for n, raw in self.outputs.items()
                ],
                log="",
                next_page_token="",
            )
        if method == "SaveKernel":
            self.before_write()
            metadata = json.loads((self.source / "kernel/kernel-metadata.json").read_bytes())
            assert request.dataset_data_sources == metadata["dataset_sources"]
            values = {
                k: v
                for k, v in metadata.items()
                if k
                in (
                    "title",
                    "language",
                    "kernel_type",
                    "is_private",
                    "enable_gpu",
                    "enable_tpu",
                    "enable_internet",
                    "machine_shape",
                )
            }
            values.update(
                ref=metadata["id"],
                id=199999999,
                current_version_number=1,
                dataset_data_sources=list(reversed([s[:-2] for s in metadata["dataset_sources"]])),
                kernel_data_sources=[],
                competition_data_sources=[],
                model_data_sources=[],
            )
            self.kernels[metadata["id"]] = NS(
                metadata=NS(**values), blob=NS(source=request.text), status="QUEUED"
            )
            self.writes.append((method, request.slug))
            if self.lost_ack == self.action:
                raise TimeoutError("SYNTHETIC_SECRET")
            return NS(error=None, ref=request.slug, version_number=1)
        raise AssertionError(method)

    def command(self, command, action="publish", **kwargs):
        import runpy
        import sys

        import mqar_final_tail_host_cli

        self.action = action
        args = [
            command,
            "--plan",
            self.plan_path,
            "--review",
            self.review_path,
            "--action",
            action,
            "--science-archive",
            self.science,
            "--parent-archive",
            self.parent,
            "--private",
            self.private,
            "--science-dir",
            SCIENCE,
            "--lifetime",
            self.lifetime,
            "--auth-env",
            self.private,
        ]
        for key, value in kwargs.items():
            args += ["--" + key.replace("_", "-"), value]
        with self.monkeypatch.context() as selected:
            selected.setattr(sys, "argv", [mqar_final_tail_host_cli.__file__, *map(str, args)])
            try:
                runpy.run_path(mqar_final_tail_host_cli.__file__, run_name="__main__")
            except SystemExit as result:
                return result.code
        raise AssertionError("SAVED_CLI_NO_EXIT")

    def prepare(self, action):
        receipt = self.root / (action + "-receipt.json")
        assert self.command("prepare", action, output=receipt) == 0, self.errors
        proof, approval = (
            self.root / (action + "-proof.json"),
            self.root / (action + "-approval.json"),
        )
        assert self.command("probe", action, receipt=receipt, output=proof) == 0, self.errors
        assert self.command("seal", action, proof=proof, output=approval) == 0, self.errors
        return receipt, approval
