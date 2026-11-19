"""Finite tail SDK capability, inherited SDK builders and actual exact readbacks."""

import builtins
import json
import time
from pathlib import Path
from types import FunctionType, SimpleNamespace

from mqar_final_tail_host_contract import exact
from mqar_final_tail_native import digest, require
from mqar_final_tail_runner import create_json


class TailCapability:
    """Live one-action capability bound to durable authority, both locks and held leases."""

    def __init__(self, authority, proof, journal):
        from seed202_pilot.preflight import lock_identity

        self.authority, self.proof, self.journal = authority, proof, Path(journal)
        self.plan, self.action = authority["source_authority"]["plan"], authority["action"]
        self.locks = [
            lock_identity(self.journal / "single-writer.lock"),
            lock_identity(self.journal.parent / "host-single-writer.lock"),
        ]
        self.state, self.uploaded = "NEW", []
        self.check()

    def check(self):
        from mqar_final_tail_host_actions import recorded
        from mqar_final_tail_host_live import validate_live
        from mqar_final_tail_host_prepare import assert_lease
        from seed202_pilot.preflight import lock_identity

        require(self.state != "DONE", "TAIL_CAPABILITY_CONSUMED")
        require(
            exact(recorded(self.plan, self.action), self.authority), "TAIL_CAPABILITY_AUTHORITY"
        )
        require(all(lock_identity(r["path"]) == r for r in self.locks), "TAIL_CAPABILITY_LOCK")
        require(
            exact(json.loads((self.journal / "fresh-under-lock.json").read_bytes()), self.proof),
            "TAIL_CAPABILITY_PROOF",
        )
        require(
            not any(
                (self.journal / n).exists()
                for n in ("unknown.json", "acknowledged.json", "reconciled-ack.json")
            ),
            "TAIL_CAPABILITY_GET_ONLY",
        )
        validate_live(self.plan, self.proof, self.action)
        assert_lease(self.proof["preparation"])
        require(
            self.proof["lock"] == self.locks[0] and 0 <= time.time() - self.proof["started"] < 120,
            "TAIL_CAPABILITY_EXPIRED",
        )

    def status(self, target, code):
        self.check()
        require(
            self.action == "publish"
            and self.state == "NEW"
            and target == self.plan["dataset_target"][:-2]
            and type(code) is int
            and code in {403, 404},
            "TAIL_CAPABILITY_STATUS",
        )
        create_json(
            self.journal / "sdk-status-observation.json",
            {
                "action": self.action,
                "target": target,
                "http_status": code,
                "classification": "FORBIDDEN_AMBIGUOUS" if code == 403 else "HTTP_NOT_FOUND",
                "proof_sha256": digest(self.journal / "fresh-under-lock.json"),
            },
        )
        self.state = "STATUS_CHECKED"

    def transport(self, method, request):
        self.check()
        if method == "SaveKernel":
            source = (Path(self.plan["candidate_root"]) / "kernel/kernel.py").read_text(
                encoding="utf-8"
            )
            require(
                self.action == "submit"
                and self.state == "NEW"
                and request.slug == self.plan["kernel_target"]
                and request.text == source
                and request.is_private is True
                and request.enable_gpu is True
                and request.enable_tpu is False
                and request.enable_internet is True
                and request.kernel_data_sources == []
                and request.competition_data_sources == []
                and request.model_data_sources == []
                and request.machine_shape == "NvidiaTeslaT4"
                and request.dataset_data_sources == self.plan["requested_versioned_inputs"],
                "TAIL_CAPABILITY_SAVE",
            )
            self.state = "DONE"
        elif method == "StartBlobUpload":
            files = self.plan["dataset_files"]
            require(
                self.action == "publish"
                and self.state in {"STATUS_CHECKED", "BYTES_SENT"}
                and request.name in files
                and request.name not in self.uploaded
                and type(request.content_length) is int
                and request.content_length == files[request.name]["bytes"],
                "TAIL_CAPABILITY_UPLOAD",
            )
            self.uploaded.append(request.name)
            self.state = "UPLOAD_STARTED"
        else:
            require(
                method == "CreateDataset"
                and self.action == "publish"
                and self.state == "BYTES_SENT"
                and set(self.uploaded) == set(self.plan["dataset_files"])
                and request.owner_slug + "/" + request.slug == self.plan["dataset_target"][:-2]
                and request.is_private is True
                and len(request.files) == len(self.plan["dataset_files"]),
                "TAIL_CAPABILITY_CREATE",
            )
            self.state = "DONE"


def create_dataset(provider, folder, capability):
    """Keep the exact audited function code; bind only its exact capability type import."""
    from seed202_pilot.provider import create_private_dataset

    original = builtins.__import__

    def importer(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "seed202_pilot.preflight"
            and tuple(fromlist) == ("CreateCapability",)
            and level == 0
        ):
            return SimpleNamespace(CreateCapability=TailCapability)
        return original(name, globals, locals, fromlist, level)

    namespace = {
        **create_private_dataset.__globals__,
        "__builtins__": {**vars(builtins), "__import__": importer},
    }
    function = FunctionType(
        create_private_dataset.__code__,
        namespace,
        create_private_dataset.__name__,
        create_private_dataset.__defaults__,
    )
    function.__kwdefaults__ = create_private_dataset.__kwdefaults__
    return function(provider, folder, capability=capability)


def mutate(plan, action, provider, capability):
    if action == "publish":
        response = create_dataset(provider, Path(plan["journal_directory"]) / "dataset", capability)
        require(
            not response.error and response.ref == plan["dataset_target"][:-2],
            "TAIL_UNKNOWN_ACK_GET_ONLY",
        )
    else:
        from seed202_pilot.provider import capability_transport

        with capability_transport(provider.api, capability):
            response = provider.api.kernels_push(
                str(Path(plan["candidate_root"]) / "kernel"), acc="NvidiaTeslaT4", no_run=False
            )
        require(
            not response.error
            and response.ref == plan["kernel_target"]
            and type(response.version_number) is int
            and response.version_number == 1,
            "TAIL_UNKNOWN_ACK_GET_ONLY",
        )


def readback(plan, action, provider):
    from seed202_pilot.provider import dataset_files_readback, kernel_readback

    if action == "publish":
        files = {n: Path(plan["journal_directory"]) / "dataset" / n for n in plan["dataset_files"]}
        return dataset_files_readback(provider, plan["dataset_target"][:-2], files)
    candidate = Path(plan["candidate_root"])
    observed = kernel_readback(
        provider,
        json.loads((candidate / "kernel/kernel-metadata.json").read_bytes()),
        (candidate / "kernel/kernel.py").read_bytes(),
    )
    from mqar_final_tail_host_actions import effect

    return {key: observed[key] for key in effect(plan, action)}
