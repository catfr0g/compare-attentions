"""Exact all-file MQAR resume publication through the existing audited SDK boundary."""

from pathlib import Path

from seed202_pilot.mqar_authority import MODE, require, validate


def local_bundle(deploy, deployment):
    return validate(
        Path(deploy) / "resume-input",
        deployment["evidence_sha256"],
        deployment["authority_sha256"],
        publication_folder=True,
    )["files"]


def readback(provider, target, filename, payload, deploy, deployment):
    from seed202_pilot.provider import dataset_files_readback, dataset_readback

    if filename != "resume.zip.bin" or deployment.get("mode") != MODE:
        return dataset_readback(provider, target, filename, payload)
    pins = local_bundle(deploy, deployment)
    before = provider.dataset(target).id
    require(type(before) is int and before > 0, "RESUME_NUMERIC_ID")
    result = dataset_files_readback(
        provider, target, {name: Path(deploy) / "resume-input" / name for name in pins}
    )
    require(
        provider.dataset(target).id == before and result["files"] == pins,
        "RESUME_IDENTITY_AND_ALL_BYTES",
    )
    return {key: result[key] for key in ("target", "head", "private", "ready", "files")} | {
        "numeric_id": before,
        "bytes": len(payload),
        "sha256": pins["resume.zip.bin"]["sha256"],
    }


def transport(capability, method, request):
    """Finite multi-file upload state machine; the outer gate rechecks leases per call."""
    from seed202_pilot.fresh_mode import source_binding

    plan = capability.approval["plan"]
    pins = source_binding(plan)["resume_local"]["files"]
    uploaded = getattr(capability, "mqar_uploaded", set())
    if method == "StartBlobUpload":
        require(
            capability.state in {"STATUS_CHECKED", "BYTES_SENT"}
            and request.name in pins
            and request.name not in uploaded
            and type(request.content_length) is int
            and request.content_length == pins[request.name]["bytes"],
            "CAPABILITY_ALL_FILES_UPLOAD_SCOPE",
        )
        uploaded.add(request.name)
        capability.mqar_uploaded = uploaded
        capability.state = "UPLOAD_STARTED"
    else:
        wanted = plan["deployment"]["resume_input_target"].rsplit("/", 1)[0]
        require(
            method == "CreateDataset"
            and capability.state == "BYTES_SENT"
            and uploaded == set(pins)
            and request.owner_slug + "/" + request.slug == wanted
            and request.is_private is True
            and len(request.files) == len(pins)
            and len({row.token for row in request.files}) == len(pins)
            and all(type(row.token) is str and row.token for row in request.files),
            "CAPABILITY_ALL_FILES_CREATE_SCOPE",
        )
        capability.state = "DONE"
