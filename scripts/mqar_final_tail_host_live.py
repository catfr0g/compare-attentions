"""Fresh tail-only namespace/quota/private controls in the preserved 120-second LIVE domain."""

import math
import os
import re
import shutil
import time
from pathlib import Path

from mqar_final_tail_host_contract import ACTIONS, CONTROL, DATA, PRIVATE, SCIENCE, exact, value_sha
from mqar_final_tail_native import require


def predecessors(plan, provider):
    """Fresh exact both real ERROR versions; SDK bare-input order is a set domain."""
    result = []
    for expected in plan["predecessors"]:
        ref = expected["target"]
        remote, head = provider.kernel(ref, 1), provider.kernel(ref)
        for row in (remote.metadata, head.metadata):
            require(
                row.ref == ref
                and type(row.id) is int
                and row.id == expected["numeric_id"]
                and type(row.current_version_number) is int
                and row.current_version_number == 1
                and row.is_private is True,
                "TAIL_PREDECESSOR_IDENTITY",
            )
            wanted = [s[:-2] for s in expected["requested_versioned_inputs"]]
            require(
                type(row.dataset_data_sources) is list
                and len(set(row.dataset_data_sources))
                == len(row.dataset_data_sources)
                == len(wanted)
                and set(row.dataset_data_sources) == set(wanted),
                "TAIL_PREDECESSOR_INPUTS",
            )
        require(
            __import__("hashlib").sha256(remote.blob.source.encode()).hexdigest()
            == expected["source_sha256"],
            "TAIL_PREDECESSOR_SOURCE",
        )
        require(
            provider.api.kernels_status(ref + "/1").status.name == "ERROR", "TAIL_PREDECESSOR_ERROR"
        )
        result.append(expected)
    return result


def local_evidence(plan, lock):
    import psutil

    process = psutil.Process()
    excluded = {process.pid, *(p.pid for p in process.parents())}
    active = [
        p.pid
        for p in psutil.process_iter(("pid", "cmdline"))
        if p.pid not in excluded
        and any(
            s in " ".join(p.info.get("cmdline") or [])
            for s in ("mqar_final_tail_host_cli.py", "seed202_pilot")
        )
    ]
    return {
        "writer_lock_held": Path(lock).is_file(),
        "active_processes": active,
        "disk_bytes": shutil.disk_usage(plan["candidate_root"]).free,
        "disk_required_bytes": 10_000_000_000,
        "pid": os.getpid(),
    }


def probe(plan, action, wrapper, provider, private, lifetime, service_absence, lock):
    from kagglesdk.datasets.types.dataset_api_service import DatasetSelectionGroup as Group
    from mqar_final_tail_host_prepare import assert_lease
    from seed202_pilot.preflight import lock_identity, observe_absent
    from seed202_pilot.preparation_auth import tag
    from seed202_pilot.prepared_remote import private_identity, refresh
    from seed202_pilot.provider import dataset_files_readback, private_environment_bytes

    assert_lease(wrapper)
    started = time.time()
    held = lock_identity(lock)
    require(action in ACTIONS, "TAIL_ACTION")
    require(
        provider.api._introspect_token(provider.api.config_values[provider.api.CONFIG_NAME_TOKEN])
        == "zimike",
        "TAIL_OWNER",
    )
    datasets = provider.namespace_rows("datasets", Group.DATASET_SELECTION_GROUP_MY)
    privates = provider.namespace_rows("datasets", Group.DATASET_SELECTION_GROUP_MY_PRIVATE)
    kernels = provider.namespace_rows("kernels")
    require(
        all(type(r.is_private) is bool for r in datasets.values())
        and set(privates) == {r for r, v in datasets.items() if v.is_private}
        and all(v.is_private is True for v in privates.values()),
        "TAIL_PRIVATE_NAMESPACE",
    )
    require(
        SCIENCE in privates and DATA in privates and PRIVATE in privates and CONTROL in kernels,
        "TAIL_NAMESPACE_CONTROLS",
    )
    require(
        provider.dataset(SCIENCE).is_private is True
        and provider.kernel(CONTROL).metadata.is_private is True,
        "TAIL_EXACT_CONTROLS",
    )
    ref, target = plan["dataset_target"][:-2], plan["kernel_target"]
    require(
        (ref in datasets) == (action == "submit") and target not in kernels,
        "TAIL_CREATE_ONLY_NAMESPACE",
    )
    observations = {target: observe_absent(provider, "kernel", target)}
    if action == "publish":
        observations[ref] = observe_absent(provider, "dataset", ref)
    active = []
    for name in kernels:
        if name.startswith(("zimike/seed202-", "zimike/sorting-quality-")):
            status = provider.api.kernels_status(name).status.name
            if status not in {"ERROR", "COMPLETE", "CANCEL_ACKNOWLEDGED"}:
                active.append({"target": name, "status": status})
    require(not active, "TAIL_ACTIVE_ALLOCATION")
    receipt = wrapper["receipt"]
    refresh(provider, receipt["remote"])
    prior = predecessors(plan, provider)
    require(exact(private_identity(provider), receipt["private_identity"]), "TAIL_PRIVATE_DRIFT")
    expected = private_environment_bytes(private)
    equal = dataset_files_readback(
        provider, PRIVATE, {"private.env": expected}, disclose_pins=False
    )
    del expected
    require(exact(private_identity(provider), receipt["private_identity"]), "TAIL_PRIVATE_DRIFT")
    service = service_absence()
    quota = provider.api.quota_view().gpu_quota
    total, used, reserved = (
        getattr(quota, k).total_seconds()
        for k in ("total_time_allowed", "time_used", "time_reserved")
    )
    proof = {
        "schema": "TAIL_LIVE_TWO_STAGE_V1",
        "plan_sha256": value_sha(plan),
        "action": action,
        "source": "LIVE_PROVIDER_API",
        "started": started,
        "finished": time.time(),
        "preparation": wrapper,
        "predecessors": prior,
        "service_evidence": service,
        "private_evidence": {
            "private_env_equal_memory_only": equal["all_bytes_equal"],
            **receipt["private_identity"],
        },
        "source_evidence": receipt["source_evidence"],
        "namespace": {
            "datasets": {r: v.is_private for r, v in datasets.items()},
            "private_datasets": sorted(privates),
            "kernels": sorted(kernels),
            "observations": observations,
            "owner": "zimike",
            "complete_owner_namespace": True,
            "controls": {
                "dataset": SCIENCE,
                "kernel": CONTROL,
                "exact_metadata_status": 200,
                "private": True,
            },
        },
        "quota": {
            "total": total,
            "used": used,
            "reserved": reserved,
            "remaining": total - used - reserved,
            "planning": (total - used - reserved) * 0.8,
            "required": 9000,
        },
        "lifetime": lifetime,
        "active": active,
        "local": local_evidence(plan, lock),
        "lock": held,
    }
    require(held == lock_identity(lock), "TAIL_LOCK_DRIFT")
    proof["authentication"] = tag(plan, proof)
    validate_live(plan, proof, action)
    return proof


def validate_live(plan, proof, action, *, fresh=True):
    from mqar_final_tail_host_prepare import validate_receipt
    from seed202_pilot.preparation_auth import verify

    require(
        type(proof) is dict
        and set(proof)
        == {
            "schema",
            "plan_sha256",
            "action",
            "source",
            "started",
            "finished",
            "preparation",
            "predecessors",
            "service_evidence",
            "private_evidence",
            "source_evidence",
            "namespace",
            "quota",
            "lifetime",
            "active",
            "local",
            "lock",
            "authentication",
        },
        "TAIL_LIVE_SCHEMA",
    )
    verify(plan, proof)
    require(
        proof["schema"] == "TAIL_LIVE_TWO_STAGE_V1"
        and proof["source"] == "LIVE_PROVIDER_API"
        and proof["plan_sha256"] == value_sha(plan)
        and proof["action"] == action
        and action in ACTIONS,
        "TAIL_LIVE_SCOPE",
    )
    for key in ("started", "finished"):
        require(type(proof[key]) in (int, float) and math.isfinite(proof[key]), "TAIL_LIVE_TIME")
    require(
        0 <= proof["finished"] - proof["started"] < 120
        and (not fresh or 0 <= time.time() - proof["started"] < 120),
        "TAIL_LIVE_EXPIRED",
    )
    wrapper = proof["preparation"]
    require(
        set(wrapper) == {"receipt", "sha256", "revalidation"}
        and wrapper["sha256"] == value_sha(wrapper["receipt"]),
        "TAIL_PREPARATION_WRAPPER",
    )
    receipt = validate_receipt(plan, wrapper["receipt"], action)
    revalidation = wrapper["revalidation"]
    require(
        set(revalidation) == {"started", "finished", "write_denial"}
        and revalidation["write_denial"] == "WIN32_FILE_SHARE_READ"
        and receipt["finished"]
        <= revalidation["started"]
        <= revalidation["finished"]
        <= proof["started"],
        "TAIL_PREPARATION_DOMAIN",
    )
    require(
        exact(proof["source_evidence"], receipt["source_evidence"])
        and exact(proof["predecessors"], plan["predecessors"])
        and exact(receipt["predecessors"], plan["predecessors"]),
        "TAIL_SOURCE_PREDECESSORS",
    )
    require(
        exact(
            proof["private_evidence"],
            {"private_env_equal_memory_only": True, **receipt["private_identity"]},
        ),
        "TAIL_PRIVATE_EQUALITY",
    )
    require(
        exact(
            proof["service_evidence"],
            {"status": "ABSENT_FRESH_TAIL_SERVICES", "nonce": plan["nonce"]},
        ),
        "TAIL_SERVICE_ABSENCE",
    )
    ns = proof["namespace"]
    require(
        set(ns)
        == {
            "datasets",
            "private_datasets",
            "kernels",
            "observations",
            "owner",
            "complete_owner_namespace",
            "controls",
        }
        and ns["owner"] == "zimike"
        and ns["complete_owner_namespace"] is True,
        "TAIL_NAMESPACE_SCHEMA",
    )
    ds, ks = ns["datasets"], ns["kernels"]
    require(
        type(ds) is dict
        and all(type(v) is bool for v in ds.values())
        and ns["private_datasets"] == sorted(k for k, v in ds.items() if v)
        and type(ks) is list
        and len(ks) == len(set(ks)),
        "TAIL_COMPLETE_PRIVATE_NAMESPACE",
    )
    require(
        SCIENCE in ds
        and ds[SCIENCE] is True
        and DATA in ds
        and ds[DATA] is True
        and PRIVATE in ds
        and ds[PRIVATE] is True
        and CONTROL in ks
        and exact(
            ns["controls"],
            {"dataset": SCIENCE, "kernel": CONTROL, "exact_metadata_status": 200, "private": True},
        ),
        "TAIL_CONTROLS",
    )
    ref, target = plan["dataset_target"][:-2], plan["kernel_target"]
    require((ref in ds) == (action == "submit") and target not in ks, "TAIL_NAMESPACE_ACTION")
    wanted = {target: "kernel"} | ({ref: "dataset"} if action == "publish" else {})
    require(set(ns["observations"]) == set(wanted), "TAIL_ABSENCE_SET")
    for name, kind in wanted.items():
        row = ns["observations"][name]
        require(
            set(row) == {"http_status", "kind", "classification"}
            and type(row["http_status"]) is int
            and row["http_status"] in {403, 404}
            and row["kind"] == kind
            and row["classification"]
            == ("FORBIDDEN_AMBIGUOUS" if row["http_status"] == 403 else "HTTP_NOT_FOUND"),
            "TAIL_ABSENCE_NOT_FAKE_404",
        )
    q = proof["quota"]
    require(
        set(q) == {"total", "used", "reserved", "remaining", "planning", "required"}
        and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in q.values()),
        "TAIL_QUOTA_TYPES",
    )
    require(
        q["reserved"] == 0
        and q["remaining"] == q["total"] - q["used"]
        and q["planning"] == q["remaining"] * 0.8
        and q["planning"] >= q["required"] == 9000,
        "TAIL_QUOTA",
    )
    lifetime = proof["lifetime"]
    require(
        set(lifetime) == {"source", "observed_unix_seconds", "documentation_text"}
        and lifetime["source"] == "https://www.kaggle.com/docs/notebooks"
        and type(lifetime["observed_unix_seconds"]) in (int, float)
        and 0 <= proof["started"] - lifetime["observed_unix_seconds"] <= 43200
        and re.search(
            r"12 hours execution time for CPU and GPU notebook sessions",
            lifetime["documentation_text"],
        )
        and "Save & Run All creates a new session with a completely clean state"
        in lifetime["documentation_text"],
        "TAIL_LIFETIME",
    )
    local = proof["local"]
    require(
        set(local)
        == {"writer_lock_held", "active_processes", "disk_bytes", "disk_required_bytes", "pid"}
        and local["writer_lock_held"] is True
        and local["active_processes"] == []
        and type(local["disk_bytes"]) is int
        and type(local["disk_required_bytes"]) is int
        and local["disk_bytes"] >= local["disk_required_bytes"] == 10_000_000_000
        and type(local["pid"]) is int
        and local["pid"] > 0
        and proof["active"] == [],
        "TAIL_PROCESS_DISK",
    )
    require(
        set(proof["lock"]) == {"path", "device", "inode", "pid"}
        and proof["lock"]["pid"] == local["pid"]
        and proof["lock"]["path"]
        == str((Path(plan["journals"][action]) / "single-writer.lock").resolve()),
        "TAIL_PROOF_LOCK",
    )
    return proof
