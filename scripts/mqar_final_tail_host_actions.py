"""Tail journal barriers and audited transaction reuse; no generic rollout controller."""

import contextlib
import json
import os
from pathlib import Path

from mqar_final_tail_host_contract import ACTIONS, exact, value_sha
from mqar_final_tail_native import digest, require
from mqar_final_tail_runner import create_json


def load(path):
    return json.loads(Path(path).read_bytes())


def effect(plan, action):
    require(action in ACTIONS, "TAIL_ACTION")
    if action == "publish":
        return {
            "target": plan["dataset_target"],
            "head": 1,
            "private": True,
            "ready": True,
            "file_count": len(plan["dataset_files"]),
            "all_bytes_equal": True,
            "files": plan["dataset_files"],
        }
    return {
        "target": plan["kernel_target"],
        "version": 1,
        "head": 1,
        "private": True,
        "source_sha256": plan["kernel_sha256"],
        "requested_versioned_inputs": plan["requested_versioned_inputs"],
    }


def authority(approval):
    plan, action = approval["plan"], approval["proof"]["action"]
    return {
        "target": effect(plan, action)["target"],
        "version": 1,
        "deployment_nonce": plan["nonce"],
        "scientific_nonce": "seed202-inference-only-" + plan["nonce"],
        "archive_sha256": plan["dataset_files"]["recovery-operator.zip.bin"]["sha256"],
        "kernel_sha256": plan["kernel_sha256"],
        "source_authority": approval,
        "action": action,
        "effect": effect(plan, action),
    }


def recorded(plan, action):
    from mqar_final_tail_host_seal import validate

    journal = Path(plan["journals"][action])
    auth = load(journal / "approval.json")
    validate(auth["source_authority"], revalidate=False)
    require(
        exact(auth["source_authority"]["plan"], plan)
        and auth["action"] == action
        and exact(auth, authority(auth["source_authority"])),
        "TAIL_RECORDED_AUTHORITY",
    )
    intent = load(journal / "intent.json")
    require(
        exact(
            intent,
            {
                "target": auth["target"],
                "version": 1,
                "deployment_nonce": plan["nonce"],
                "scientific_nonce": auth["scientific_nonce"],
                "archive_sha256": auth["archive_sha256"],
                "kernel_sha256": plan["kernel_sha256"],
                "approval_sha256": value_sha(auth),
                "approved_journal": str(journal.resolve()),
            },
        ),
        "TAIL_RECORDED_INTENT",
    )
    return auth


def acknowledged(plan, action):
    journal = Path(plan["journals"][action])
    if not (journal / "intent.json").exists():
        require(
            not any(
                (journal / n).exists()
                for n in (
                    "approval.json",
                    "unknown.json",
                    "acknowledged.json",
                    "reconciled-ack.json",
                    "fresh-under-lock.json",
                )
            ),
            "TAIL_ORPHAN_JOURNAL",
        )
        return False
    recorded(plan, action)
    if (journal / "acknowledged.json").exists():
        require(exact(load(journal / "acknowledged.json"), effect(plan, action)), "TAIL_ACK_EFFECT")
        return True
    if (journal / "reconciled-ack.json").exists():
        require(
            exact(
                load(journal / "reconciled-ack.json"),
                {
                    "state": "RECONCILED_GET_ONLY",
                    "intent_sha256": digest(journal / "intent.json"),
                    "remote_mutations": 0,
                    "readback": effect(plan, action),
                },
            ),
            "TAIL_RECONCILED_EFFECT",
        )
        return True
    return False


@contextlib.contextmanager
def command_scope(plan, action, *, reconcile=False):
    """Every pending/unknown/orphan/prior-action barrier is before authentication."""
    require(action in ACTIONS, "TAIL_ACTION")
    root = Path(plan["journal_directory"])
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "host-single-writer.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        for selected in ACTIONS:
            journal = Path(plan["journals"][selected])
            require(not (journal / "single-writer.lock").exists(), "TAIL_WRITER_LOCK_PRESENT")
            done = acknowledged(plan, selected)
            pending = (journal / "intent.json").exists()
            if reconcile and selected == action:
                require(pending and not done, "TAIL_EXACT_UNKNOWN_REQUIRED")
            else:
                require(not pending or done, "TAIL_PENDING_GET_ONLY")
                require(selected != action or not pending, "TAIL_ACTION_ATTEMPTED_GET_ONLY")
                if ACTIONS.index(selected) < ACTIONS.index(action):
                    require(done, "TAIL_PRIOR_ACK_REQUIRED")
        yield
    finally:
        lock.unlink()


def perform(approval, wrapper, provider, private, lifetime, services):
    from mqar_final_tail_host_live import probe
    from mqar_final_tail_host_sdk import TailCapability, mutate, readback
    from mqar_final_tail_host_seal import validate
    from seed202_pilot.audited_publish import transaction

    validate(approval, revalidate=False)
    plan, action = approval["plan"], approval["proof"]["action"]
    journal = Path(plan["journals"][action])
    auth = authority(approval)
    parent = wrapper["receipt"]["parent_archive"]

    def mutation():
        create_json(journal / "approval.json", auth)
        proof = probe(
            plan,
            action,
            wrapper,
            provider,
            private,
            lifetime,
            services,
            journal / "single-writer.lock",
        )
        create_json(journal / "fresh-under-lock.json", proof)
        return mutate(plan, action, provider, TailCapability(auth, proof, journal))

    authorize = transaction(
        parent,
        journal,
        approved_journal=journal,
        validator=lambda v: exact(v, auth),
        effect=lambda v: exact(v, effect(plan, action)),
    )
    return authorize(auth, journal, mutation, lambda _ack: readback(plan, action, provider))


def reconcile(plan, action, provider, parent):
    from mqar_final_tail_host_sdk import readback
    from seed202_pilot.audited_publish import reconcile as inherited_reconcile

    auth = recorded(plan, action)
    return inherited_reconcile(
        parent,
        auth,
        Path(plan["journals"][action]),
        readback=lambda: readback(plan, action, provider),
        effect=lambda v: exact(v, effect(plan, action)),
    )
