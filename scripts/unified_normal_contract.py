"""One request-owned predecessor policy and typed evidence boundary, HOST only."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Policy(StrEnum):
    ORDINARY = "AUTHENTIC_HOST_NATIVE_V2_FULL_REPLAY"
    SORTING_CLASSIC = "QUALIFIED_SORTING_CLASSIC_TAIL_V1"
    SORTING_DELTA = "QUALIFIED_SORTING_DELTA_TAIL_V1"
    MQAR_CONTINUATION = "QUALIFIED_MQAR_CONTINUATION_TAIL_V1"
    COMPLETED_NORMAL = "QUALIFIED_MQAR_DELTA303_COMPLETED_NATIVE_V1"
    COMPLETED_SORTING404 = "QUALIFIED_SORTING_CLASSIC404_COMPLETED_NATIVE_V1"


# This is an authority-approved historical case list, not task dispatch logic.
QUALIFIED = {
    Policy.SORTING_CLASSIC: (
        "sorting-classic-seed202",
        "d0376599d2934db6ea87b5e455a9ca90f524cbf10d98ec5129bf56f931c14494",
    ),
    Policy.SORTING_DELTA: (
        "sorting-deltanet-seed202",
        "d5fbc3a20b67b3be0400f786fbd461993ad07fe640bdca6a718364eeaa160a77",
    ),
    Policy.MQAR_CONTINUATION: (
        "mqar-k64-classic-seed202",
        "9a75cfe268660608e835582acf74f1a4438ccc1281e3143b07c6c84a3cd347ed",
    ),
}
SCHEMA = "UNIFIED_VERIFIED_PREDECESSOR_V1"
_REPLAY_CACHE = {}  # Pure result cache; the complete locked byte identity is its key.


def require(condition, code):
    if not condition:
        raise ValueError(code)


def valid_evidence_row(row):
    """Finite wire schema only; request equality and whole replay remain mandatory."""
    if type(row) is not dict or set(row) != {"case_id", "acceptance_sha256", "replay_scope"}:
        return False
    try:
        policy = Policy(row["replay_scope"])
    except (TypeError, ValueError):
        return False
    if policy in QUALIFIED:
        return (row["case_id"], row["acceptance_sha256"]) == QUALIFIED[policy]
    return (
        type(row["case_id"]) is str
        and (policy != Policy.COMPLETED_NORMAL or row["case_id"] == "mqar-k64-deltanet-seed303")
        and (policy != Policy.COMPLETED_SORTING404 or row["case_id"] == "sorting-classic-seed404")
        and row["case_id"] not in {case for case, _sha in QUALIFIED.values()}
        and type(row["acceptance_sha256"]) is str
        and len(row["acceptance_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in row["acceptance_sha256"])
    )


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    from unified_normal_live import held_digest

    authenticated = held_digest(path)
    if authenticated is not None:
        return authenticated
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


@dataclass(frozen=True)
class VerifiedPredecessor:
    """A value, not authority: every consumer replays the request-owned verifier."""

    case_id: str
    policy: Policy
    acceptance_sha256: str
    whole_object_sha256: str
    request_sha256: str

    def evidence(self):
        return {
            "case_id": self.case_id,
            "acceptance_sha256": self.acceptance_sha256,
            "replay_scope": self.policy.value,
        }

    def serialized(self):
        return {
            "schema": SCHEMA,
            **self.evidence(),
            "whole_object_sha256": self.whole_object_sha256,
            "request_sha256": self.request_sha256,
        }


def descriptor(candidate):
    path = Path(candidate) / "payload/unified/request.json"
    value = read(path)
    require(
        type(value) is dict
        and set(value) == {"schema", "run", "predecessors", "normal", "science", "roots", "files"}
        and value["schema"] == "UNIFIED_NORMAL_REQUEST_V1",
        "UNIFIED_REQUEST_SCHEMA",
    )
    require(type(value["predecessors"]) is list, "UNIFIED_PREDECESSOR_SEQUENCE")
    ids = []
    for row in value["predecessors"]:
        require(
            type(row) is dict and set(row) == {"case_id", "policy", "path", "sha256", "replay"},
            "UNIFIED_PREDECESSOR_SCHEMA",
        )
        policy = Policy(row["policy"])
        if policy == Policy.COMPLETED_NORMAL:
            require(
                value["normal"]["profile"]
                in {
                    "REMAINING_FOUR_NONBLOCKING_TRACKING_V1",
                    "REMAINING_THREE_FINISH_IMPORT_FIXED_V1",
                }
                and row["case_id"] == "mqar-k64-deltanet-seed303",
                "UNIFIED_COMPLETED_CASE3_ONLY",
            )
        if policy == Policy.COMPLETED_SORTING404:
            require(
                value["normal"]["profile"] == "REMAINING_THREE_FINISH_IMPORT_FIXED_V1"
                and row["case_id"] == "sorting-classic-seed404",
                "UNIFIED_COMPLETED_CASE4_ONLY",
            )
        if policy in QUALIFIED:
            require(
                (row["case_id"], row["sha256"]) == QUALIFIED[policy],
                "UNIFIED_QUALIFICATION_AUTHORITY",
            )
        else:
            require(
                row["case_id"] not in {case for case, _sha in QUALIFIED.values()},
                "UNIFIED_POLICY_DOWNGRADE",
            )
        ids.append(row["case_id"])
        require(
            str(Path(row["path"]).resolve()) in value["files"]
            and value["files"][str(Path(row["path"]).resolve())]["sha256"] == row["sha256"],
            "UNIFIED_PREDECESSOR_INPUT_PIN",
        )
    require(len(ids) == len(set(ids)), "UNIFIED_DUPLICATE_PREDECESSOR")
    from unified_normal_profiles import is_remaining_profile

    if is_remaining_profile(value["normal"]["profile"]):
        from normal_remaining_scope import require_prefix
        from unified_normal_remaining import require_plan_pin

        require_prefix(value["run"]["id"], ids)
        require_plan_pin(
            value["run"]["id"], value["run"]["plan_sha256"], value["normal"]["profile"]
        )
        from unified_normal_remaining import prefix_policies

        prefix_policies(value["predecessors"], value["normal"]["profile"])
    return path, value


def evidence_paths(candidate):
    path, value = descriptor(candidate)
    return [str(path), *value["files"]]


_HELD_INPUTS = {}  # Only populated while real Win32 read-denial handles are held.


@contextlib.contextmanager
def held_inputs(candidate, locked):
    """Reuse a caller's existing audited lease; never synthesize a boolean proof."""
    from seed202_pilot.prepared_files import check

    key = str(Path(candidate).resolve())
    require(key not in _HELD_INPUTS, "UNIFIED_DUPLICATE_INPUT_SCOPE")
    required = {str(Path(p).resolve()) for p in evidence_paths(candidate)}
    require(required <= set(locked), "UNIFIED_LEASE_CLOSURE")
    selected = {p: locked[p] for p in sorted(required)}
    check(selected)
    _path, value = descriptor(candidate)
    verify_files(value, selected)
    _HELD_INPUTS[key] = selected
    try:
        yield
    finally:
        _HELD_INPUTS.pop(key)
        check(selected)


@contextlib.contextmanager
def request_lease(candidate):
    """Only fallback acquisition hashes bytes; LIVE reuses held immutable inputs."""
    from unified_normal_live import retained_lease as lease

    selected = _HELD_INPUTS.get(str(Path(candidate).resolve()))
    if selected is not None:
        # Caller retains the real handles and performs whole-scope checks at
        # entry/exit; the audited preparation capability checks them again.
        yield selected
    else:
        with lease(evidence_paths(candidate)) as locked:
            yield locked


def verify_files(value, locked=None):
    """Use unchanged safe paths; inventory closure includes every import root."""
    from seed202_pilot.prepared_files import safe_path, safe_root

    require(type(value["files"]) is dict and bool(value["files"]), "UNIFIED_INPUTS")
    for name, pin in value["files"].items():
        path = safe_path(Path(name))
        require(
            set(pin) == {"bytes", "sha256"}
            and path.stat().st_size == pin["bytes"]
            and (locked[str(path)]["sha256"] if locked is not None else digest(path))
            == pin["sha256"],
            "UNIFIED_INPUT_DRIFT",
        )
    # Match Path.is_relative_to's lexical, platform-normalized components without
    # rebuilding every ancestor sequence for every root/row. All byte, native
    # path and actual directory-closure checks above/below still run each time.
    file_parts = {
        name: tuple(os.path.normcase(part) for part in Path(name).parts) for name in value["files"]
    }
    for name in value["roots"]:
        root = safe_root(Path(name))
        actual = set()
        for path in root.rglob("*"):
            require(not path.is_symlink() and not path.is_junction(), "UNIFIED_INPUT_LINK")
            if path.is_file():
                actual.add(str(path.resolve()))
        root_parts = tuple(os.path.normcase(part) for part in root.parts)
        expected = {n for n, parts in file_parts.items() if parts[: len(root_parts)] == root_parts}
        require(actual == expected, "UNIFIED_IMPORT_ROOT_CLOSURE")


def _replay(candidate, row):
    """Legacy serializers are isolated behind this one policy-selected verifier."""
    from seed202_pilot.prepared_files import check

    request_path, inputs = descriptor(candidate)
    require(row in inputs["predecessors"], "UNIFIED_DECLARED_POLICY_REQUIRED")
    # The existing Windows write-denial lease prevents concurrent source mutation.
    with request_lease(candidate) as locked:
        verify_files(inputs, locked)
        value = read(row["path"])
        expected = VerifiedPredecessor(
            row["case_id"],
            Policy(row["policy"]),
            digest(row["path"]),
            hashlib.sha256(canonical(value).encode()).hexdigest(),
            digest(request_path),
        )
        cache_key = (row["case_id"], digest(request_path), canonical(locked))
        if cache_key in _REPLAY_CACHE:
            require(_REPLAY_CACHE[cache_key] == expected, "UNIFIED_CACHED_WITNESS_DRIFT")
            check(locked)
            return expected
        env = {
            k: v
            for k, v in os.environ.items()
            if k.upper()
            in {
                "PATH",
                "SYSTEMROOT",
                "WINDIR",
                "HOME",
                "USERPROFILE",
                "LOCALAPPDATA",
                "APPDATA",
                "USERNAME",
                "USER",
                "LOGNAME",
                "TMPDIR",
                "TEMP",
                "TMP",
                "UV_CACHE_DIR",
            }
        }
        env.update(
            CUDA_VISIBLE_DEVICES="-1",
            PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_OFFLINE="1",
            CLEARML_OFFLINE_MODE="1",
            PYTHON_DOTENV_DISABLED="1",
        )
        scratch = Path(os.environ["TMPDIR"])
        env.update(TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch))
        for key in (
            "UV_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "TORCH_HOME",
            "HF_HOME",
            "CLEARML_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "XDG_CACHE_HOME",
        ):
            env[key] = str(scratch / "unified-normal-caches" / key.lower())
        driver = Path(__file__).with_name("unified_normal_replay.py")
        child = subprocess.run(
            [
                "uv",
                "run",
                "--offline",
                "--no-project",
                "--no-sync",
                "--python",
                sys.executable,
                "python",
                "-B",
                str(driver),
                "--request",
                str(request_path),
                "--case",
                row["case_id"],
            ],
            env=env,
            capture_output=True,
            timeout=300,
            check=False,
        )
        require(child.returncode == 0, "UNIFIED_WHOLE_REPLAY_FAILED")
        result = json.loads(child.stdout)
        value = read(row["path"])
        expected = VerifiedPredecessor(
            row["case_id"],
            Policy(row["policy"]),
            digest(row["path"]),
            hashlib.sha256(canonical(value).encode()).hexdigest(),
            digest(request_path),
        )
        require(canonical(result) == canonical(expected.serialized()), "UNIFIED_WITNESS_REPLAY")
        check(locked)
        _REPLAY_CACHE[cache_key] = expected
        return expected


def binding(candidate):
    request_path, value = descriptor(candidate)
    with request_lease(candidate) as locked:
        scope = (
            contextlib.nullcontext()
            if str(Path(candidate).resolve()) in _HELD_INPUTS
            else held_inputs(candidate, locked)
        )
        with scope:
            witnesses = {
                row["case_id"]: _replay(candidate, row).serialized()
                for row in value["predecessors"]
            }
    return {
        "schema": "UNIFIED_PREDECESSOR_BINDING_V1",
        "inputs_sha256": digest(request_path),
        "witnesses": witnesses,
    }


def replay_predecessor(plan, path, case_id):
    candidate = plan["candidate_root"]
    _request_path, value = descriptor(candidate)
    rows = [row for row in value["predecessors"] if row["case_id"] == case_id]
    require(len(rows) == 1, "UNIFIED_DECLARED_PREDECESSOR")
    row = rows[0]
    require(Path(path).resolve() == Path(row["path"]).resolve(), "UNIFIED_PREDECESSOR_PATH")
    witness = _replay(candidate, row)
    require(
        canonical(witness.serialized())
        == canonical(plan["qualified_launch_bridge"]["witnesses"][case_id]),
        "UNIFIED_PLAN_WITNESS",
    )
    return witness.serialized()


def expected_evidence(plan):
    _path, value = descriptor(plan["candidate_root"])
    return [
        {
            "case_id": row["case_id"],
            "acceptance_sha256": row["sha256"],
            "replay_scope": row["policy"],
        }
        for row in value["predecessors"]
    ]


def proof_rows(plan):
    _path, value = descriptor(plan["candidate_root"])
    for row in value["predecessors"]:
        replay_predecessor(plan, row["path"], row["case_id"])
    return expected_evidence(plan)


def next_qualified_case(candidate, classic, acceptances):
    """A verified sequence selects a case; it never grants a new allocation."""
    from seed202_pilot.cli import _candidate
    from seed202_pilot.contract import require_authorized_case
    from unified_normal_source import operator_profile

    _candidate(Path(candidate))
    _path, value = descriptor(candidate)
    global_plan = read(Path(candidate) / "PLAN-UNAPPROVED.json")
    deployments = [
        row for row in global_plan["deployments"] if row["case_id"] == value["run"]["id"]
    ]
    require(len(deployments) == 1, "UNIFIED_NEXT_DEPLOYMENT")
    operator_profile(candidate, deployments[0])
    rows = value["predecessors"]
    require(
        [
            str(Path(p).resolve())
            for p in ([classic] if classic is not None else []) + list(acceptances)
        ]
        == [str(Path(row["path"]).resolve()) for row in rows],
        "UNIFIED_NEXT_PREFIX",
    )
    plan = {"candidate_root": str(candidate), "qualified_launch_bridge": binding(candidate)}
    proof_rows(plan)
    return require_authorized_case(value["run"]["id"])
