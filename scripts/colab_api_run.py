"""Explicit no-Drive input delivery and detached Colab jobs. Default is a zero-I/O plan."""

# Exact JSON types reject bool-as-int; compound boundary contracts are intentionally explicit.
# pylint: disable=unidiomatic-typecheck,too-many-boolean-expressions

import argparse
import hashlib
import json
import os
import re
import uuid
import zipfile
from pathlib import Path

import colab_api_auth as auth
import colab_api_client as google
import colab_api_run_remote as remote
from colab_api_mcp import MCP
from colab_api_transfer import Contents

SOURCE = "1beba22ca8f9897e0b7367e07f4a016559946f1183fc8413deff2490c6e85ef1"
DATA = "1c660aa5f54d88ea9e0c1ce4eba0c25084eee164c55ab6c1d80d5db14d3e59fb"


class Journal:
    """No-clobber append-only barrier. Any existing journal forbids another attempt."""

    def __init__(self, path, binding):
        self.path, self.binding, self.stream = Path(path), binding, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise ValueError("JOURNAL_EXISTS") from None
        self.stream = os.fdopen(descriptor, "w", encoding="utf-8")
        self.record(dict(self.binding, phase="prepared"))
        return self

    def record(self, value):
        """Only controller-created secret-free fields reach this sink."""
        self.stream.write(json.dumps(value, sort_keys=True) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def __exit__(self, *_args):
        self.stream.close()


def validate_identity(runtime, run_name):
    """Literal runtime and RFC1034 run name; never normalize caller identifiers."""
    if not re.fullmatch(r"runtimes/[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?", runtime):
        raise ValueError("INVALID_RUNTIME")
    if not re.fullmatch(r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?", run_name):
        raise ValueError("INVALID_RUN_NAME")


def export_identity(directory):
    """Pin actual export bytes and scientific manifest before any account call."""
    directory = Path(directory)
    archive = directory / "colab-package.zip"
    report = json.loads((directory / "export-report.json").read_text(encoding="utf-8"))
    with archive.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if (
        report.get("archive_sha256") != sha
        or report.get("archive_bytes") != archive.stat().st_size
        or report.get("source_sha256") != SOURCE
        or report.get("training_started") is not False
    ):
        raise ValueError("EXPORT_REPORT_MISMATCH")
    with zipfile.ZipFile(archive) as package:
        manifest = json.loads(package.read("manifest.json"))
    if (
        manifest.get("source_sha256") != SOURCE
        or manifest.get("data_fingerprint") != DATA
        or manifest.get("storage") != "hf"
        or manifest.get("schema_version") != 3
        or manifest.get("periodic_full_checkpoints") is not True
        or manifest.get("tracking") != "clearml"
        or manifest.get("hf_upload") is not True
        or manifest.get("default_family") != "deltanet"
        or manifest.get("expected_updates", {}).get("deltanet") != 47265
    ):
        raise ValueError("UNAPPROVED_MANIFEST")
    # Reuse the exporter's complete exact scientific contract, including all fields.
    from colab_package import CONFIGS  # pylint: disable=import-outside-toplevel
    from export_colab import scientific_configs  # pylint: disable=import-outside-toplevel

    configs = manifest["configs"]
    spec = manifest["dataset_spec"]
    kv = {key: value for key, value in spec.items() if key not in ("vocab_size", "seed")}
    kv["dataset_seed"] = spec["seed"]
    identity = {
        "data": {"spec": spec},
        "configs": {CONFIGS[family]: dict(config, kv=kv) for family, config in configs.items()},
    }
    if scientific_configs({"identity": identity}) != configs:
        raise ValueError("UNAPPROVED_CONFIG")
    notebook = json.loads((directory / "launch_colab.ipynb").read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    if sha not in code or "drive.mount" in code or "google.colab import userdata" in code:
        raise ValueError("NOTEBOOK_IDENTITY_MISMATCH")
    return archive, sha


def private_bytes(path):
    """Validate exactly seven JSON-quoted dotenv fields; never return their metadata."""
    from colab_services import KEYS  # pylint: disable=import-outside-toplevel

    original = Path(path)
    path = original.resolve()
    repo = Path(__file__).resolve().parents[1]
    if path.is_relative_to(repo) or original.is_symlink():
        raise ValueError("PRIVATE_FILE_OUTSIDE_REPO_REQUIRED")
    raw = path.read_bytes()
    if not raw or len(raw) > 65536:
        raise ValueError("INVALID_PRIVATE_INPUT")
    try:
        values = {}
        for line in raw.decode("utf-8").splitlines():
            key, separator, value = line.partition("=")
            if not separator or key not in KEYS or key in values:
                raise ValueError
            values[key] = json.loads(value)
        if set(values) != set(KEYS) or any(
            not isinstance(v, str) or not v.strip() or any(ord(c) < 32 or ord(c) == 127 for c in v)
            for v in values.values()
        ):
            raise ValueError
    except (ValueError, UnicodeError, TypeError):
        raise ValueError("INVALID_PRIVATE_INPUT") from None
    return raw


class Execution:  # pylint: disable=too-few-public-methods
    """One owned MCP session, fresh execution IDs only; never replay final executions."""

    def __init__(self, client, runtime, journal, session=None):
        self.client, self.journal = client, journal
        if session is not None:
            expected = runtime + "/sessions/"
            if not session.startswith(expected) or not re.fullmatch(
                r"job-[a-f0-9]{32}", session.removeprefix(expected)
            ):
                raise ValueError("SESSION_UNVERIFIED")
            rows = client.call("list_sessions", {"parent": runtime}).get("sessions", [])
            if not isinstance(rows, list) or not any(
                isinstance(row, dict) and row.get("name") == session for row in rows
            ):
                raise ValueError("SESSION_UNVERIFIED")
            self.session = session
            journal.record({"phase": "session_reuse_verified", "session": session})
            return
        sid = "job-" + uuid.uuid4().hex
        self.session = runtime + "/sessions/" + sid
        journal.record({"phase": "create_session_pending", "session": self.session})
        created = client.call("create_session", {"parent": runtime, "sessionId": sid})
        if created.get("name") != self.session:
            raise ValueError("SESSION_UNVERIFIED")

    def code(self, code):
        """All programs are fixed controller-generated code without private input values."""
        identifier = str(uuid.uuid4())
        self.journal.record({"phase": "execute_pending", "execution_id": identifier})
        result = self.client.call(
            "execute_code", {"session": self.session, "executionId": identifier, "code": code}
        )
        if (
            result.get("session") != self.session
            or result.get("executionId") != identifier
            or result.get("executionError") is not None
            or result.get("outputTruncated", False) is not False
        ):
            raise ValueError("EXECUTION_UNKNOWN_OR_FAILED")
        stdout = result.get("stdout", "")
        if not isinstance(stdout, str) or len(stdout) > 32768:
            raise ValueError("EXECUTION_INVALID_RESULT")
        lines = [
            line[len(remote.MARKER) :]
            for line in stdout.splitlines()
            if line.startswith(remote.MARKER)
        ]
        if len(lines) != 1:
            raise ValueError("EXECUTION_UNKNOWN_OR_FAILED")
        try:
            value = json.loads(lines[0])
            if not isinstance(value, dict):
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError("EXECUTION_INVALID_RESULT") from None
        return value


def transfer(args, binding, archive, execution, token, journal):
    """Preflight, root proof, capability proof, bounded upload, readback, SHA proof."""
    nonce = uuid.uuid4().hex
    stage = "compare-archive-" + nonce + ".zip"
    journal.record({"phase": "transfer_prepared", "nonce": nonce, "staging_archive": stage})
    if execution.code(remote.marker_code(binding, nonce)) != {"guard_passed": True}:
        raise ValueError("PREFLIGHT_UNVERIFIED")
    # Provider response is transient: never feed connectionInfo into journals or prints.
    response = google.get(google.BASE + args.runtime, token)
    if response.get("name") != args.runtime:
        raise ValueError("RUNTIME_MISMATCH")
    contents = Contents(response.get("connectionInfo", {}))
    del response
    prefix = contents.discover_root("compare-transport-" + nonce, nonce)
    journal.record({"phase": "root_verified", "mapping": prefix})
    contents.probe_chunks(prefix + "compare-chunk-" + nonce, journal.record)
    contents.upload_archive(prefix + stage, archive, journal.record)
    # Secret is read only after the target runtime hardware/root/capability are verified.
    contents.write_private(
        prefix + "private/input.env", private_bytes(args.env_file), journal.record
    )
    result = execution.code(remote.finalize_code(binding, stage))
    if result != {"inputs_verified": True, "sha256": binding["sha256"]}:
        raise ValueError("INPUTS_UNVERIFIED")
    journal.record({"phase": "inputs_verified", **binding})


def load_binding(path, runtime, run_name):
    """Read only identity fields from the original setup journal."""
    with Path(path).open(encoding="utf-8") as stream:
        first = json.loads(stream.readline())
    binding = {key: first.get(key) for key in ("runtime", "run_name", "sha256")}
    if (
        binding["runtime"] != runtime
        or binding["run_name"] != run_name
        or not isinstance(binding["sha256"], str)
        or not re.fullmatch("[a-f0-9]{64}", binding["sha256"])
    ):
        raise ValueError("JOURNAL_IDENTITY_MISMATCH")
    if "recovery_sha256" in first:
        from colab_api_recovery import ARCHIVE_SHA, RecoverySpec  # pylint: disable=import-outside-toplevel

        if (
            first["recovery_sha256"] != RecoverySpec(runtime, run_name).sha256()
            or binding["sha256"] != ARCHIVE_SHA
        ):
            raise ValueError("RECOVERY_BINDING_MISMATCH")
        binding["recovery_sha256"] = first["recovery_sha256"]
    return binding


def run(args):  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """No automatic creation, retries, resume, deletion or GPU allocation."""
    validate_identity(args.runtime, args.run_name)
    if getattr(args, "recovery", False) and args.command != "setup":
        raise ValueError("RECOVERY_FLAG_SETUP_ONLY")
    spec = None
    if args.command in ("restore", "resume") or getattr(args, "recovery", False):
        from colab_api_recovery import RecoverySpec  # pylint: disable=import-outside-toplevel

        spec = RecoverySpec(args.runtime, args.run_name)
    approval = tuple(
        getattr(args, name, None) for name in ("recovery_helper_sha256", "restore_proof_sha256")
    )
    if any(value is not None for value in approval) and args.command != "resume":
        raise ValueError("RESTORE_PROOF_RESUME_ONLY")
    if args.command == "resume" and (args.execute or any(value is not None for value in approval)):
        if not all(
            isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value) for value in approval
        ):
            raise ValueError("RESTORE_PROOF_SHA_REQUIRED")
    if not args.execute:
        plan = {
            "phase": "plan",
            "command": args.command,
            "runtime": args.runtime,
            "run_name": args.run_name,
            "storage": "hf",
            "training_started": False,
        }
        if spec is not None:
            plan["recovery_sha256"] = spec.sha256()
        return plan
    if args.journal is None:
        raise ValueError("JOURNAL_REQUIRED")
    archive = None
    if args.command == "setup":
        if args.export_dir is None or args.env_file is None:
            raise ValueError("EXPORT_AND_PRIVATE_FILE_REQUIRED")
        archive, sha = export_identity(args.export_dir)
        binding = {"runtime": args.runtime, "run_name": args.run_name, "sha256": sha}
        if spec is not None:
            if sha != spec.value()["archive_sha256"]:
                raise ValueError("RECOVERY_BINDING_MISMATCH")
            binding["recovery_sha256"] = spec.sha256()
    else:
        if args.setup_journal is None:
            raise ValueError("SETUP_JOURNAL_REQUIRED")
        binding = load_binding(args.setup_journal, args.runtime, args.run_name)
    if args.command == "train" and "recovery_sha256" in binding:
        raise ValueError("RECOVERY_REQUIRES_EXPLICIT_RESUME")
    if args.command in ("restore", "resume") and binding.get("recovery_sha256") != spec.sha256():
        raise ValueError("RECOVERY_BINDING_MISMATCH")
    with Journal(args.journal, binding) as journal:
        token = auth.access_token(args.token_file, args.client_file, required_scope=auth.MANAGE)
        client = MCP(token)
        journal.record({"phase": "initialize_pending"})
        client.initialize()
        session = None
        if args.command != "setup":
            with args.setup_journal.open(encoding="utf-8") as source:
                sessions = [
                    json.loads(line).get("session")
                    for line in source
                    if '"create_session_pending"' in line
                ]
            if len(sessions) != 1 or not isinstance(sessions[0], str):
                raise ValueError("SESSION_UNVERIFIED")
            session = sessions[0]
        execution = Execution(client, args.runtime, journal, session=session)
        if args.command in ("status", "logsummary"):
            result = execution.code(remote.status_code(binding))
            # Do not accept arbitrary remote strings even from the controlled program.
            result = clean_status(result, binding)
        else:
            bootstrap = ""
            if args.command == "setup":
                from export_colab import bootstrap_code  # pylint: disable=import-outside-toplevel

                bootstrap = bootstrap_code(
                    binding["sha256"],
                    archive_path=remote.ARCHIVE,
                    work_dir=remote.PACKAGE,
                    venv=remote.VENV,
                    result_dir=remote.RESULTS + args.run_name,
                    env_file=remote.PRIVATE,
                )
                transfer(args, binding, archive, execution, token, journal)
            options = (
                {"helper_sha": args.recovery_helper_sha256, "proof_sha": args.restore_proof_sha256}
                if args.command == "resume"
                else {}
            )
            result = execution.code(remote.launch_code(binding, args.command, bootstrap, **options))
            if (
                set(result) != {"phase", "state", "pid"}
                or result["phase"] != args.command
                or result["state"] != "spawned"
                or type(result["pid"]) is not int
            ):
                raise ValueError("LAUNCH_UNVERIFIED")
            # An acknowledged Popen is NOT yet proof that setup/training succeeded.
            result = clean_status(execution.code(remote.status_code(binding)), binding)
        journal.record({"phase": "readback_verified"})
        return result


def clean_status(value, binding):
    """Strict output schema prevents raw logs/URLs/credentials leaking via status."""
    recovery = "recovery_sha256" in binding
    if recovery:
        from colab_api_recovery import validate_status  # pylint: disable=import-outside-toplevel

        validate_status(value.get("recovery"), binding)
    if (
        set(value)
        != {"sha256", "stages", "metrics", "hf_checkpoint"} | ({"recovery"} if recovery else set())
        or value["sha256"] != binding["sha256"]
        or not isinstance(value["stages"], dict)
        or set(value["stages"])
        != ({"setup", "restore", "train"} if recovery else {"setup", "train"})
        or not isinstance(value["metrics"], dict)
    ):
        raise ValueError("STATUS_INVALID")
    for stage in value["stages"].values():
        if (
            not isinstance(stage, dict)
            or set(stage) - {"state", "returncode", "process_present"}
            or stage.get("state") not in ("not_started", "spawn_unknown", "running", "completed")
            or ("returncode" in stage and type(stage["returncode"]) is not int)
            or ("process_present" in stage and type(stage["process_present"]) is not bool)
        ):
            raise ValueError("STATUS_INVALID")
    import math  # pylint: disable=import-outside-toplevel

    if set(value["metrics"]) - {
        "step",
        "steps",
        "update",
        "train_loss",
        "val_loss",
        "loss",
        "train_flops",
        "flops",
    }:
        raise ValueError("STATUS_INVALID")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value["metrics"].values()):
        raise ValueError("STATUS_INVALID")
    proof = value["hf_checkpoint"]
    if proof is not None:
        if (
            not isinstance(proof, dict)
            or set(proof) - {"verified", "revision", "step"}
            or proof.get("verified") is not True
            or not isinstance(proof.get("revision"), str)
            or not re.fullmatch("[a-f0-9]{40}", proof["revision"])
            or ("step" in proof and (type(proof["step"]) is not int or proof["step"] < 0))
        ):
            raise ValueError("STATUS_INVALID")
    return value


def main(argv=None):
    """All mutations require --execute; status is also explicit remote execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("setup", "train", "restore", "resume", "status", "logsummary")
    )
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--recovery-helper-sha256")
    parser.add_argument("--restore-proof-sha256")
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--setup-journal", type=Path)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--client-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args), allow_nan=False))
        return 0
    except ValueError as error:
        allowed = {
            "CHUNK_UPLOAD_UNSUPPORTED",
            "CONTENTS_ROOT_UNVERIFIED",
            "JOURNAL_EXISTS",
            "CONTENTS_STATE_UNKNOWN",
            "PRIVATE_READBACK_FAILED",
            "INVALID_PROXY",
            "TARGET_EXISTS",
            "EXPORT_REPORT_MISMATCH",
            "UNAPPROVED_MANIFEST",
            "UNAPPROVED_CONFIG",
            "NOTEBOOK_IDENTITY_MISMATCH",
            "EXECUTION_UNKNOWN_OR_FAILED",
            "LAUNCH_UNVERIFIED",
            "STATUS_INVALID",
        }
        code = str(error) if str(error) in allowed else "CONTROLLER_STOPPED_RECONCILE_JOURNAL"
        print(json.dumps({"error": code}))
        return 2
    except Exception:  # pylint: disable=broad-exception-caught
        # No raw exceptions: provider messages may contain signed URLs or auth headers.
        print(json.dumps({"error": "CONTROLLER_STOPPED_RECONCILE_JOURNAL"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
