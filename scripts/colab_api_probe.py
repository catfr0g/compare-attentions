"""Fixed Colab hardware/storage readiness only; no setup, mounting or training."""

import argparse
import json
import os
import re
import uuid
from pathlib import Path

import colab_api_auth as auth
from colab_api_mcp import MCP, ProbeError
from colab_runner import PREFLIGHT

ARCHIVE_SHA = "e23fe304d1276bfe9e64e187f1c39556173ebd19927330c2bce50b06599cc3cd"
MARKER = "COMPARE_ATTENTION_READINESS_V1:"
CODE = (
    PREFLIGHT
    + """import os, json, hashlib
from pathlib import Path
mounted = os.path.ismount('/content/drive')
root = Path('/content/drive/MyDrive')
mydrive = mounted and root.is_dir()
archive = root / 'compare-attention/colab-package.zip'
archive_exists = mydrive and archive.is_file()
env_exists = mydrive and (root / 'compare-attention/kaggle-env.env').is_file()
digest = None
if archive_exists:
    hasher = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    digest = hasher.hexdigest()
print("""
    + repr(MARKER)
    + """ + json.dumps({
    'python': sys.version.split()[0], 'gpu': gpu.split(',')[0],
    'guard_passed': True, 'drive_mounted': mounted, 'mydrive_present': mydrive,
    'archive_exists': archive_exists, 'archive_sha256': digest,
    'archive_matches': digest == """
    + repr(ARCHIVE_SHA)
    + """, 'env_file_exists': env_exists}))
"""
)


BOOL_FIELDS = (
    "guard_passed",
    "drive_mounted",
    "mydrive_present",
    "archive_exists",
    "archive_matches",
    "env_file_exists",
)


def parse_result(result, session, execution_id):  # pylint: disable=too-many-branches
    """Validate only the controlled marker, never persist arbitrary remote output."""
    if (
        result.get("executionError") is not None
        or result.get("outputTruncated", False) is not False
        or result.get("session") != session
        or result.get("executionId") != execution_id
    ):
        raise ProbeError("REMOTE_PREFLIGHT_FAILED")
    try:
        stdout = result.get("stdout", "")
        if not isinstance(stdout, str) or len(stdout) > 16384:
            raise ValueError
        lines = [line[len(MARKER) :] for line in stdout.splitlines() if line.startswith(MARKER)]
        if len(lines) != 1:
            raise ValueError
        data = json.loads(lines[0])
        if not isinstance(data, dict) or set(data) != set(BOOL_FIELDS) | {
            "python",
            "gpu",
            "archive_sha256",
        }:
            raise ValueError
        if any(not isinstance(data[field], bool) for field in BOOL_FIELDS):
            raise ValueError
        if not data["guard_passed"] or data["gpu"] not in ("Tesla T4", "T4"):
            raise ValueError
        if not isinstance(data["python"], str) or not re.fullmatch(
            r"3\.13\.\d{1,3}", data["python"]
        ):
            raise ValueError
        digest = data["archive_sha256"]
        if digest is not None and (
            not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
        ):
            raise ValueError
        if data["archive_matches"] != (digest == ARCHIVE_SHA):
            raise ValueError
        if data["archive_exists"] != (digest is not None):
            raise ValueError
        if not data["drive_mounted"] and any(
            data[key] for key in ("mydrive_present", "archive_exists", "env_file_exists")
        ):
            raise ValueError
        if not data["mydrive_present"] and (data["archive_exists"] or data["env_file_exists"]):
            raise ValueError
    except (ValueError, TypeError, KeyError, RecursionError):
        raise ProbeError("REMOTE_INVALID_RESULT") from None
    return {
        "phase": "complete" if all(data[field] for field in BOOL_FIELDS) else "blocked",
        "readiness": data,
    }


def record(stream, **entry):
    stream.write(json.dumps(entry, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def journal_path(path):
    if path is None:
        raise ProbeError("JOURNAL_REQUIRED")
    path = Path(path).resolve()
    repo = Path(__file__).resolve().parents[1]
    allowed = repo / "local/reports/colab-api/probe"
    if path.is_relative_to(repo) and not path.is_relative_to(allowed):
        raise ProbeError("PRIVATE_JOURNAL_REQUIRED")
    if path.suffix != ".jsonl":
        raise ProbeError("PRIVATE_JOURNAL_REQUIRED")
    return path


def cleanup_session(client, stream, runtime, session):
    """Delete only our exact confirmed session and verify absence read-only."""
    record(stream, phase="delete_session_pending")
    client.call("delete_session", {"name": session})
    record(stream, phase="list_sessions_pending")
    sessions = client.call("list_sessions", {"parent": runtime}).get("sessions", [])
    if (
        not isinstance(sessions, list)
        or any(
            not isinstance(item, dict) or not isinstance(item.get("name"), str) for item in sessions
        )
        or any(item["name"] == session for item in sessions)
    ):
        raise ProbeError("SESSION_CLEANUP_REQUIRED")
    record(stream, phase="session_cleanup_verified")


def run_probe(args):  # pylint: disable=too-many-locals
    """One attempt, durable pre-call IDs, own-kernel cleanup and readback."""
    if not re.fullmatch(r"runtimes/[A-Za-z0-9_-]{1,128}", args.runtime):
        raise ProbeError("INVALID_RUNTIME")
    path = journal_path(args.journal)
    session_id = "readiness-" + uuid.uuid4().hex
    session = args.runtime + "/sessions/" + session_id
    execution_id = str(uuid.uuid4())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ProbeError("JOURNAL_EXISTS") from None
    except OSError:
        raise ProbeError("JOURNAL_WRITE_FAILED") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        record(
            stream,
            phase="prepared",
            runtime=args.runtime,
            session=session,
            execution_id=execution_id,
        )
        try:
            token = auth.access_token(args.token_file, args.client_file, required_scope=auth.MANAGE)
        except Exception:  # pylint: disable=broad-exception-caught
            raise ProbeError("MANAGEMENT_AUTH_REQUIRED") from None
        client = MCP(token)
        owned, primary, outcome = False, None, None
        try:
            record(stream, phase="initialize_pending")
            client.initialize()
            record(stream, phase="create_session_pending")
            created = client.call(
                "create_session", {"parent": args.runtime, "sessionId": session_id}
            )
            if created.get("name") != session:
                raise ProbeError("REMOTE_SESSION_MISMATCH")
            owned = True
            record(stream, phase="execute_code_pending")
            result = client.call(
                "execute_code", {"session": session, "executionId": execution_id, "code": CODE}
            )
            outcome = parse_result(result, session, execution_id)
            record(stream, **outcome)
        except ProbeError as error:
            primary = error
            record(stream, phase="stopped", error=str(error))
        finally:
            if owned:
                try:
                    cleanup_session(client, stream, args.runtime, session)
                except ProbeError:
                    record(stream, phase="session_cleanup_required")
                    primary = primary or ProbeError("SESSION_CLEANUP_REQUIRED")
        if primary:
            raise primary
        return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--client-file", type=Path)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--journal", type=Path)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"runtimes/[A-Za-z0-9_-]{1,128}", args.runtime):
        print(json.dumps({"error": "INVALID_RUNTIME"}))
        return 2
    if not args.execute:
        print(
            json.dumps(
                {
                    "phase": "plan",
                    "runtime": args.runtime,
                    "purpose": "fixed hardware/storage readiness; no training",
                    "cleanup": "parent must delete runtime after probe",
                }
            )
        )
        return 0
    try:
        result = run_probe(args)
        print(json.dumps(result))
        return 0 if result["phase"] == "complete" else 3
    except ProbeError as error:
        print(json.dumps({"error": str(error)}))
        return 2
    except Exception:  # pylint: disable=broad-exception-caught
        print(json.dumps({"error": "PROBE_STOPPED_RECONCILE_JOURNAL"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
