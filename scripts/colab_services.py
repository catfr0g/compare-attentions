"""Private phase-specific credentials; imported by service children, never setup."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import runpy
import sys
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from urllib.parse import urlsplit

sys.dont_write_bytecode = True

from colab_package import checksum  # noqa: E402  # pylint: disable=wrong-import-position
from colab_runner import clean_environment, preflight  # noqa: E402  # pylint: disable=wrong-import-position

KEYS = (
    "CLEARML_API_ACCESS_KEY",
    "CLEARML_API_SECRET_KEY",
    "CLEARML_API_HOST",
    "CLEARML_WEB_HOST",
    "CLEARML_FILES_HOST",
    "HF_TOKEN",
    "HF_REPO",
)


def load_credentials(path):
    """Read only the explicitly supplied file, without environment interpolation."""
    from dotenv import dotenv_values  # pylint: disable=import-outside-toplevel

    path = Path(path)
    if not path.is_file():
        raise ValueError("An existing separate private env file is required")
    try:
        values = dotenv_values(path, interpolate=False)
    except Exception:  # pylint: disable=broad-exception-caught
        raise ValueError("Cannot parse private env file") from None
    missing = [
        key for key in KEYS if not isinstance(values.get(key), str) or not values[key].strip()
    ]
    if missing:
        raise ValueError("Missing private env keys: " + ", ".join(missing))
    for key in KEYS:
        value = values[key]
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("Invalid private env value: " + key)
        if key.endswith("HOST"):
            try:
                url = urlsplit(value)
                valid = (
                    url.scheme in ("https", "http")
                    and url.hostname
                    and url.username is None
                    and url.password is None
                    and (url.port is None or url.port > 0)
                    and value == value.strip()
                    and not any(c.isspace() for c in value)
                )
            except ValueError:
                valid = False
            if not valid:
                raise ValueError("Invalid private env URL: " + key)
    # Do not import huggingface_hub here: it snapshots HF_HUB_OFFLINE at import.
    repo = values["HF_REPO"]
    segment = r"[A-Za-z0-9_](?:[A-Za-z0-9_.-]{0,94}[A-Za-z0-9_])?"
    if (
        not re.fullmatch(segment + "/" + segment, repo)
        or ".." in repo
        or "--" in repo
        or repo.endswith(".git")
    ):
        raise ValueError("Invalid private env HF_REPO") from None
    return {key: values[key] for key in KEYS}


def service_environment(values, phase):
    """Enable only the service needed by this child, never inherited config."""
    env = clean_environment()
    if phase == "train":
        env.update({key: values[key] for key in KEYS if key.startswith("CLEARML_")})
        env["CLEARML_CONFIG_FILE"] = os.devnull
    elif phase == "upload":
        env.pop("HF_HUB_OFFLINE", None)
        env.update({key: values[key] for key in ("HF_TOKEN", "HF_REPO")})
    else:
        raise ValueError("Unknown service phase")
    return env


class RedactedOutput:
    """Buffer lines so fragmented SDK writes cannot expose private values."""

    def __init__(self, target, values):
        self.target = target
        self.values = sorted(set(values), key=len, reverse=True)
        self.pending = ""

    def write(self, text):
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self._emit(line + "\n")
        return len(text)

    def _emit(self, text):
        for value in self.values:
            text = text.replace(value, "[REDACTED]")
        self.target.write(text)

    def flush(self):
        self.target.flush()

    def finish(self):
        self._emit(self.pending)
        self.pending = ""
        self.flush()

    @property
    def encoding(self):
        """Expose text encoding metadata without forwarding raw stream access."""
        return self.target.encoding

    @property
    def closed(self):
        """Expose only the stream state required by ClearML attachment."""
        return self.target.closed

    def isatty(self):
        return False


@contextlib.contextmanager
def clearml_stream_redaction(values):
    """Redact before SDK capture, not just its terminal copy; retain console logging."""
    from clearml.backend_interface.logger import PrintPatchLogger  # pylint: disable=import-outside-toplevel

    original_write = PrintPatchLogger.write
    buffers = {}
    lock = RLock()

    def write(stream, text):
        with lock:
            if id(stream) not in buffers:
                target = SimpleNamespace(
                    write=lambda clean: original_write(stream, clean), flush=lambda: None
                )
                buffers[id(stream)] = RedactedOutput(target, values)
            return buffers[id(stream)].write(text)

    PrintPatchLogger.write = write
    try:
        yield
    finally:
        try:
            with lock:
                for output in buffers.values():
                    # Complete a final partial line for both SDK console and terminal.
                    if output.pending:
                        output.write("\n")
                    output.finish()
        finally:
            PrintPatchLogger.write = original_write


def require_checkpoint_proof(package_dir, arguments):
    """Require the isolated pretrain probe receipt before any trainer import."""
    from colab_checkpoint_backup import identity, receipt_path  # pylint: disable=import-outside-toplevel

    locator = argparse.ArgumentParser(add_help=False)
    locator.add_argument("--result-dir", required=True, type=Path)
    result, _ = locator.parse_known_args(arguments)
    proof = json.loads(receipt_path(result.result_dir).read_text(encoding="utf-8"))
    if (
        proof.get("verified") is not True
        or not re.fullmatch(r"[0-9a-f]{40}", str(proof.get("revision")))
        or proof.get("identity") != identity(package_dir, result.result_dir)
    ):
        raise ValueError("Verified HF checkpoint startup proof required")


def train(env_file, arguments, package_dir=None, *, hf_checkpoints=False):
    """ClearML SDK must be imported only after injecting the private environment."""
    preflight()
    values = load_credentials(env_file)
    flags = [arg for arg in arguments if arg.startswith("--tracker")]
    if flags != ["--tracker"] or arguments[
        arguments.index("--tracker") + 1 : arguments.index("--tracker") + 2
    ] != ["clearml"]:
        raise ValueError("Training requires --tracker clearml; no offline fallback")
    if any(arg.startswith("--hf-repo") for arg in arguments):
        raise ValueError("HF upload belongs to the post-gate phase")
    if hf_checkpoints:
        if package_dir is None:
            raise ValueError("HF checkpoints require --package-dir")
        require_checkpoint_proof(package_dir, arguments)
    env = service_environment(values, "train")
    os.environ.clear()
    os.environ.update(env)
    original_argv = sys.argv
    sys.argv = ["compare_attention.cli.train", *arguments]
    stdout = RedactedOutput(sys.stdout, values.values())
    stderr = RedactedOutput(sys.stderr, values.values())
    try:
        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            clearml_stream_redaction(values.values()),
            contextlib.ExitStack() as stack,
        ):
            if hf_checkpoints:
                from colab_checkpoint_backup import checkpoint_backups  # pylint: disable=import-outside-toplevel

                stack.enter_context(checkpoint_backups(env_file, package_dir))
            runpy.run_module("compare_attention.cli.train", run_name="__main__", alter_sys=True)
    finally:
        sys.argv = original_argv
        stdout.finish()
        stderr.finish()


def gated_result_files(result, package):
    """Bind upload-only retries to the complete already-gated local result."""
    provenance = json.loads((result / "colab-provenance.json").read_text())
    if provenance["package_manifest_sha256"] != checksum(package / "manifest.json"):
        raise ValueError("Result package identity mismatch")
    paths = {}
    for path in result.rglob("*"):
        if path.is_symlink() or path.name.endswith((".env", ".env.example")):
            raise ValueError("Unexpected private or symlink result input")
        if path.is_file():
            paths[path.relative_to(result).as_posix()] = path
    expected = {name: checksum(path) for name, path in paths.items()}
    bound = {name: sha for name, sha in expected.items() if name != "colab-provenance.json"}
    if (
        bound != provenance["result_sha256"]
        or not {"model.pt", "test-evaluation.json", "colab-native-gate.json"} <= bound.keys()
    ):
        raise ValueError("Incomplete or changed gated results")
    return paths, expected


def upload(env_file, result, package, *, api_factory=None, download=None, upload_fn=None):  # pylint: disable=too-many-locals
    """Upload complete results, then verify private HEAD, listing and every byte."""
    values = load_credentials(env_file)
    env = service_environment(values, "upload")
    os.environ.clear()
    os.environ.update(env)
    result = Path(result)
    try:
        # Keep remote exceptions/progress from echoing credentials or private identifiers.
        with (
            open(os.devnull, "w", encoding="utf-8") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            if api_factory is None or download is None:
                from huggingface_hub import HfApi, hf_hub_download  # pylint: disable=import-outside-toplevel

                api_factory = api_factory or HfApi
                download = download or hf_hub_download
            paths, expected = gated_result_files(result, package)
            repo, token = values["HF_REPO"], values["HF_TOKEN"]
            api = api_factory(token=token)
            from huggingface_hub.errors import RepositoryNotFoundError  # pylint: disable=import-outside-toplevel

            try:
                existing = api.model_info(repo, revision="main", token=token)
            except RepositoryNotFoundError:
                pass  # Existing upload_results creates a missing repository private=True.
            else:
                if existing.private is not True:
                    raise ValueError("Refusing to upload into a public repository")
            if upload_fn is None:
                from compare_attention.train.hf import upload_results  # pylint: disable=import-outside-toplevel

                upload_results(result, repo, api_factory=lambda: api)
            else:
                upload_fn(result, repo)
            info = api.model_info(repo, revision="main", token=token)
            if info.private is not True or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
                raise ValueError("Private repository HEAD was not verified")
            prefix = f"runs/{result.name}/"
            listing = set(
                api.list_repo_files(repo, revision=info.sha, repo_type="model", token=token)
            )
            if {prefix + name for name in paths} != {
                name for name in listing if name.startswith(prefix)
            }:
                raise ValueError("Remote result files missing")
            for name, sha in expected.items():
                downloaded = download(
                    repo_id=repo,
                    filename=prefix + name,
                    repo_type="model",
                    revision=info.sha,
                    token=token,
                    force_download=True,
                )
                if checksum(Path(downloaded)) != sha or checksum(paths[name]) != sha:
                    raise ValueError("Remote/local result bytes mismatch")
            receipt = {"verified": True, "revision": info.sha, "sha256": expected}
            receipt_path = result.parent / (result.name + ".hf-upload-receipt.json")
            with receipt_path.open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2)
    except Exception:  # pylint: disable=broad-exception-caught
        raise RuntimeError(
            "HF upload/readback failed; local results retained while session lives; "
            "use --upload-only"
        ) from None
    print("HF upload and pinned-revision readback verified; receipt saved beside result directory")


def main(argv=None):
    """Never expose credential values or original SDK exception text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "train", "upload"))
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--hf-checkpoints", action="store_true")
    args, remainder = parser.parse_known_args(argv)
    try:
        if args.phase == "check":
            load_credentials(args.env_file)
            print("Configured private env keys: " + ", ".join(KEYS))
        elif args.phase == "train":
            train(
                args.env_file,
                remainder[1:] if remainder[:1] == ["--"] else remainder,
                package_dir=args.package_dir,
                hf_checkpoints=args.hf_checkpoints,
            )
        else:
            if args.result_dir is None or args.package_dir is None:
                raise ValueError("upload requires result-dir and package-dir")
            upload(args.env_file, args.result_dir, args.package_dir)
    except Exception:  # pylint: disable=broad-exception-caught
        print(
            "Service phase failed; check private env keys, service access and gated results. "
            "No offline fallback.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
