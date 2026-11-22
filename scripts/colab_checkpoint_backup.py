"""Opaque native checkpoint transport; imports no ML SDK at module load."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from colab_checkpoint_protocol import checksum, identity, receipt_path

__all__ = [
    "checkpoint_backups",
    "checksum",
    "identity",
    "probe",
    "publish",
    "receipt_path",
    "restore",
]


def transport(function, *arguments, **options):
    """No SDK output, account identifiers or raw exception text may escape."""
    disabled = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with (
            open(os.devnull, "w", encoding="utf-8") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            return function(*arguments, **options)
    except Exception:  # pylint: disable=broad-exception-caught
        raise RuntimeError(
            "Checkpoint transport failed; stop and reconcile pinned state; no automatic retry"
        ) from None
    finally:
        logging.disable(disabled)


def probe(package, result, **options):
    """Prove private HF writes and pinned bytes before trainer construction."""
    from colab_checkpoint_store import probe_store  # pylint: disable=import-outside-toplevel

    return transport(probe_store, package, result, **options)


def publish(package, result, **options):
    """Publish only an already successful unchanged native full snapshot."""
    from colab_checkpoint_store import publish_store  # pylint: disable=import-outside-toplevel

    return transport(publish_store, package, result, **options)


def restore(package, run, revision, target, **options):
    """Fetch a pinned opaque snapshot; unchanged trainer remains resume authority."""
    from colab_checkpoint_store import restore_store  # pylint: disable=import-outside-toplevel

    return transport(restore_store, package, run, revision, target, **options)


@contextlib.contextmanager
def checkpoint_backups(env_file, package_dir):
    """Wrap native persistence temporarily; child never touches parent RNG/credentials."""
    from colab_runner import clean_environment  # pylint: disable=import-outside-toplevel

    from compare_attention.train.backup import BackupTracker  # pylint: disable=import-outside-toplevel

    original = BackupTracker.save_checkpoint

    def save(tracker, payload, step, keep_last):
        original(tracker, payload, step, keep_last)
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "publish",
                    "--env-file",
                    str(env_file),
                    "--package-dir",
                    str(package_dir),
                    "--result-dir",
                    str(tracker._result_dir),  # pylint: disable=protected-access
                ],
                env=clean_environment(),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            raise RuntimeError(
                "Checkpoint backup failed; local snapshot retained; stop and reconcile"
            ) from None

    BackupTracker.save_checkpoint = save
    try:
        yield
    finally:
        BackupTracker.save_checkpoint = original


# Keep all child-only SDK/credential state inside one bounded lifetime.
# pylint: disable-next=too-many-arguments,too-many-locals
def service_phase(
    env_file,
    package,
    result,
    phase,
    *,
    api_factory=None,
    download=None,
    run=None,
    revision=None,
    target=None,
):
    """Child-only credential loading: online flags and ephemeral cache precede SDK."""
    from colab_services import load_credentials, service_environment  # pylint: disable=import-outside-toplevel

    env_file, package = Path(env_file), Path(package)
    if env_file.resolve().is_relative_to(package.resolve()) or (
        result is not None and env_file.resolve().is_relative_to(Path(result).resolve())
    ):
        raise ValueError("Separate private credential file required")
    with tempfile.TemporaryDirectory(prefix="hf-checkpoint-sdk-") as cache:
        values = load_credentials(env_file)
        environment = service_environment(values, "upload")
        environment.update(
            HF_HOME=cache, HF_XET_CACHE=str(Path(cache) / "xet"), HF_HUB_DISABLE_PROGRESS_BARS="1"
        )
        os.environ.clear()
        os.environ.update(environment)
        if api_factory is None or download is None:
            from huggingface_hub import HfApi, hf_hub_download  # pylint: disable=import-outside-toplevel

            api_factory = api_factory or HfApi
            download = download or hf_hub_download
        options = {
            "api": api_factory(token=values["HF_TOKEN"]),
            "download": download,
            "repo": values["HF_REPO"],
            "token": values["HF_TOKEN"],
        }
        if phase == "probe":
            return probe(package, result, **options)
        if phase == "publish":
            return publish(package, result, **options)
        if phase == "restore" and all((run, revision, target)):
            return restore(package, run, revision, target, **options)
        raise ValueError("Checkpoint phase arguments missing")


def main(argv=None):
    """Explicit checkpoint service entry point with fail-closed secret-safe output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("probe", "publish", "restore"))
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--run")
    parser.add_argument("--revision")
    parser.add_argument("--target", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.phase != "restore" and args.result_dir is None:
            raise ValueError("result-dir required")
        transport(
            service_phase,
            args.env_file,
            args.package_dir,
            args.result_dir,
            args.phase,
            run=args.run,
            revision=args.revision,
            target=args.target,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        print(
            "Checkpoint transport failed; local snapshots retained; reconcile before retry",
            file=sys.stderr,
        )
        return 1
    print("Checkpoint transport pinned readback verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
