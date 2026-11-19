"""Bounded fresh synthetic CPU/Gloo lifecycle evidence worker."""

# ruff: noqa: E402 -- assert the CPU/bytecode contract before scientific imports.

import json
import os
import sys
from pathlib import Path

assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"

from compare_attention.train.backup import BackupTracker
from compare_attention.train.config import TrainConfig
from compare_attention.train.loop import train


def main():
    root = Path(sys.argv[1])
    config = TrainConfig.model_validate_json((root / "config.json").read_bytes())

    class FailingBackup(BackupTracker):
        def finish(self, result, model):
            super().finish(result, model)
            if (root / "sanitized-service").exists():
                from compare_attention.train.lifecycle import failure_phase

                try:
                    with failure_phase("clearml_flush"):
                        raise LookupError("PRIVATE_MESSAGE_NOT_ALLOWED")
                except LookupError:
                    raise RuntimeError("SAFE_SERVICE_WRAPPER") from None
            raise LookupError("PRIVATE_MESSAGE_NOT_ALLOWED")

    try:
        train(config, FailingBackup(root / "out"), distributed=True)
    except RuntimeError as error:
        # The real collective wrapper is a console consumer, not just JSON.
        observed = str(error)
        assert "PRIVATE_MESSAGE" not in observed, observed
        assert "LookupError" in observed, observed
        print(observed, flush=True)
        # Must already contain original class/phase, not the coordinated wrapper.
        report = json.loads((root / "out/lifecycle-failure.json").read_bytes())
        assert report["exception_class"] == "LookupError", report
        expected = "clearml_flush" if (root / "sanitized-service").exists() else "finish"
        assert report["phase"] == expected, report
        assert "PRIVATE_MESSAGE" not in json.dumps(report)
        (root / f"rank-{os.environ['RANK']}-observed.json").write_text(json.dumps(report))
    else:
        raise AssertionError("expected coordinated finish failure")


if __name__ == "__main__":
    main()
