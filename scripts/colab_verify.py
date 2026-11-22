"""Run the frozen public scientific gate plus pinned full-snapshot verifier."""

import json
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from colab_package import verify_directory  # noqa: E402  # pylint: disable=wrong-import-position


def verify(package, result, family, gpu):
    # Lazy import: package inspection on Windows needs no Torch or FLA.
    from compare_attention.exploration_gate import verify_exploration_run  # pylint: disable=import-outside-toplevel

    manifest = verify_directory(package)
    frozen = manifest["configs"][family]
    expected = {key: frozen[key] for key in ("model", "schedule", "optim")}
    expected.update(data=str(package / "data"), dataset=manifest["dataset_spec"])
    report = verify_exploration_run(result, expected_config=expected, expected_gpu=gpu)
    helper = runpy.run_path(str(package / "support/controls_verifier.py"))
    helper["verify_snapshot"](result, expected)
    with (result / "colab-native-gate.json").open("x", encoding="utf-8") as stream:
        json.dump({"scientific_gate": report, "snapshot_verified": True}, stream, indent=2)


if __name__ == "__main__":
    verify(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4])
