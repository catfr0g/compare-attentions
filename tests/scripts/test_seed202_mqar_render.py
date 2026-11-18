"""Saved MQAR continuation entrypoint, without model execution or cloud writes."""

import ast
import sys
from pathlib import Path

WORK = Path("C:/Users/Michael/AppData/Local/hermes/cache/scratch/mqar202-repair-work-v1")


def test_saved_mqar_renderer_binds_all_five_inputs_and_public_resume_parser():
    sys.path.insert(0, str(WORK / "parent/source/src"))
    try:
        from seed202_mqar_render import render_kernel

        source, metadata = render_kernel(
            "mqar-k64-classic-seed202",
            "a" * 32,
            "b" * 64,
            WORK / "parent/inherited/runtime-order.zip.bin",
            resume_sha256="c" * 64,
        )
        compile(source, "kernel.py", "exec")
        binding = next(
            ast.literal_eval(node.value)
            for node in ast.parse(source).body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "BINDING" for t in node.targets)
        )
        assert binding["case_id"] == "mqar-k64-classic-seed202"
        assert binding["resume_sha256"] == "c" * 64
        assert binding["continuation_mode"] == "MQAR_CLASSIC_STEP980_V1"
        assert metadata["dataset_sources"] == [
            "zimike/sorting-mqar-seed202-" + "a" * 16 + "/1",
            "zimike/sorting-quality-input-fcff1b5daf574b70/1",
            "zimike/ca-kv-1c660aa5f54d88ea9e0c1ce4/1",
            "zimike/k64-private-20261109a1b2/1",
            "zimike/seed202-resume-input-" + "a" * 16 + "/1",
        ]
        assert '"seed202_pilot.mqar_resume_runner"' in source
        assert '"--resume-input"' in source
        assert 'resume_root / "original"' in source
        assert source.index("inspect_hardware(driver)") < source.index("operator_archive =")
        assert source.index("resume_archive =") < source.index("setup(science,")
        assert source.index("setup(science,") < source.index("    bounded_run(command")
    finally:
        sys.path.remove(str(WORK / "parent/source/src"))
