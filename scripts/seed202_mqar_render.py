"""Scoped saved-main adapter for the sole MQAR Classic step980 continuation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

RENDER_SHA = "d076e5d5665922b1c644156c6be42d21e07e892296f1449a64e122fe56c9bffa"
CASE = "mqar-k64-classic-seed202"
MODE = "MQAR_CLASSIC_STEP980_V1"


def render_kernel(
    case_id,
    nonce,
    operator_sha256,
    runtime_archive,
    *,
    resume_sha256,
    evidence_sha256="0" * 64,
    authority_sha256="0" * 64,
):
    """Reuse authenticated fresh MQAR mounts and add one explicit resume mount."""
    from seed202_pilot import render

    if (
        case_id != CASE
        or not isinstance(resume_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", resume_sha256)
    ):
        raise ValueError("EXACT_MQAR_RESUME_BINDING")
    if hashlib.sha256(Path(render.__file__).read_bytes()).hexdigest() != RENDER_SHA:
        raise ValueError("EXACT_PARENT_RENDER_SOURCE")
    source, metadata = render.render_kernel(case_id, nonce, operator_sha256, runtime_archive)
    before = '        "python", "-B", "-m", "seed202_pilot.cloud_runner",'
    after = '        "python", "-B", "-m", "seed202_pilot.mqar_resume_runner",'
    if source.count(before) != 1:
        raise ValueError("EXACT_PARENT_RUNNER_CALL")
    source = source.replace(before, after)
    # The inherited saved main already owns safe extraction and the command branch.
    # Bind that branch only for this adapter; the old Sorting guard is unchanged.
    import ast

    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "BINDING" for t in node.targets)
    )
    binding = ast.literal_eval(node.value)
    binding.update(
        resume_sha256=resume_sha256,
        resume_slug="seed202-resume-input-" + nonce[:16],
        continuation_mode=MODE,
        evidence_sha256=evidence_sha256,
        authority_sha256=authority_sha256,
    )
    old = ast.get_source_segment(source, node)
    if old is None or source.count(old) != 1:
        raise ValueError("EXACT_PARENT_BINDING")
    source = source.replace(old, "BINDING = " + repr(binding))
    main = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    resume_node = next(
        node
        for node in main.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Constant)
        and node.test.left.value == "resume_sha256"
    )
    branch = "    " + ast.get_source_segment(source, resume_node)
    early, late = branch.rsplit("\n", 1)
    # This adapter has mandatory resume bytes; no optional branch reaches setup.
    early = "\n".join(line[4:] for line in early.splitlines()[1:])
    if not late.strip().startswith('command += ["--resume-input",'):
        raise ValueError("EXACT_PRESETUP_CHECKPOINT_EXTRACTION_SEAM")
    source = source.replace(branch, "    " + late.strip())
    seam = "    from process_control import bounded_run\n"
    if source.count(seam) != 1:
        raise ValueError("EXACT_AUTHORITY_IMPORT_SEAM")
    source = source.replace(
        seam,
        """    # Read the authenticated validator without consulting ambient import caches.
    authority_scope = {}
    authority_source = release / "src/seed202_pilot/mqar_authority.py"
    exec(compile(authority_source.read_bytes(), str(authority_source), "exec"), authority_scope)
    authority_root = resolve_mount(
        mount_root, "zimike", BINDING["resume_slug"], "authority-inventory.json"
    ).parent
    authority_result = authority_scope["validate"](
        authority_root, BINDING["evidence_sha256"], BINDING["authority_sha256"]
    )
"""
        + early
        + '\n    imported_before = authority_scope["import_inventory"](release, science)\n'
        + seam,
    )
    seam = '    output = working / ("seed202-pilot-" + BINDING["nonce"])'
    source = source.replace(
        seam,
        """    authority_scope["check_imports"](release, science, imported_before)
    if authority_scope["validate"](
        authority_root, BINDING["evidence_sha256"], BINDING["authority_sha256"]
    ) != authority_result:
        raise ValueError("POSTIMPORT_AUTHORITY_DRIFT")
"""
        + seam,
    )
    seam = "    bounded_run(command, env, science, "
    if source.count(seam) != 1:
        raise ValueError("EXACT_AUTHORITY_CHILD_SEAM")
    source = source.replace(
        seam,
        """    command += ["--authority-root", str(authority_root),
                "--evidence-sha256", BINDING["evidence_sha256"],
                "--authority-sha256", BINDING["authority_sha256"]]
"""
        + seam,
    )
    metadata["dataset_sources"].append("zimike/" + binding["resume_slug"] + "/1")
    compile(source, "seed202-mqar-resume-kernel.py", "exec")
    return source, metadata
