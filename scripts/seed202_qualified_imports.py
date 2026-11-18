"""Authenticate and lease the plan-bound HOST import roots before credentials.

Use the existing V7 Win32 lease implementation, not a replacement controller.
This context surrounds the existing command scope and normal HOST execution.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path


def _require(condition, code):
    if not condition:
        raise ValueError(code)


@contextlib.contextmanager
def authenticated_imports(args):
    """Retain source/evidence write-denial handles through SDK and service reads."""
    from seed202_pilot.cli import _candidate
    from seed202_pilot.prepared_files import check, lease
    from seed202_pilot.qualified_bridge import descriptor, verify_files

    candidate = Path(args.candidate_dir).resolve()
    _path, inputs = descriptor(candidate)
    _require(
        args.science_dir is not None
        and Path(args.science_dir).resolve() == Path(inputs["science"]).resolve(),
        "QUALIFIED_SCIENCE_IMPORT_ROOT",
    )
    payload = candidate / "payload"
    manifest = json.loads((payload / "manifest.json").read_bytes())
    files = [payload / name for name in [*manifest["files"], "manifest.json"]]
    files += [
        candidate / name
        for name in (
            "candidate.json",
            "PLAN-UNAPPROVED.json",
            "sorting-mqar-seed202-pilot-v1.zip",
        )
    ]
    files += [Path(name) for name in inputs["files"]]
    with lease(list(dict.fromkeys(map(str, files)))) as local:
        # Replay authenticated archive/extracted equality while all source handles
        # are already held. A self-consistent mutable manifest is insufficient.
        _candidate(candidate)
        verify_files(inputs)
        check(local)
        try:
            yield
        finally:
            check(local)
