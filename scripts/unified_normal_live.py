"""Retain genuine Win32 byte leases across one public probe/seal/action chain.

No approval, freshness result or source validation is cached here. Reuse is
possible only inside the owning context's live write-denial handles. Every
public CLI invocation still executes all its original guards and source checks.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import get_ident


@dataclass
class _LeaseOwner:
    """Shared lifetime state; copied Contexts must observe owner release."""

    pins: dict
    thread_id: int
    active: bool = True

    def usable(self):
        """Only the synchronous owning thread may borrow these live handles."""
        return self.active and self.thread_id == get_ident()


_LEASES: ContextVar[tuple[_LeaseOwner, ...]] = ContextVar("unified_normal_live_leases", default=())


@contextlib.contextmanager
def retained_lease(paths):
    """Hash on acquisition; reuse exact covered files only while handles stay held."""
    from seed202_pilot.prepared_files import check, lease, safe_path

    required = {str(safe_path(p)) for p in paths}
    for owner in reversed(_LEASES.get()):
        if not owner.usable():
            continue
        held = owner.pins
        if required <= held.keys():
            selected = {name: held[name] for name in sorted(required)}
            check(selected)
            try:
                yield selected
            finally:
                check(selected)
            return
    with lease(sorted(required)) as held:
        owner = _LeaseOwner(held, get_ident())
        token = _LEASES.set((*_LEASES.get(), owner))
        try:
            yield held
        finally:
            owner.active = False
            _LEASES.reset(token)


def held_digest(path):
    """Return initial SHA only for a file still protected by this context's handles."""
    if not _LEASES.get():
        return None
    from seed202_pilot.prepared_files import check, safe_path

    name = str(Path(path).absolute())
    for owner in reversed(_LEASES.get()):
        if not owner.usable():
            continue
        held = owner.pins
        if name in held:
            safe_path(path)
            check({name: held[name]})
            return held[name]["sha256"]
    return None


def _commands(commands):
    """Accept one exact action, never a retry, arbitrary command list or old proof."""
    from seed202_pilot.cli import _parser

    if type(commands) is not list or len(commands) != 3:
        raise ValueError("LIVE_EXACT_CHAIN")
    if any(type(v) is not list or any(type(s) is not str for s in v) for v in commands):
        raise ValueError("LIVE_ARGV_SCHEMA")
    probe, seal, action = [_parser().parse_args(v) for v in commands]
    if (probe.command, seal.command, action.command) not in {
        ("probe", "seal", "publish"),
        ("probe", "seal", "submit"),
    } or probe.action != action.command:
        raise ValueError("LIVE_EXACT_CHAIN")
    for name in (
        "candidate_dir",
        "case",
        "parent_archive",
        "private",
        "auth_env",
        "science_dir",
        "mqar_data",
        "lifetime",
        "journal",
        "plan",
        "review",
    ):
        if getattr(probe, name) is None and name not in {"private", "auth_env"}:
            raise ValueError("LIVE_SCOPE_REQUIRED")
        if not getattr(probe, name) == getattr(seal, name) == getattr(action, name):
            raise ValueError("LIVE_SCOPE_MISMATCH")
    if probe.preparation is None or probe.preparation != action.preparation:
        raise ValueError("LIVE_PREPARATION_SCOPE")
    if probe.output != seal.proof or seal.output != action.approval:
        raise ValueError("LIVE_OUTPUT_SCOPE")
    for path in (probe.output, seal.output):
        if path is None or Path(path).exists():
            raise ValueError("LIVE_CREATE_ONLY")
    return probe


def run(commands):
    """Prehash before LIVE, then execute the unchanged public operator, stop on failure."""
    from seed202_pilot import cli
    from seed202_pilot.preparation import paths
    from seed202_pilot.seal import _plan
    from unified_normal_contract import held_inputs

    logging.disable(logging.CRITICAL)
    probe = _commands(commands)
    # Process-only auth_env/private are intentionally not artifact inputs.
    files = [*paths(probe), probe.plan, probe.review, probe.preparation, probe.lifetime]
    with retained_lease(files) as local:
        with held_inputs(probe.candidate_dir, local):
            _plan(json.loads(probe.plan.read_bytes()))
        for argv in commands:
            # Seal alone has no authenticated_imports scope. Reuse the existing
            # supported scope here, never nest it inside another CLI command.
            scope = (
                held_inputs(probe.candidate_dir, local)
                if argv[0] == "seal"
                else contextlib.nullcontext()
            )
            with scope:
                result = cli.main(argv)
            if result:
                return result
    return 0


def main(argv=None):
    """Public driver: a reviewed JSON array of three ordinary CLI argv vectors."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--commands", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    try:
        return run(json.loads(args.commands.read_bytes()))
    except Exception as error:  # Same finite privacy boundary as the inherited operator.
        print(
            json.dumps(
                {"status": "STOP_GET_ONLY_NO_RETRY", "exception_class": type(error).__name__}
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
