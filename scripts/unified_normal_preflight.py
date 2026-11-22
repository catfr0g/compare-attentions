"""HOST-only immutable plan preflight; no provider proof or freshness is cached."""

from __future__ import annotations

import contextlib
import json
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from unified_normal_contract import canonical, require
from unified_normal_live import lease_owner


@dataclass
class _Preflight:
    owner: object
    pins: dict
    plan: str
    roots: tuple
    closure: tuple
    active: bool = True


_PREFLIGHT: ContextVar[_Preflight | None] = ContextVar("unified_static_preflight", default=None)


def _closure(roots):
    from seed202_pilot.prepared_files import safe_root

    rows = []
    for name in roots:
        root = safe_root(name)
        for path in root.rglob("*"):
            require(not path.is_symlink() and not path.is_junction(), "UNIFIED_INPUT_LINK")
            rows.append((str(path), path.is_dir()))
    return tuple(sorted(rows))


def _current(state):
    from seed202_pilot.prepared_files import check

    require(state.active and state.owner is lease_owner(state.pins), "STATIC_PREFLIGHT_OWNERSHIP")
    check(state.pins)
    require(_closure(state.roots) == state.closure, "STATIC_PREFLIGHT_CLOSURE")


def authenticated_plan(plan):
    """Only exact prevalidated JSON under the same live handles may skip reconstruction."""
    state = _PREFLIGHT.get()
    if state is None:
        return False
    _current(state)
    require(canonical(plan) == state.plan, "EXACT_PLAN_PINS")
    return True


@contextlib.contextmanager
def static_preflight(candidate, pins, plan_path, review_path):
    """Run unchanged reconstruction BEFORE LIVE and invalidate reuse on every exit."""
    from seed202_pilot.prepared_files import check, safe_path
    from seed202_pilot.seal import _plan

    require(_PREFLIGHT.get() is None, "STATIC_PREFLIGHT_NESTED")
    owner = lease_owner(pins)
    require(owner is not None, "STATIC_PREFLIGHT_OWNERSHIP")
    candidate = Path(candidate).resolve()
    plan_path, review_path = safe_path(plan_path), safe_path(review_path)
    require({str(plan_path), str(review_path)} <= pins.keys(), "STATIC_PREFLIGHT_COVERAGE")
    plan = json.loads(plan_path.read_bytes())
    require(Path(plan["candidate_root"]).resolve() == candidate, "EXACT_PLAN_PINS")
    request_path = candidate / "payload/unified/request.json"
    roots = [candidate / "payload" if (candidate / "payload").is_dir() else candidate]
    if request_path.exists():
        require(
            {
                str(candidate / name)
                for name in (
                    "candidate.json",
                    "PLAN-UNAPPROVED.json",
                    "sorting-mqar-seed202-pilot-v1.zip",
                )
            }
            <= pins.keys(),
            "STATIC_PREFLIGHT_COVERAGE",
        )
        request = json.loads(request_path.read_bytes())
        require({str(request_path), *request["files"]} <= pins.keys(), "STATIC_PREFLIGHT_COVERAGE")
        roots += list(map(Path, request["roots"]))
    roots = tuple(sorted(set(map(str, roots))))
    before = _closure(roots)
    require(
        all(name in pins for name, directory in before if not directory),
        "STATIC_PREFLIGHT_COVERAGE",
    )
    check(pins)
    from unified_normal_contract import held_inputs

    scope = held_inputs(candidate, pins) if request_path.exists() else contextlib.nullcontext()
    with scope:
        _plan(plan)
    state = _Preflight(owner, pins, canonical(plan), roots, before)
    _current(state)
    token = _PREFLIGHT.set(state)
    try:
        yield
    finally:
        try:
            _current(state)
        finally:
            state.active = False
            _PREFLIGHT.reset(token)
