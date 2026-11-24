"""Finite future-only Delta404 scheduling, not a new scientific case or retry."""

from unified_normal_profiles import RULER_DELTA_TIME
from unified_normal_ruler import build_request as build_request
from unified_normal_ruler import deployment as ruler_deployment
from unified_normal_ruler import integrate_group as ruler_integrate_group
from unified_normal_ruler import transition as transition
from unified_normal_ruler import validate_request as validate_request


def integrate_group(source):
    """Keep historical contract rows intact; only this emitted host selects new rows."""
    from unified_normal_operator import replace

    ruler_integrate_group(source)
    replace(
        source / "contract.py",
        "from normal_ruler_scope import case_specs",
        "from normal_ruler_scope import deltanet404_case_specs as case_specs",
    )


def deployment(payload, plan, run, generation):
    return ruler_deployment(payload, plan, run, generation, profile_name=RULER_DELTA_TIME)
