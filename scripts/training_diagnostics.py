"""Best-effort auxiliary tracking, never an authority over native science."""

import json
import warnings


def observe(phase, operation, state=None):
    """Retain safe failure facts and raw caller observations without raising SDK errors."""
    state = {} if state is None else state
    try:
        result = operation()
    except Exception as error:  # pylint: disable=broad-exception-caught
        diagnostic = {
            "status": "warning",
            "authority": "AUXILIARY_ONLY",
            "phase": phase,
            "exception_class": type(error).__name__
            if type(error).__module__ == "builtins"
            else "Exception",
        }
        # Warning-as-error configuration must not turn a diagnostic back into a gate.
        try:
            warnings.warn("auxiliary tracking warning: " + phase, RuntimeWarning, stacklevel=2)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return {**state, "diagnostic": diagnostic}
    observed = {**state, **(result if isinstance(result, dict) else {})}
    observed.setdefault("diagnostic", {"status": "verified", "authority": "AUXILIARY_ONLY"})
    return observed


def validate_pages(raw):
    """Check every retained page copy; never normalize away corruption or missing rows."""
    pages, rows, cursors = raw["pages"], [], set()
    if type(pages) is not list or not pages or type(raw["total"]) is not int:
        raise ValueError("CLEARML_PAGE_SCHEMA")
    for index, page in enumerate(pages):
        events = page.get("events")
        if (
            type(events) is not list
            or type(page.get("total")) is not int
            or page["total"] != raw["total"]
            or (
                "returned" in page
                and (type(page["returned"]) is not int or page["returned"] != len(events))
            )
        ):
            raise ValueError("CLEARML_PAGE_SCHEMA")
        rows.extend(events)
        if len(rows) > raw["total"]:
            raise ValueError("CLEARML_PAGE_OVERSHOOT")
        if index + 1 < len(pages):
            cursor = page.get("scroll_id")
            if not events or type(cursor) is not str or not cursor or cursor in cursors:
                raise ValueError("CLEARML_PAGE_CURSOR")
            cursors.add(cursor)

    def canonical(row):
        return json.dumps(row, sort_keys=True, allow_nan=False)

    encoded = [canonical(row) for row in rows]
    if (
        len(rows) != raw["total"]
        or len(set(encoded)) != len(rows)
        or sorted(encoded) != sorted(canonical(row) for row in raw["events"])
    ):
        raise ValueError("CLEARML_PAGE_EVENT_JOIN")
