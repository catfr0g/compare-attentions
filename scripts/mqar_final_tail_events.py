"""Strict raw-wire pagination for actual inference quality events."""

import json
import math

from mqar_final_tail_native import require


def read_quality(task_id, metrics):
    from clearml import Task
    from clearml.backend_api.services import events as api_events
    from mqar_final_tail_tracking import verify_unsampled_events

    rows, pages, seen, cursors = [], [], set(), set()
    scroll, total = None, None
    while True:
        request = api_events.GetTaskEventsRequest(
            task=task_id,
            order="asc",
            scroll_id=scroll,
            batch_size=10000,
            event_type="training_stats_scalar",
        )
        result = Task._get_default_session().send(request)  # pylint: disable=protected-access
        raw = result.response_data
        require(
            type(raw) is dict
            and type(raw.get("events")) is list
            and type(raw.get("total")) is int
            and raw["total"] == len(metrics),
            "RAW_QUALITY_PAGE",
        )
        total = raw["total"] if total is None else total
        require(raw["total"] == total, "QUALITY_TOTAL_DRIFT")
        pages.append(raw)
        for row in raw["events"]:
            require(type(row) is dict and row.get("task", task_id) == task_id, "QUALITY_TASK")
            require(
                type(row.get("value")) in (float, int) and math.isfinite(row["value"]),
                "QUALITY_FINITE",
            )
            identity = json.dumps(row, sort_keys=True, allow_nan=False)
            require(identity not in seen, "QUALITY_DUPLICATE")
            seen.add(identity)
            rows.append(row)
        require(len(rows) <= total, "QUALITY_OVERSHOOT")
        if len(rows) == total:
            break
        scroll = raw.get("scroll_id")
        require(
            bool(raw["events"]) and type(scroll) is str and bool(scroll) and scroll not in cursors,
            "QUALITY_CURSOR",
        )
        cursors.add(scroll)
    normalized = verify_unsampled_events({"events": rows, "total": total}, metrics)
    return {**normalized, "raw_pages": pages}
