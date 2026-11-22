"""Versioned read-only HOST semantics; never rewrite producer records or inputs."""

import hashlib
import json
from itertools import groupby
from pathlib import Path

RELEASE = "normal-host-readonly-tiny-domain-v4"


def canonical(value):
    """Use the producer's JSON comparison domain, retaining all full-row fields."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def event_view(events):
    """Canonicalize only contiguous equal-iteration groups, preserving multiplicity."""
    return [
        row
        for _, rows in groupby(events, lambda row: row["iter"])
        for row in sorted(rows, key=canonical)
    ]


def raw_view(raw):
    """Normalize independent flat/page streams, then preserve original page lengths."""
    pages = raw["pages"]
    if type(pages) is not list or not pages:
        raise ValueError("CLEARML_PAGE_SCHEMA")
    rows, cursors, seen = [], set(), set()
    for index, page in enumerate(pages):
        if (
            type(page.get("events")) is not list
            or type(page.get("total")) is not int
            or page["total"] != raw["total"]
            or type(page.get("returned")) is not int
            or page["returned"] != len(page["events"])
        ):
            raise ValueError("CLEARML_PAGE_SCHEMA")
        for row in page["events"]:
            key = canonical(row)
            if key in seen:
                raise ValueError("CLEARML_PAGE_DUPLICATE")
            seen.add(key)
            rows.append(row)
        if len(rows) > raw["total"]:
            raise ValueError("CLEARML_PAGE_OVERSHOOT")
        if index + 1 < len(pages):
            if not page["events"] or len(rows) >= raw["total"]:
                raise ValueError("CLEARML_PAGE_INCOMPLETE")
            cursor = page.get("scroll_id")
            if type(cursor) is not str or not cursor or cursor in cursors:
                raise ValueError("CLEARML_PAGE_CURSOR")
            cursors.add(cursor)
    if len(rows) != raw["total"] or type(raw["total"]) is not int:
        raise ValueError("CLEARML_PAGE_INCOMPLETE")
    stream = event_view(rows)
    if stream != event_view(raw["events"]):
        raise ValueError("UNIFIED_HOST_PAGE_EVENT_JOIN")
    offset = 0
    viewed_pages = []
    for page in pages:
        length = len(page["events"])
        viewed_pages.append({**page, "events": stream[offset : offset + length]})
        offset += length
    return {**raw, "events": event_view(raw["events"]), "pages": viewed_pages}


def training_view(value):
    """Retain every envelope/raw/page field except equal-iteration row ordering."""
    return {**value, "raw": raw_view(value["raw"])}


def lifecycle(case, source, native, training, publications):
    """One constructor shared by genuine observation and strict whole-object replay."""
    return {
        "schema": "normal-training-lifecycle-acceptance-v1",
        "case": case,
        "scope": "ENGINEERING_CPU_SDK_FIXTURE_ONLY"
        if case["engineering_only"]
        else "TARGET_NORMAL_LIFECYCLE",
        "status": "ACCEPTED_NORMAL_LIFECYCLE",
        "source": source,
        "native": native,
        "training": training["receipt"],
        "immutable_publications": publications,
        "task_quality_acceptance": "SEPARATE_TASK_SPECIFIC_ACCEPTANCE_REQUIRED",
        "next_launch_authorized": False,
    }


def reader_identity(*, candidate=None, recorded=None, inputs=None):
    """Pin V4 separately; authenticate historical readers against their own bytes."""
    if recorded is not None and recorded.get("release") in {
        "normal-host-readonly-equal-iteration-v2",
        "normal-host-readonly-nonblocking-tracking-v3",
    }:
        root = Path(candidate) / "payload/source/src"
        names = {
            "unified_normal_host.py",
            "unified_normal_host_view.py",
            "unified_normal_ordinary.py",
            "unified_normal_publications.py",
        }
        if (
            set(recorded) != {"release", "purpose", "files"}
            or recorded["purpose"] != "posttraining-readonly-observer-client"
            or set(recorded["files"]) != names
        ):
            raise ValueError("UNIFIED_LEGACY_READER_BYTES")
        roots = {root}
        if inputs is not None:
            roots = {
                Path(path).parent
                for path, expected in inputs["files"].items()
                if Path(path).name == "unified_normal_host_view.py"
                and expected["sha256"] == recorded["files"]["unified_normal_host_view.py"]
            }
        for selected_root in roots:
            if all(
                (selected_root / name).is_file()
                and (inputs is None or str((selected_root / name).resolve()) in inputs["files"])
                and hashlib.sha256((selected_root / name).read_bytes()).hexdigest()
                == recorded["files"][name]
                for name in names
            ):
                return recorded
        raise ValueError("UNIFIED_LEGACY_READER_BYTES")
    root = Path(__file__).parent
    return {
        "release": RELEASE,
        "purpose": "posttraining-readonly-observer-client",
        "files": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "unified_normal_host.py",
                "unified_normal_host_view.py",
                "unified_normal_ordinary.py",
                "unified_normal_publications.py",
            )
        },
    }
