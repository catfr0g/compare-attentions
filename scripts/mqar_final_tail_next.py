"""Offline whole-composite replay and conditional MQAR Classic predecessor witness."""

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from types import FunctionType

from mqar_final_tail_native import canonical, digest, require
from mqar_final_tail_runner import create_json

SORTING = {
    "sorting-classic-seed202": "d0376599d2934db6ea87b5e455a9ca90f524cbf10d98ec5129bf56f931c14494",
    "sorting-deltanet-seed202": "d5fbc3a20b67b3be0400f786fbd461993ad07fe640bdca6a718364eeaa160a77",
}
STATE_SHA = "566595f993e6266557345bb33fb8c64e93ab875d4adbd4cd7e378ea3c37238a9"


def ordinary_is_unchanged(candidate):
    path = Path(candidate) / "inherited-two-stage-host/seed202_pilot/state.py"
    require(digest(path) == STATE_SHA, "UNCHANGED_ORDINARY_ACCEPTANCE_SOURCE")
    return True


def review_transition(value, composite_sha, release_sha):
    require(
        type(value) is dict
        and set(value)
        == {
            "status",
            "independent",
            "scope",
            "composite_sha256",
            "release_sha256",
            "permission",
            "sorting_predecessors",
        }
        and value["status"] == "PASS"
        and value["independent"] is True
        and value["scope"] == "MQAR202_QUALIFIED_CLASSIC_TO_DELTA_WITNESS"
        and value["composite_sha256"] == composite_sha
        and value["release_sha256"] == release_sha
        and value["permission"] == "STANDING_CONDITIONAL_MQAR_DELTA_ONLY_NO_SUBMISSION"
        and value["sorting_predecessors"] == SORTING,
        "INDEPENDENT_TRANSITION_REVIEW_REQUIRED",
    )


def replay(spec):
    from mqar_final_tail_composite import composite
    from mqar_final_tail_host_science import paths
    from mqar_final_tail_native import audit
    from mqar_final_tail_package import check

    require(
        set(spec)
        == {
            "release",
            "result",
            "native",
            "services",
            "provider",
            "review",
            "plan",
            "data",
            "input",
            "output",
        },
        "EXACT_QUALIFIED_REPLAY_INPUTS",
    )
    release = Path(spec["release"])
    candidate = release.parent
    ordinary_is_unchanged(candidate)
    check(candidate)
    from mqar_final_tail_host_contract import activate

    activate({"candidate_root": str(candidate)})
    plan = json.loads(Path(spec["plan"]).read_bytes())
    require(
        digest(candidate / "dataset/recovery-input.zip.bin") == plan["input_sha256"],
        "INPUT_ARCHIVE",
    )
    paths(spec["input"], candidate / "dataset/recovery-input.zip.bin", plan["input_sha256"])
    native = audit(
        Path(spec["input"]) / "original",
        None,
        spec["data"],
        release / "configs/mqar-k64-classic-seed202.yaml",
    )
    require(
        canonical(native) == canonical(json.loads(Path(spec["native"]).read_bytes())),
        "NATIVE_REPLAY",
    )
    output = Path(spec["output"])
    before = output.read_bytes()
    provider = json.loads(Path(spec["provider"]).read_bytes())
    provider = {"ref" if k == "target" else k: v for k, v in provider.items()}
    value = composite(
        release,
        spec["result"],
        native,
        json.loads(Path(spec["services"]).read_bytes()),
        provider,
        json.loads(Path(spec["review"]).read_bytes()),
        plan,
        spec["data"],
        output,
    )
    require(
        output.read_bytes() == before and canonical(value) == canonical(json.loads(before)),
        "EXACT_WHOLE_CANONICAL_REPLAY",
    )
    return value


def witness(spec, review, sorting_paths):
    from mqar_final_tail_host_contract import activate

    release = Path(spec["release"])
    review_transition(review, digest(spec["output"]), digest(release / "manifest.json"))
    value = replay(spec)
    require(set(sorting_paths) == set(SORTING), "BOTH_IMMUTABLE_SORTING_PREDECESSORS")
    for case, path in sorting_paths.items():
        require(digest(path) == SORTING[case], "EXACT_ACCEPTED_SORTING_BYTES")
        old = json.loads(Path(path).read_bytes())
        require(old["later_pilots_authorized"] is False, "PRESERVE_SORTING_AUTHORITY")
    # Import only the authenticated unchanged original state consumer. The callback is scoped.
    activate({"candidate_root": str(release.parent)})
    from seed202_pilot import state

    original = state._accepted  # pylint: disable=protected-access

    def accepted(path, expected):
        if expected.case_id in SORTING:
            require(
                Path(path).resolve() == Path(sorting_paths[expected.case_id]).resolve(),
                "SORTING_PATH",
            )
        elif expected.case_id == "mqar-k64-classic-seed202":
            require(
                Path(path).resolve() == Path(spec["output"]).resolve(), "EXACT_CLASSIC_PREDECESSOR"
            )
            require(json.loads(Path(path).read_bytes()) == value, "CLASSIC_WHOLE_REPLAY_DRIFT")
        else:
            original(path, expected)

    next_case = FunctionType(
        state.next_case.__code__,
        {**state.next_case.__globals__, "_accepted": accepted},
        state.next_case.__name__,
        state.next_case.__defaults__,
    )
    selected = next_case([Path(sorting_paths[k]) for k in SORTING] + [Path(spec["output"])])
    require(selected.case_id == "mqar-k64-deltanet-seed202", "ONLY_NEXT_CONDITIONAL_MQAR_DELTA")
    require(state._accepted is original, "ORDINARY_ACCEPTANCE_NOT_PATCHED")  # pylint: disable=protected-access
    return {
        "schema": "MQAR202_TAGGED_QUALIFIED_CLASSIC_WITNESS_V1",
        "case_id": "mqar-k64-classic-seed202",
        "next_case": selected.case_id,
        "composite_sha256": digest(spec["output"]),
        "qualification": value["qualification"],
        "sorting_predecessors": SORTING,
        "transition_review": review,
        "original_provider_statuses": ["ERROR", "ERROR"],
        "ordinary_training_acceptance_fabricated": False,
        "submitted": False,
        "fresh_launch_authority_required": True,
        "later_pilots_authorized": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("spec", "review", "sorting-classic", "sorting-delta", "science", "work", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "OFFLINE_CPU_WITNESS_ONLY")
    socket.socket.connect = lambda *_a, **_k: (_ for _ in ()).throw(
        PermissionError("OFFLINE_WITNESS")
    )
    from mqar_final_tail_package import bind_wheel, check

    check(args.science)
    spec = json.loads(args.spec.read_bytes())
    bind_wheel(args.science / "compare_attention-0.1.0-py3-none-any.whl", args.work / "wheel")
    sys.path[1:1] = [
        str(Path(spec["release"]) / "seed202"),
        *[str(args.science / p) for p in ("operator", "src", "tools", "operator/inherited")],
    ]
    result = witness(
        spec,
        json.loads(args.review.read_bytes()),
        {
            "sorting-classic-seed202": args.sorting_classic,
            "sorting-deltanet-seed202": args.sorting_delta,
        },
    )
    create_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
