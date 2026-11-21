"""Fresh CPU structural whole-object replay; no genuine model or RNG restoration."""

import argparse
import contextlib
import hashlib
import logging
import os
import socket
import sys
import tempfile
from pathlib import Path

from unified_normal_contract import (
    Policy,
    VerifiedPredecessor,
    canonical,
    descriptor,
    digest,
    read,
    require,
)
from unified_normal_profiles import PROFILES


def replay(request, case_id):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "UNIFIED_REPLAY_CPU")
    require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "UNIFIED_REPLAY_NO_BYTECODE")
    path, value = descriptor(Path(request).parents[2])
    require(path.resolve() == Path(request).resolve(), "UNIFIED_REQUEST_PATH")
    rows = [row for row in value["predecessors"] if row["case_id"] == case_id]
    require(len(rows) == 1, "UNIFIED_REPLAY_CASE")
    row = rows[0]
    require(digest(row["path"]) == row["sha256"], "UNIFIED_RAW_ACCEPTANCE_PIN")
    policy = Policy(row["policy"])
    spec = row["replay"]
    logging.disable(logging.CRITICAL)

    def prohibited(*_args, **_kwargs):
        raise PermissionError("UNIFIED_STRUCTURAL_REPLAY_ONLY")

    socket.socket.connect = prohibited
    socket.create_connection = prohibited
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            if policy in {Policy.SORTING_CLASSIC, Policy.SORTING_DELTA}:
                import importlib.util

                module_spec = importlib.util.spec_from_file_location(
                    "frozen_sorting_replay", spec["driver"]
                )
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                module._run_replay(Path(spec["descriptor"]), case_id)  # pylint: disable=protected-access
                result = read(row["path"])
            elif policy == Policy.MQAR_CONTINUATION:
                science = Path(spec["science"])
                release = Path(spec["spec"]["release"])
                sys.path[:0] = [
                    str(release / "src"),
                    str(release / "seed202"),
                    *[str(science / p) for p in ("operator", "src", "tools", "operator/inherited")],
                ]
                from mqar_final_tail_package import bind_wheel

                with tempfile.TemporaryDirectory(
                    prefix="unified-replay-", dir=os.environ["TMPDIR"]
                ) as work:
                    bind_wheel(
                        science / "compare_attention-0.1.0-py3-none-any.whl", Path(work) / "wheel"
                    )
                    import torch

                    torch.nn.Module.__call__ = prohibited
                    torch.Tensor.backward = prohibited
                    torch.autograd.backward = prohibited
                    torch.optim.Optimizer.__init__ = prohibited
                    torch.set_rng_state = prohibited
                    torch.cuda.set_rng_state = prohibited
                    torch.cuda.set_rng_state_all = prohibited
                    from mqar_final_tail_next import replay as genuine_replay
                    from mqar_final_tail_next import review_transition

                    review_transition(
                        read(spec["transition_review"]),
                        digest(row["path"]),
                        digest(release / "manifest.json"),
                    )
                    result = genuine_replay(spec["spec"])
            elif policy == Policy.COMPLETED_NORMAL:
                from completed_normal_publication import replay_completed

                require(case_id == "mqar-k64-deltanet-seed303", "UNIFIED_COMPLETED_CASE3_ONLY")
                result = replay_completed(spec, read(row["path"]), value)
            else:
                require("profile" in spec, "UNIFIED_ORDINARY_PROFILE_REQUIRED")
                if spec["profile"] in PROFILES:
                    from unified_normal_ordinary import replay as replay_normal

                    require(
                        read(row["path"])["case"]["id"] == case_id, "UNIFIED_ORDINARY_CASE_JOIN"
                    )
                    result = replay_normal(spec, read(row["path"]), value)
                else:
                    require(
                        spec == {"profile": "LEGACY_SEED202_ORDINARY_V2"},
                        "UNIFIED_ORDINARY_PROFILE_REQUIRED",
                    )
                    from seed202_pilot.state import replay_acceptance

                    result = replay_acceptance(Path(row["path"]), case_id)
    require(canonical(result) == canonical(read(row["path"])), "UNIFIED_WHOLE_ACCEPTANCE_EQUALITY")
    if policy == Policy.COMPLETED_NORMAL:
        require(result["next_launch_authorized"] is False, "UNIFIED_PRESERVE_AUTHORITY")
    elif policy != Policy.ORDINARY:
        require(result["later_pilots_authorized"] is False, "UNIFIED_PRESERVE_AUTHORITY")
    return VerifiedPredecessor(
        case_id,
        policy,
        digest(row["path"]),
        hashlib.sha256(canonical(result).encode()).hexdigest(),
        digest(path),
    ).serialized()


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args(argv)
    try:
        print(canonical(replay(args.request, args.case)))
        return 0
    except Exception as error:  # pylint: disable=broad-exception-caught
        import traceback

        print(
            canonical(
                {
                    "status": "REJECTED",
                    "class": type(error).__name__,
                    "frames": [
                        {"file": Path(f.filename).name, "function": f.name, "line": f.lineno}
                        for f in traceback.extract_tb(error.__traceback__)
                    ],
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
