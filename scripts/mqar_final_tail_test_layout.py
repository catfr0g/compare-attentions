"""Explicit engineering fixture locations; never deployment authority."""

import json
import os
import sys
from pathlib import Path

assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
BASE = Path("E:/HermesArtifacts/compare-attention/sorting-mqar-seed202-pilot-v1")
ART = BASE / "mqar-final-tail-development01"
LAYOUT = json.loads((ART / "input-layout.json").read_bytes())
INPUT = Path(LAYOUT["input"])
SCIENCE = Path(LAYOUT["science"])
DATA = Path(LAYOUT["data"])
CANDIDATE = Path(os.environ.get("MQAR_FINAL_TAIL_CANDIDATE", str(ART / "candidate-01")))
MOUNT = BASE / "mqar-classic-resume-repair-v1/mount-postreview-real03"
OLD = (
    BASE / "remaining-three-after-tail-v1/qualified-launch-bridge-v1/execution-classic-20261118-v1"
)
old_plan = json.loads((OLD / "plan.json").read_bytes())
ORIGINAL_SOURCES = [
    Path(old_plan["candidate_root"]) / "payload/deploy/mqar-k64-classic-seed202/kernel/kernel.py",
    MOUNT / "payload/deploy/mqar-k64-classic-seed202/kernel/kernel.py",
]
for root in (
    CANDIDATE / "payload/src",
    CANDIDATE / "inherited-two-stage-host",
    *[SCIENCE / p for p in ("operator", "src", "tools", "operator/inherited")],
):
    sys.path.append(str(root))
# Selected original authenticated wheel; no mutable scientific checkout is used in fixtures.
wheel = ART / "wheel"
sys.path.insert(0, str(wheel))
