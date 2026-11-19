"""Engineering-only transport fixture for the actual new NORMAL host program."""

import json
import logging
import os
import runpy
import sys
from pathlib import Path

assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normal_training_http_fixture import NormalHTTP  # noqa: E402
from pytest import MonkeyPatch  # noqa: E402

module, *arguments = sys.argv[1:]
plan = Path(arguments[arguments.index("--plan") + 1])
assert json.loads(plan.read_bytes())["engineering_only"] is True
private = Path(arguments[arguments.index("--private") + 1])
fixture = NormalHTTP(MonkeyPatch(), private.parent / "http")
fixture.install()
sys.argv = [module, *arguments]
sys.path.insert(0, str(Path(module).resolve().parent))
try:
    runpy.run_path(module, run_name="__main__")
finally:
    fixture.save()
