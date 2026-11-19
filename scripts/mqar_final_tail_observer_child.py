"""Fresh no-network SDK observer child; fixture service bytes supplied explicitly."""

import base64
import json
import os
import sys
from pathlib import Path

assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
sys.dont_write_bytecode = True
release = Path(sys.argv[sys.argv.index("--release") + 1])
sys.path.insert(0, str(release / "src"))
from mqar_final_tail_http_fixture import FixtureHTTP  # noqa: E402
from pytest import MonkeyPatch  # noqa: E402

state_path = Path(os.environ["FIXTURE_SERVICE_STATE"])
state = json.loads(state_path.read_bytes())
http = FixtureHTTP(MonkeyPatch(), state_path.parent)
http.tasks, http.files, http.head = (
    state["tasks"],
    {key: base64.b64decode(raw) for key, raw in state["files"].items()},
    state["head"],
)
http.install()
from mqar_final_tail_service_cli import main  # noqa: E402

code = main(sys.argv[1:])
assert not any("/commit/" in c[1] or c[1].endswith("tasks.create") for c in http.calls)
assert sum("/resolve/" in c[1] for c in http.calls) == len(
    __import__("mqar_final_tail_tracking").FILES
)
if "--hf-reconcile-only" not in sys.argv:
    assert sum(c[1].endswith("events.get_task_events") for c in http.calls) >= 2
else:
    assert not any(c[1].endswith("events.get_task_events") for c in http.calls)
assert all(
    c[0] == "GET"
    or c[1].endswith(("auth.login", "tasks.get_all", "tasks.get_by_id", "events.get_task_events"))
    for c in http.calls
)
print(json.dumps({"scope": "FRESH_SDK_OBSERVER_TRANSPORT_FIXTURE", "requests": len(http.calls)}))
raise SystemExit(code)
