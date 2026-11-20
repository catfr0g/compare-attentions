"""Read-only HOST comparison retains full records and both event streams."""

import copy
import importlib.util
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def helper():
    spec = importlib.util.spec_from_file_location(
        "host_view_under_test", SCRIPTS / "unified_normal_host_view.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation():
    rows = [
        dict(iter=0, metric="a", variant="x", value=1, timestamp=1),
        dict(iter=1, metric="a", variant="x", value=2, timestamp=2),
        dict(iter=1, metric="b", variant="x", value=3, timestamp=3),
        dict(iter=2, metric="a", variant="x", value=4, timestamp=4),
    ]
    return dict(
        receipt={"count": 4},
        fresh_sdk_readback=True,
        raw=dict(
            task_id="synthetic",
            status="completed",
            parameters={"x": "1"},
            total=4,
            events=copy.deepcopy(rows),
            pages=[
                dict(events=copy.deepcopy(rows[:2]), returned=2, total=4, scroll_id="cursor1"),
                dict(events=copy.deepcopy(rows[2:]), returned=2, total=4, scroll_id="cursor2"),
            ],
        ),
    )


class EventOrderTests(unittest.TestCase):
    def test_ties_across_page_boundary_are_equal_without_mutation(self):
        left = observation()
        right = copy.deepcopy(left)
        right["raw"]["events"][1:3] = right["raw"]["events"][1:3][::-1]
        right["raw"]["pages"][0]["events"][1], right["raw"]["pages"][1]["events"][0] = (
            right["raw"]["pages"][1]["events"][0],
            right["raw"]["pages"][0]["events"][1],
        )
        before = copy.deepcopy((left, right))
        self.assertEqual(helper().training_view(left), helper().training_view(right))
        self.assertEqual((left, right), before)

    def test_page_only_corruption_and_metadata_changes_are_rejected(self):
        left = observation()
        for field, value in [("value", 99), ("metric", "changed"), ("timestamp", 99)]:
            right = copy.deepcopy(left)
            right["raw"]["pages"][0]["events"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                helper().training_view(right)
        for field, value in [("total", 5), ("returned", 1), ("scroll_id", "changed")]:
            right = copy.deepcopy(left)
            right["raw"]["pages"][0][field] = value
            with self.subTest(field=field):
                try:
                    different = helper().training_view(right) != helper().training_view(left)
                except ValueError:
                    different = True
                self.assertTrue(different)
        for mutation in ("duplicate", "missing", "cursor"):
            right = copy.deepcopy(left)
            if mutation == "duplicate":
                right["raw"]["pages"][1]["events"][0] = copy.deepcopy(
                    right["raw"]["pages"][0]["events"][0]
                )
            elif mutation == "missing":
                right["raw"]["pages"][1]["events"].pop()
            else:
                right["raw"]["pages"][0]["scroll_id"] = ""
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                helper().training_view(right)

    def test_actual_consume_does_not_repeat_lifecycle_or_write_collected_inputs(self):
        import json
        import tempfile
        import types
        from unittest.mock import patch

        sys_path = __import__("sys").path
        sys_path.insert(0, str(SCRIPTS))
        import unified_normal_host as host

        source = {"wheel_sha256": "synthetic", "source_rank_records": 2}
        native = {"final_sha256": "synthetic"}
        training = observation()
        case = {"engineering_only": False}

        def forbidden(*args, **kwargs):
            raise AssertionError("legacy lifecycle writes into immutable input tree")

        acceptance = types.SimpleNamespace(
            accept=forbidden,
            verify_sources=lambda *a: source,
            training_readback=lambda *a: training,
        )
        native_module = types.SimpleNamespace(verify_native=lambda *a: native)
        plan = types.SimpleNamespace(canonical=helper().canonical)
        publications = types.SimpleNamespace(publication_readback=lambda *a: {"publications": []})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "download" / "normal-synthetic"
            root.mkdir(parents=True)
            (root / "native-proof.json").write_text(json.dumps(native))
            (root / "training-fresh-readback.json").write_text(json.dumps(training))
            before = {str(p): p.read_bytes() for p in root.parent.rglob("*") if p.is_file()}
            with (
                patch.dict(
                    "sys.modules",
                    normal_training_acceptance=acceptance,
                    normal_training_native=native_module,
                    normal_training_plan=plan,
                    unified_normal_publications=publications,
                ),
                patch.object(host, "task_quality", return_value={"final_sha256": "synthetic"}),
                patch.object(host, "inference_readback", return_value={}),
            ):
                result = host.consume(
                    "synthetic", root, case, {}, "synthetic", snapshot="synthetic", accept=True
                )
            self.assertEqual(result["status"], "ACCEPTED_UNIFIED_NORMAL")
            self.assertEqual(
                before, {str(p): p.read_bytes() for p in root.parent.rglob("*") if p.is_file()}
            )
        sys_path.remove(str(SCRIPTS))

    def test_public_route_rejects_output_inside_immutable_download(self):
        import sys
        from types import SimpleNamespace

        sys.path.insert(0, str(SCRIPTS))
        import unified_normal_host as host

        args = SimpleNamespace(
            command="accept",
            destination=Path("synthetic/download"),
            index="index",
            output=Path("synthetic/download/receipt.json"),
            private="unused",
            service_snapshot=Path("synthetic/snapshot"),
            task_data="data",
        )
        with self.assertRaisesRegex(ValueError, "UNIFIED_HOST_IMMUTABLE_OUTPUT"):
            host.validate_arguments(args)
        sys.path.remove(str(SCRIPTS))

    def test_next_request_declares_every_ordinary_replay_root(self):
        import json
        import sys
        import tempfile
        from unittest.mock import patch

        sys.path.insert(0, str(SCRIPTS))
        import unified_normal_remaining as builder
        from unified_normal_contract import digest

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            previous = base / "previous"
            payload = previous / "payload"
            result = base / "download/normal-synthetic"
            data, snapshot, normal = (base / name for name in ("data", "snapshot", "normal"))
            required = [
                payload,
                payload / "normal-package",
                payload / "source/src",
                result,
                result.parent,
                data,
                snapshot,
            ]
            for root in [*required, normal]:
                root.mkdir(parents=True, exist_ok=True)
            request = payload / "unified/request.json"
            request.parent.mkdir()
            request.write_text("{}")

            def record(name, value):
                path = base / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
                return str(path)

            runtime = {"execution_runtime": {}}
            for key in reversed(["completed", "0", "acceptance", "timing", "metadata"]):
                runtime = {key: runtime}
            runtime["completed"] = [runtime["completed"]["0"]]
            runtime_path = record("runtime.json", runtime)
            auth = record("permission.json", {})
            row = {
                "case_id": "sorting-classic-seed303",
                "policy": "AUTHENTIC_HOST_NATIVE_V2_FULL_REPLAY",
                "path": record("accepted.json", {"synthetic": True}),
                "sha256": "",
                "replay": {
                    "profile": "REMAINING_NORMAL_FIXED_V1",
                    "source_request": str(request),
                    "result": str(result),
                    "data": str(data),
                    "snapshot": str(snapshot),
                    "index": record("index.json", {}),
                },
            }
            row["sha256"] = digest(row["path"])
            inputs = {
                "source_profile": "REMAINING_NORMAL_FIXED_V1",
                "sorting_candidate": str(base / "sorting"),
                "normal_package": str(normal),
                "normal_case": record("case.json", {}),
                "runtime_record": runtime_path,
                "generation": record("generation.json", {}),
                "normal_plan": record("plan.json", {}),
                "normal_recipe": record("recipe.json", {}),
                "predecessors": [row],
            }
            record("sorting/payload/qualified-launch/bridge.json", {"science": {}})
            generation = {"case_id": "sorting-deltanet-seed303", "authorization": {"path": auth}}
            output = base / "output"

            def prepare(_inputs, target, _runtime):
                path = target / "payload/unified/normal-plan.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps({"cases": {"sorting-deltanet-seed303": {"data_files": {}}}})
                )
                return {"id": "sorting-deltanet-seed303", "config": {"data": str(data)}}

            with (
                patch.object(builder, "authenticate_package"),
                patch.object(builder, "require_frozen_plan"),
                patch("unified_normal_generation.intent", return_value=generation),
                patch("unified_normal_generation.prepare_case", side_effect=prepare),
                patch("unified_normal_source.RUNTIME_RECORD", digest(runtime_path)),
            ):
                builder.build_request(inputs, output)
            actual = json.loads((output / "payload/unified/request.json").read_bytes())
            self.assertTrue(set(map(str, required)).issubset(actual["roots"]))
        sys.path.remove(str(SCRIPTS))

    def test_parameters_receipt_and_group_order_stay_strict(self):
        left = observation()
        for key in ("parameters", "receipt", "groups"):
            right = copy.deepcopy(left)
            if key == "parameters":
                right["raw"]["parameters"]["x"] = "2"
            elif key == "receipt":
                right["receipt"]["count"] = 5
            else:
                right["raw"]["events"].reverse()
                right["raw"]["pages"][0]["events"].reverse()
                right["raw"]["pages"][1]["events"].reverse()
                right["raw"]["pages"].reverse()
            self.assertNotEqual(helper().training_view(left), helper().training_view(right))


if __name__ == "__main__":
    unittest.main()
