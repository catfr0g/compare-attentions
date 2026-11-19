"""Distinct inference task and one immutable FINAL commit; no training finish path."""

import os
import re
from pathlib import Path

from mqar_final_tail_native import FINAL as PARAMETERS_SHA
from mqar_final_tail_native import digest, require

TRAINING_TASK = "17a3c249cedc4f37a358ca99f653ac52"
TERMINAL_REVISION = "ac58dd57dd1d347fc7bf0c2252ef1147c01ae15c"
FILES = {
    "evaluation-intent.json",
    "inference-task.json",
    "mqar-val.json",
    "mqar-val-records.json",
    "no-optimization.json",
    "recovery-provenance.json",
    "source-model-receipt.json",
    "scientific-source.json",
    "historical-hf-proof.json",
    "execution-authority.json",
}


def binding(nonce):
    require(type(nonce) is str and re.fullmatch("[0-9a-f]{32}", nonce), "RECOVERY_NONCE")
    return {
        "nonce": nonce,
        "source_training_task_id": TRAINING_TASK,
        "final_parameters_sha256": PARAMETERS_SHA,
        "training_continuation": False,
        "optimizer_updates_permitted": 0,
        "training_flops_permitted": 0,
        "split": "val",
        "rows": 1000,
        "batch_size": 32,
        "query_targets": 64000,
        "terminal_recovery_revision": TERMINAL_REVISION,
        "original_provider_errors": "145778849:ERROR,145850896:ERROR",
        "original_provider_statuses": "ERROR",
        "original_service_history_complete": True,
        "original_missing_train_events": 0,
        "original_missing_validation_events": 0,
    }


def verify_unsampled_events(value, metrics):
    """Exactly the three quality samples at iteration zero; canonicalize only event order."""
    require(
        type(value) is dict
        and set(value) in ({"events", "total"}, {"events", "total", "unsampled_complete"}),
        "INFERENCE_UNSAMPLED_SCHEMA",
    )
    require(
        "unsampled_complete" not in value or value["unsampled_complete"] is True,
        "INFERENCE_UNSAMPLED_SCHEMA",
    )
    events, total = value["events"], value["total"]
    require(
        type(total) is int and type(events) is list and total == len(events) == len(metrics),
        "INFERENCE_UNSAMPLED_COUNT",
    )
    normalized = []
    for row in events:
        require(
            type(row) is dict
            and row.get("metric") == "quality"
            and row.get("variant") in metrics
            and type(row.get("iter")) is int
            and row["iter"] == 0
            and type(row.get("value")) in (int, float)
            and row["value"] == metrics[row["variant"]],
            "INFERENCE_UNSAMPLED_VALUE",
        )
        normalized.append({k: row[k] for k in ("metric", "variant", "iter", "value")})
    require(
        len({row["variant"] for row in normalized}) == len(metrics), "INFERENCE_UNSAMPLED_DUPLICATE"
    )
    normalized.sort(key=lambda row: list(metrics).index(row["variant"]))
    return {"events": normalized, "total": total, "unsampled_complete": True}


def read_unsampled_events(task_id, metrics):
    from mqar_final_tail_events import read_quality

    return read_quality(task_id, metrics)


def publish_final(folder, prefix, files, transport, *, revision=None):
    """Use the approved pending/unknown journal and exact HEAD-CAS transport unchanged."""
    from quality_recovery import RecoveryPublisher

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "single-writer.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        publisher = RecoveryPublisher(folder / "hf.jsonl", prefix, transport)
        records = publisher.records()
        if not records:
            return publisher.publish(1, files)
        expected = {
            n: {"sha256": digest(p), "bytes": Path(p).stat().st_size}
            for n, p in sorted(files.items())
        }
        require(
            records[-1]["files"] == expected and records[-1]["prefix"] == prefix,
            "FINAL_LOCAL_BYTES_DRIFT",
        )
        if records[-1]["state"] == "verified":
            proof = records[-1]
            require(
                revision is None or revision == proof["revision"], "FINAL_VERIFIED_REVISION_DRIFT"
            )
            actual = transport.read(proof, proof["revision"])
            require(
                set(actual["files"]) == set(files)
                and actual["private"] is True
                and actual["revision"] == proof["revision"],
                "FINAL_GET_READBACK",
            )
            require(
                all(actual["files"][n] == Path(p).read_bytes() for n, p in files.items()),
                "FINAL_GET_BYTES",
            )
            return proof
        require(revision is not None, "FINAL_UNKNOWN_GET_ONLY_EXPLICIT_REVISION_REQUIRED")
        return publisher.reconcile(1, revision)
    finally:
        lock.unlink()


class RecoveryServices:
    """Real SDKs, lazy credential import, no automatic offline fallback."""

    def __init__(self, values, nonce, original, release):
        from quality_inherit import leaf

        self.values, self.nonce = values, nonce
        self.original, self.release = Path(original), Path(release)
        self.bound = binding(nonce)
        self.service = leaf("services").ServiceTracker(
            values, "mqar202-final10780-tail-recovery-" + nonce
        )
        self.task = None
        self.historical_proof = None

    @classmethod
    def from_private(cls, private, nonce, original, release):
        from quality_inherit import leaf

        return cls(leaf("services").credentials(private, rank=0), nonce, original, release)

    def start(self):
        def operation():
            from clearml import Task
            from huggingface_hub import HfApi

            require(
                not Task.get_tasks(
                    project_name="compare-attention", task_name="^" + self.service.name + "$"
                ),
                "REMOTE_INFERENCE_EXISTS_SERVICE_SALVAGE_ONLY",
            )
            api = HfApi(token=self.values["HF_TOKEN"])
            info = api.repo_info(
                self.values["HF_REPO"], repo_type="model", token=self.values["HF_TOKEN"]
            )
            require(info.private is True, "RECOVERY_PRIVATE_REPOSITORY")
            existing = api.list_repo_files(
                self.values["HF_REPO"],
                repo_type="model",
                revision=info.sha,
                token=self.values["HF_TOKEN"],
            )
            prefix = "mqar202-final10780-tail-recovery/" + self.nonce + "/FINAL/"
            require(
                not any(name.startswith(prefix) for name in existing),
                "REMOTE_FINAL_EXISTS_SERVICE_SALVAGE_ONLY",
            )
            self.task = Task.init(
                project_name="compare-attention",
                task_name=self.service.name,
                task_type=Task.TaskTypes.inference,
                reuse_last_task_id=False,
                auto_connect_frameworks=dict.fromkeys(
                    (
                        "matplotlib",
                        "tensorflow",
                        "tensorboard",
                        "pytorch",
                        "xgboost",
                        "scikit",
                        "fastai",
                        "lightgbm",
                        "hydra",
                        "detect_repository",
                        "tfdefines",
                        "joblib",
                        "megengine",
                        "catboost",
                        "gradio",
                    ),
                    False,
                ),
                auto_connect_streams=False,
                auto_connect_arg_parser=False,
                auto_resource_monitoring=False,
            )
            require(self.task.id != TRAINING_TASK, "DISTINCT_INFERENCE_TASK")
            self.task.connect(self.bound, name="recovery-binding")
            self.task.add_tags(
                ["mqar202-final10780-tail-recovery", "inference-only", "no-continuation"]
            )
            return self.task.id

        return self.service.invoke("recovery_start", operation)

    def attach(self, identifier):
        def operation():
            from clearml import Task

            require(
                type(identifier) is str and identifier != TRAINING_TASK, "DISTINCT_INFERENCE_TASK"
            )
            self.task = Task.get_task(task_id=identifier)
            self.verify_binding(self.task)

        self.service.invoke("recovery_attach", operation)

    def verify_binding(self, task):
        from quality_services import clearml_domain

        require(
            task.name == self.service.name and str(task.task_type).split(".")[-1] == "inference",
            "INFERENCE_TASK_IDENTITY",
        )
        require(
            task.get_parameters_as_dict()["recovery-binding"] == clearml_domain(self.bound),
            "INFERENCE_TASK_BINDING",
        )
        require(
            {"mqar202-final10780-tail-recovery", "inference-only", "no-continuation"}
            <= set(task.get_tags()),
            "INFERENCE_TASK_TAGS",
        )

    def complete_tracking(self, output, report, *, read_only=False):
        def operation():
            from clearml import Task

            task = self.task
            self.verify_binding(task)
            status = str(task.get_status()).split(".")[-1].lower()
            require(
                status in {"created", "in_progress", "completed", "stopped"},
                "RECOVERY_TASK_NOT_FINISHABLE",
            )
            from mqar_final_tail_report import quality_metrics

            metrics = quality_metrics(report)
            if read_only:
                require(status == "completed", "GET_ONLY_COMPLETED_RECOVERY_REQUIRED")
            if status != "completed":
                for name, value in metrics.items():
                    task.get_logger().report_scalar(
                        title="quality", series=name, value=value, iteration=0
                    )
                for name in sorted(FILES):
                    require(
                        task.upload_artifact(
                            name, artifact_object=str(output / name), wait_on_upload=True
                        )
                        is True,
                        "RECOVERY_ARTIFACT_UPLOAD",
                    )
                task.flush(wait_for_uploads=True)
                task.close()
            observed = Task.get_task(task_id=task.id)
            observed.reload()
            self.verify_binding(observed)
            require(
                str(observed.get_status()).split(".")[-1].lower() == "completed",
                "RECOVERY_TASK_COMPLETION",
            )
            scalars = observed.get_last_scalar_metrics()
            require(
                all(scalars["quality"][n]["last"] == v for n, v in metrics.items()),
                "RECOVERY_SCALAR_READBACK",
            )
            artifacts = {}
            unsampled = read_unsampled_events(task.id, metrics)
            for name in sorted(FILES):
                downloaded = Path(
                    observed.artifacts[name].get_local_copy(
                        extract_archive=False, raise_on_error=True, force_download=True
                    )
                )
                require(
                    downloaded.read_bytes() == (output / name).read_bytes(),
                    "RECOVERY_ARTIFACT_READBACK",
                )
                artifacts[name] = digest(downloaded)
            return {
                "task_id": task.id,
                "status": "completed",
                "type": "inference",
                "binding": self.bound,
                "metrics": metrics,
                "unsampled": unsampled,
                "artifacts": artifacts,
            }

        return self.service.invoke("recovery_complete", operation)
