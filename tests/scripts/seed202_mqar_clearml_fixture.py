"""Synthetic task server below actual event-request DTOs and production consumers."""

import os
import shutil
from types import SimpleNamespace


class TaskRecord:
    def __init__(self, identifier, directory):
        self.id, self.directory = identifier, directory
        self.events, self.parameters, self.artifacts = [], {}, {}
        self.status, self.flushes, self.reloads = "in_progress", 0, 0

    def connect(self, value, name="General"):
        from quality_services import clearml_domain

        self.parameters[name] = clearml_domain(value)

    def get_logger(self):
        return self

    def report_scalar(self, title, series, value, iteration):
        self.events.append({"metric": title, "variant": series, "value": value, "iter": iteration})

    def flush(self, *, wait_for_uploads):
        assert wait_for_uploads is True
        self.flushes += 1

    def close(self):
        assert self.flushes > 0
        self.status = "completed"

    def reload(self):
        self.reloads += 1

    def get_status(self):
        return self.status

    def get_parameters_as_dict(self):
        return self.parameters

    def get_last_scalar_metrics(self):
        result = {}
        for event in self.events:
            result.setdefault(event["metric"], {})[event["variant"]] = {"last": event["value"]}
        return result

    def upload_artifact(self, name, *, artifact_object, wait_on_upload):
        assert wait_on_upload is True
        target = self.directory / (self.id + "-uploaded.json")
        shutil.copyfile(artifact_object, target)

        def get_local_copy(*, extract_archive, raise_on_error, force_download):
            assert extract_archive is False and raise_on_error is True and force_download is True
            fresh = self.directory / (self.id + f"-fresh-{self.reloads}.json")
            shutil.copyfile(target, fresh)
            return str(fresh)

        self.artifacts[name] = SimpleNamespace(get_local_copy=get_local_copy)
        return True


class ClearMLFixture:
    def __init__(self, monkeypatch, directory):
        monkeypatch.setenv("CLEARML_CONFIG_FILE", os.devnull)
        monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
        monkeypatch.setenv("CLEARML_DISABLE_VAULT_SUPPORT", "1")
        from clearml import Task

        self.directory = directory
        directory.mkdir()
        self.tasks, self.pages = {}, []

        def create(**kwargs):
            assert kwargs["reuse_last_task_id"] is False
            record = TaskRecord("ENGINEERING_ONLY_" + str(len(self.tasks)), directory)
            self.tasks[record.id] = record
            return record

        monkeypatch.setattr(Task, "init", create)
        monkeypatch.setattr(Task, "get_task", lambda *, task_id: self.tasks[task_id])

        def send(request):
            assert request.order == "asc" and request.batch_size == 10000
            assert request.event_type == "training_stats_scalar"
            events = self.tasks[request.task].events
            offset = int(request.scroll_id or 0)
            page = events[offset : offset + 10000]
            self.pages.append((request.task, offset, len(page)))
            return SimpleNamespace(
                response=SimpleNamespace(
                    events=page, total=len(events), scroll_id=str(offset + len(page))
                )
            )

        monkeypatch.setattr(Task, "_get_default_session", lambda: SimpleNamespace(send=send))
