"""Локальная страховка запуска: metrics.jsonl и чекпоинт модели рядом с ним."""

# Exact integer checks reject bool; publication is one ordered transaction.
# pylint: disable=unidiomatic-typecheck,too-many-locals,too-many-branches

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path

import torch

from compare_attention.models.gpt2 import GPT2
from compare_attention.train.config import TrainConfig
from compare_attention.train.result import TrainResult


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _temporary(folder: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=folder)
    os.close(descriptor)
    return Path(name)


def _atomic_json(target: Path, value: object) -> None:
    pending = _temporary(target.parent)
    try:
        with pending.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)


def _owned_checkpoint(folder: Path, row: object) -> bool:
    if not isinstance(row, dict) or set(row) != {"step", "path", "sha256"}:
        return False
    step = row["step"]
    if type(step) is not int or step < 0 or row["path"] != f"step_{step:08d}.pt":
        return False
    path = folder / row["path"]
    return (
        not path.is_symlink()
        and path.is_file()
        and isinstance(row["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None
        and _sha256(path) == row["sha256"]
    )


class BackupTracker:
    """Пишет потери по шагам, итог и веса модели в каталог результата."""

    def __init__(self, result_dir: Path) -> None:
        self._result_dir = result_dir

    def validate_resume_destination(self) -> None:
        """Reject a used result sink before replay can append or overwrite evidence."""
        artifacts = ("metrics.jsonl", "run_identity.json", "model.pt", "checkpoints")
        if any((self._result_dir / name).exists() for name in artifacts):
            raise ValueError("Resume requires a new result directory; refusing existing artifacts")

    def start(self, _config: TrainConfig) -> None:
        """Готовим каталог результата."""
        self._result_dir.mkdir(parents=True, exist_ok=True)

    def record_scalar_readback(self, value: dict[str, object]) -> None:
        """Retain the complete unsampled scalar/page response before validation."""
        _atomic_json(self._result_dir / "training-unsampled-readback.json", value)

    def record_failure(self, report: dict[str, object]) -> None:
        """Flush/fsync safe original facts independently of final identity writes."""
        rank = report.get("rank")
        if rank is not None:
            if type(rank) is not int or rank < 0:
                raise ValueError("invalid diagnostic rank")
            _atomic_json(self._result_dir / f"lifecycle-failure-rank-{rank}.json", report)
        if rank is None or rank == 0:
            _atomic_json(self._result_dir / "lifecycle-failure.json", report)

    def record_identity(self, identity: dict[str, object]) -> None:
        """Persist initialization immediately; completion replaces this atomically."""
        target = self._result_dir / "run_identity.json"
        pending = target.with_suffix(".json.tmp")
        pending.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(target)

    # Keep the publication/rollback transaction ordered in one method.
    # pylint: disable-next=too-many-statements
    def save_checkpoint(self, payload: dict, step: int, keep_last: int) -> None:
        """Publish a verified atomic snapshot before pruning authenticated owned files."""
        if type(step) is not int or step < 0 or type(keep_last) is not int or keep_last < 1:
            raise ValueError("checkpoint step/keep_last must be nonnegative/positive integers")
        if not isinstance(payload, dict) or payload.get("step", step) != step:
            raise ValueError("checkpoint payload step mismatch")
        folder = self._result_dir / "checkpoints"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"step_{step:08d}.pt"
        if target.exists():
            raise ValueError("checkpoint step already exists; refusing overwrite")
        owned_path = folder / ".owned.json"
        latest_path = folder / "latest.json"
        if latest_path.exists():
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            if not _owned_checkpoint(folder, latest):
                raise ValueError("existing checkpoint latest manifest is invalid")
            if latest["step"] >= step:
                raise ValueError("checkpoint step must increase beyond latest")
        original_owned = (
            json.loads(owned_path.read_text(encoding="utf-8")) if owned_path.exists() else None
        )
        owned = [row for row in original_owned or [] if _owned_checkpoint(folder, row)]
        if any(row["step"] >= step for row in owned):
            raise ValueError("checkpoint step must increase")
        pending = _temporary(folder)
        published = False
        ownership_published = False
        try:
            with pending.open("wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            loaded = torch.load(pending, weights_only=True, map_location="cpu")
            if not isinstance(loaded, dict) or loaded.get("step", step) != step:
                raise ValueError("serialized checkpoint step mismatch")
            digest = _sha256(pending)
            pending.replace(target)
            if _sha256(target) != digest:
                raise ValueError("published checkpoint checksum mismatch")
            row = {"step": step, "path": target.name, "sha256": digest}
            # Ownership may contain an unpublished file after interruption; never delete it
            # until a later latest.json has been successfully committed.
            _atomic_json(owned_path, [*owned, row])
            ownership_published = True
            _atomic_json(folder / "latest.json", row)
            published = True
        finally:
            pending.unlink(missing_ok=True)
            if not published:
                target.unlink(missing_ok=True)
                if ownership_published:
                    if original_owned is None:
                        owned_path.unlink(missing_ok=True)
                    else:
                        _atomic_json(owned_path, original_owned)
        for stale in sorted([*owned, row], key=lambda item: item["step"], reverse=True)[keep_last:]:
            try:
                if _owned_checkpoint(folder, stale):
                    (folder / stale["path"]).unlink()
            except OSError:
                logging.getLogger(__name__).warning(
                    "checkpoint retention deferred: file unavailable"
                )

    def log_loss(self, loss: float, flops: int) -> None:
        """Каждый шаг обучения отражается строкой в metrics.jsonl."""
        row = {"type": "step", "loss": loss, "flops": flops}
        with (self._result_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_learning_rate(self, learning_rate: float, flops: int) -> None:
        """Persist the LR actually used by a successful scheduled update."""
        row = {"type": "learning_rate", "learning_rate": learning_rate, "flops": flops}
        with (self._result_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_validation(self, loss: float, accuracy: float, flops: int) -> None:
        """Сохраняем промежуточное качество до завершения обучения."""
        row = {"type": "validation", "loss": loss, "accuracy": accuracy, "flops": flops}
        with (self._result_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def finish(self, result: TrainResult, model: GPT2) -> None:
        """Итог запуска и веса модели остаются рядом с метриками."""
        row = {"type": "result", **result.model_dump(mode="json")}
        with (self._result_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        torch.save(model.state_dict(), self._result_dir / "model.pt")
