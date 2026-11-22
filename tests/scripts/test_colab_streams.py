"""Execute pinned SDK stream classes without importing SDK/config or using a network."""

# pylint: disable=import-outside-toplevel,protected-access
import ast
import importlib
import io
import logging
import sys
import threading
from collections.abc import Callable
from importlib.metadata import distribution
from pathlib import Path
from time import time
from types import SimpleNamespace
from typing import Any, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def sdk_streams():
    """Use both complete, unmodified classes from the installed pinned SDK."""
    sdk = distribution("clearml")
    assert sdk.version == "2.1.12"
    source = sdk.locate_file("clearml/backend_interface/logger.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"StdStreamPatch", "PrintPatchLogger"}
    ]
    assert len(tree.body) == 2
    namespace = {
        "logging": logging,
        "sys": sys,
        "time": time,
        "Optional": Optional,
        "Callable": Callable,
        "Any": Any,
        "ForkSafeRLock": threading.RLock,
        "running_remotely": lambda: False,
        "config": {},
        "DEBUG_SIMULATE_REMOTE_TASK": SimpleNamespace(get=lambda: False),
    }
    # Execute installed classes only: no SDK imports or configuration side effects.
    exec(  # pylint: disable=exec-used
        compile(tree, str(sdk.locate_file("clearml/backend_interface/logger.py")), "exec"),
        namespace,
    )
    return SimpleNamespace(**namespace)


@pytest.fixture(name="stream_session")
def make_stream_session():
    """Isolate every standard stream and SDK class state from pytest/host config."""
    sdk = sdk_streams()
    core = importlib.import_module("colab_services")
    targets = [io.StringIO(), io.StringIO()]
    streams = [core.RedactedOutput(target, ["synthetic-secret"]) for target in targets]
    messages = []
    logger = SimpleNamespace(_console=lambda text, **kwargs: messages.append((text, kwargs)))
    return sdk, streams, targets, messages, logger


def test_real_sdk_attaches_stdout_stderr_and_keeps_console(stream_session, monkeypatch):
    sdk, streams, targets, messages, logger = stream_session
    for name, stream in zip(("stdout", "stderr"), streams, strict=True):
        monkeypatch.setattr(sys, name, stream)
        monkeypatch.setattr(sys, "__" + name + "__", stream)
    sdk.StdStreamPatch.patch_std_streams(logger, load_config_defaults=False)
    assert sdk.StdStreamPatch._stdout_proxy is not None
    assert sdk.StdStreamPatch._stderr_proxy is not None
    sys.stdout.write("metric=1\n")
    sys.stderr.write("ordinary diagnostic\n")
    for stream in streams:
        stream.finish()
    assert targets[0].getvalue() == "metric=1\n"
    assert targets[1].getvalue() == "ordinary diagnostic\n"
    assert [text for text, _ in messages] == ["metric=1", "ordinary diagnostic"]
    assert [meta["level"] for _, meta in messages] == [logging.INFO, logging.ERROR]


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
@pytest.mark.parametrize("failure", [False, True])
def test_train_redacts_before_sdk_capture_and_finishes_pending(
    stream_session, monkeypatch, tmp_path, stream_name, failure
):  # pylint: disable=too-many-locals
    """Raw SDK capture must never see fragments, including CR/flush/exception exit."""
    import os

    core = importlib.import_module("colab_services")
    sdk, _, targets, messages, logger = stream_session
    original_write = sdk.PrintPatchLogger.write
    values = {key: "synthetic-secret" for key in core.KEYS}
    monkeypatch.setattr(core, "load_credentials", lambda _: values)
    monkeypatch.setattr(core, "preflight", lambda: None)
    monkeypatch.setattr(core.os, "environ", dict(os.environ))
    monkeypatch.setitem(sys.modules, "clearml.backend_interface.logger", sdk)
    for name, target in zip(("stdout", "stderr"), targets, strict=True):
        monkeypatch.setattr(sys, name, target)
        monkeypatch.setattr(sys, "__" + name + "__", target)

    def trainer(*_args, **_kwargs):
        assert os.environ["CLEARML_API_SECRET_KEY"] == "synthetic-secret"
        assert sdk.PrintPatchLogger.write is not original_write
        sdk.StdStreamPatch.patch_std_streams(logger, load_config_defaults=False)
        stream = getattr(sys, stream_name)
        stream.write("metric=1\n")
        stream.write("synthetic-")
        stream.flush()
        assert all("synthetic-" not in text for text, _ in messages)
        stream.write("secret\rprogress=2\n")
        stream.write("tail=synthetic-")
        stream.flush()
        stream.write("secret")
        if failure:
            raise RuntimeError("synthetic trainer failure")

    monkeypatch.setattr(core.runpy, "run_module", trainer)
    if failure:
        with pytest.raises(RuntimeError, match="synthetic trainer failure"):
            core.train(tmp_path / "unused.env", ["--tracker", "clearml"])
    else:
        core.train(tmp_path / "unused.env", ["--tracker", "clearml"])
    terminal = "".join(target.getvalue() for target in targets)
    remote = "\n".join(text for text, _ in messages)
    for content in (terminal, remote):
        assert "synthetic-" not in content
        assert "metric=1" in content
        assert "[REDACTED]\rprogress=2" in content
        assert "tail=[REDACTED]" in content
    assert all(meta["omit_console"] for _, meta in messages)
    assert sdk.PrintPatchLogger.write is original_write
