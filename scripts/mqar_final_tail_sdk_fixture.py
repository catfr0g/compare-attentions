"""Exercise installed pinned Kaggle SDK without package auto-authentication."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path


def read_response(transport, method, request, response_type):
    """Typed responses below the genuine status/list/quota SDK methods."""
    import datetime

    from kagglesdk.datasets.types import dataset_api_service as datasets
    from kagglesdk.kernels.types import kernels_api_service as kernels

    if method == "GetDatasetStatus":
        ref = request.owner_slug + "/" + request.dataset_slug
        if ref not in transport.datasets:
            transport.http_error(404)
        result = response_type()
        result.status = datasets.DatabundleVersionStatus.READY
        return result
    if method == "ListDatasetFiles":
        assert request.dataset_version_number == 1 and request.page_size == 100
        ref = request.owner_slug + "/" + request.dataset_slug
        result = response_type()
        result.dataset_files = []
        for name, raw in transport.datasets[ref].items():
            row = datasets.ApiDatasetFile()
            row.name, row.total_bytes = name, len(raw)
            result.dataset_files.append(row)
        return result
    if method == "GetKernelSessionStatus":
        ref = request.user_name + "/" + request.kernel_slug
        result = response_type()
        result.status = kernels.KernelWorkerStatus[transport.kernels[ref].status]
        return result
    if method == "GetAcceleratorQuotaStatistics":
        quota = kernels.ApiAcceleratorQuota()
        quota.total_time_allowed = datetime.timedelta(seconds=108000)
        quota.time_used = datetime.timedelta(seconds=10556.615)
        quota.time_reserved = datetime.timedelta(0)
        result = response_type()
        result.gpu_quota = quota
        return result
    raise AssertionError(method)


def sdk_api(monkeypatch, transport):
    """Load genuine SDK module; bypass only package auth and HTTP/blob transport."""
    assert importlib.metadata.version("kaggle") == "2.2.5"
    package = Path(importlib.metadata.distribution("kaggle").locate_file("kaggle"))
    stub = types.ModuleType("kaggle")
    stub.__path__ = [str(package)]
    stub.__version__ = "2.2.5"
    monkeypatch.setitem(sys.modules, "kaggle", stub)
    api_package = types.ModuleType("kaggle.api")
    api_package.__path__ = [str(package / "api")]
    monkeypatch.setitem(sys.modules, "kaggle.api", api_package)
    spec = importlib.util.spec_from_file_location(
        "kaggle.api.kaggle_api_extended", package / "api/kaggle_api_extended.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Isolate the SDK's persistent upload state between remote-world fixtures.
    # A lost ACK deliberately retains SDK state; another fixture is NOT that world.
    # Keep genuine ResumableUploadContext code, including within-test persistence.
    import os
    import tempfile

    cache = tempfile.mkdtemp(prefix="seed202-absence-sdk-", dir=os.environ["TMPDIR"])
    scoped_tempfile = types.SimpleNamespace(**vars(module.tempfile))
    scoped_tempfile.gettempdir = lambda: cache
    module.tempfile = scoped_tempfile
    api = module.KaggleApi.__new__(module.KaggleApi)
    for name, value in vars(transport.api).items():
        if name not in {
            "dataset_status",
            "dataset_list_files",
            "kernels_status",
            "quota_view",
            "_kernel_version_label",
        }:
            setattr(api, name, value)
    api.CONFIG_NAME_USER = "username"
    api.config_values["username"] = "zimike"
    api.args = []
    api.with_retry = lambda method, *args, **kwargs: method
    api.upload_complete = transport.upload_complete
    transport.sdk_module = module
    transport.provider.api = transport.api = api
    transport.sdk_failures = []
    for name in ("dataset_create_new", "kernels_push"):
        method = getattr(api, name)

        def traced(*args, _method=method, **kwargs):
            try:
                return _method(*args, **kwargs)
            except Exception as error:
                transport.sdk_failures.append((type(error).__name__, getattr(error, "name", None)))
                raise

        setattr(api, name, traced)
    return api
