"""Execute generated input validation locally; never pip/train/cloud."""

import ast
import json
import sys
import types

import pytest

from compare_attention.kaggle_series import stage_series
from compare_attention.kaggle_series_kernel import render_series_kernel
from tests.compare_attention.test_kaggle_artifacts import make_inputs
from tests.compare_attention.test_kaggle_series import Runner


@pytest.fixture(name="staged_kernel")
def make_staged_kernel(tmp_path):
    configs, data = make_inputs(tmp_path)
    path = stage_series(tmp_path, configs, data, tmp_path / "cache", "user", Runner())
    state = json.loads(path.read_text())
    name = next(iter(state["runs"]))
    kernel = render_series_kernel(state, name, "user/private-credentials")
    assert kernel.metadata["is_private"] is True
    assert kernel.metadata["dataset_sources"][-1] == "user/private-credentials"
    code = kernel.code.replace(
        "/kaggle/temp/compare-attention-release", (tmp_path / "unpacked").as_posix()
    )
    tree = ast.parse(code, feature_version=(3, 13))
    tree.body.pop()  # main call: do not install dependencies, read credentials, or train.
    namespace = {}
    exec(compile(tree, "kernel.py", "exec"), namespace)  # pylint: disable=exec-used
    return namespace, path, state, name


@pytest.mark.parametrize(
    "case",
    [
        (False, 0, "3.6.0", "float16", False, False),
        (True, 0, "3.6.0", "float16", False, False),
        (True, 1, None, "float16", False, False),
        (True, 1, "3.6.0", "bfloat16", False, False),
        (True, 1, "3.6.0", "float16", False, True),
        (True, 1, "3.6.0", "bfloat16", True, True),
    ],
)
def test_runtime_preflight_allowlisted_and_fail_closed(staged_kernel, tmp_path, monkeypatch, case):
    cuda, count, triton, dtype, native, passed = case
    namespace = staged_kernel[0]
    bf16_calls = []

    def bf16_supported(*, including_emulation):
        bf16_calls.append(including_emulation)
        return native

    fake = types.SimpleNamespace(
        __version__="2.11+cu128" if cuda else "2.11+cpu",
        version=types.SimpleNamespace(cuda="12.8" if cuda else None),
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda,
            device_count=lambda: count,
            current_device=lambda: 0,
            get_device_name=lambda _: "Tesla T4",
            get_device_capability=lambda _: (7, 5),
            is_bf16_supported=bf16_supported,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    namespace["_package_version"] = lambda _: triton
    namespace["_WORK"] = tmp_path
    namespace["_DTYPE"] = dtype
    namespace["_COMPILE"] = True
    if passed:
        namespace["_runtime_preflight"]()
    else:
        with pytest.raises(RuntimeError, match="runtime preflight"):
            namespace["_runtime_preflight"]()
    report = json.loads((tmp_path / "runtime-preflight.json").read_text())
    assert report["passed"] is passed
    assert set(report) == {
        "torch",
        "cuda",
        "cuda_available",
        "device_count",
        "gpu",
        "triton",
        "dtype",
        "compile",
        "native_bf16_supported",
        "passed",
        "failure",
    }
    assert report["gpu"] == (
        {"name": "Tesla T4", "index": 0, "compute_capability": [7, 5]} if cuda and count else None
    )
    assert bf16_calls == ([False] if cuda and count else [])


@pytest.mark.parametrize(
    "accelerator, gpu, capability, cuda, passed",
    [
        ("NvidiaTeslaT4", "NVIDIA A100-SXM4-40GB", (8, 0), True, False),
        ("NvidiaTeslaT4", "Tesla T4", (7, 5), True, True),
        ("NvidiaTeslaT4", "Unknown GPU", (7, 5), True, False),
        ("NvidiaTeslaT4", "Tesla T4", (8, 0), True, False),
        ("NvidiaTeslaT4", "Tesla T4", (7, 5), False, False),
        ("NvidiaL4", "NVIDIA L4", (8, 9), True, True),
        ("NvidiaL4", "Tesla T4", (7, 5), True, False),
        ("NvidiaTeslaA100", "NVIDIA A100-SXM4-40GB", (8, 0), True, True),
        ("NvidiaTeslaA100", "Tesla T4", (7, 5), True, False),
    ],
)
def test_requested_gpu_identity_gates_main(
    staged_kernel, tmp_path, monkeypatch, accelerator, gpu, capability, cuda, passed
):
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    namespace = staged_kernel[0]
    if accelerator != "NvidiaTeslaT4":
        _, _, state, name = staged_kernel
        tree = ast.parse(render_series_kernel(state, name, None, accelerator=accelerator).code)
        tree.body.pop()
        namespace = {}
        exec(compile(tree, "kernel.py", "exec"), namespace)  # pylint: disable=exec-used
    fake = types.SimpleNamespace(
        __version__="2.11+cu128",
        version=types.SimpleNamespace(cuda="12.8"),
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda,
            device_count=lambda: 1 if cuda else 0,
            current_device=lambda: 0,
            get_device_name=lambda _: gpu,
            get_device_capability=lambda _: capability,
            is_bf16_supported=lambda **_kwargs: False,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setenv("HF_TOKEN", "must-not-appear-in-report")
    namespace.update(_WORK=tmp_path, _DTYPE="float16", _COMPILE=True)
    namespace["_package_version"] = lambda _: "3.6.0"
    calls = []
    namespace["_credentials"] = lambda: calls.append("credentials")

    def inputs():
        calls.append("inputs")
        return tmp_path, tmp_path

    namespace["_inputs"] = inputs
    monkeypatch.setattr(namespace["subprocess"], "check_call", lambda _: calls.append("pip"))
    namespace["_train"] = lambda *_args: calls.append("train")
    if passed:
        namespace["main"]()
    else:
        with pytest.raises(RuntimeError, match="runtime preflight"):
            namespace["main"]()
    assert calls == (["credentials", "inputs", "pip", "pip", "train"] if passed else [])
    report_text = (tmp_path / "runtime-preflight.json").read_text()
    assert "must-not-appear-in-report" not in report_text
    report = json.loads(report_text)
    assert report["passed"] is passed
    assert report["gpu"] == (
        {"name": gpu, "index": 0, "compute_capability": list(capability)} if cuda else None
    )
    phases = json.loads((tmp_path / "phase-timings.json").read_text())
    assert phases[0]["phase"] == "runtime_preflight"
    if not passed:
        assert len(phases) == 1


def test_runtime_preflight_precedes_dependencies_and_credentials(staged_kernel):
    namespace = staged_kernel[0]
    calls = []
    namespace["_runtime_preflight"] = lambda: calls.append("preflight")
    namespace["_credentials"] = lambda: calls.append("credentials")
    namespace["_inputs"] = lambda: ("root", "data")

    def phase(name, _action):
        calls.append(name)
        if name == "verify_inputs":
            return "root", "data"
        return None

    namespace["_phase"] = phase
    namespace["main"]()
    assert calls.index("runtime_preflight") < calls.index("credentials")
    assert calls.index("runtime_preflight") < calls.index("install_dependencies")


def test_generated_preflight_uses_immutable_schedule(staged_kernel):
    _, _, state, name = staged_kernel
    state["identity"]["configs"][name]["schedule"].update(dtype="float16", compile=True)
    code = render_series_kernel(state, name, None).code
    tree = ast.parse(code)
    tree.body.pop()
    namespace = {}
    exec(compile(tree, "kernel.py", "exec"), namespace)  # pylint: disable=exec-used
    assert namespace["_DTYPE"] == "float16"
    assert namespace["_COMPILE"] is True


def test_generated_kernel_explicitly_requests_gpu(staged_kernel):
    _, _, state, name = staged_kernel
    kernel = render_series_kernel(state, name, "user/private-credentials")
    assert kernel.metadata.get("enable_gpu") is True


def test_generated_input_validation_real_arrays_and_corruption(staged_kernel):
    namespace, path, state, name = staged_kernel
    namespace["_find"] = lambda filename, dataset: (
        path.parent / ("release" if dataset == state["release_id"] else "data") / filename
    )
    root, mounted_data = namespace["_inputs"]()
    assert mounted_data == path.parent / "data"
    assert (root / name).is_file()
    # Checksummed identity catches corruption even with a valid .npy header/shape.
    with (mounted_data / "train_inputs.npy").open("ab") as stream:
        stream.write(b"broken")
    with pytest.raises(RuntimeError, match="checksum"):
        namespace["_inputs"]()


def test_private_credentials_survive_unavailable_secrets(staged_kernel, tmp_path, monkeypatch):
    namespace = staged_kernel[0]
    env = tmp_path / "kaggle-env.env"
    env.write_text("HF_TOKEN=dummy-test-token\nKAGGLE_KEY=must-not-load\n")
    namespace["_find"] = lambda *_args: env

    def unavailable():
        raise RuntimeError("Kaggle Secrets unavailable")

    monkeypatch.setitem(
        sys.modules, "kaggle_secrets", types.SimpleNamespace(UserSecretsClient=unavailable)
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    namespace["_credentials"]()
    assert namespace["os"].environ["HF_TOKEN"] == "dummy-test-token"
    assert "KAGGLE_KEY" not in namespace["os"].environ
    monkeypatch.delenv("HF_TOKEN")
