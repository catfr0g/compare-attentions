"""Exercise the frozen final uploader's real SDK auth headers without network."""

# pylint: disable=import-outside-toplevel
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_final_upload_explicit_token_survives_implicit_auth_disabled(tmp_path, monkeypatch):
    """The actual SDK create_repo request must authenticate, not only fake uploads."""
    import colab_services as services
    import httpx2
    import huggingface_hub.hf_api as sdk
    from huggingface_hub import constants

    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text("{}")
    result = tmp_path / "completed-run"
    result.mkdir()
    for name in ("model.pt", "test-evaluation.json", "colab-native-gate.json"):
        (result / name).write_bytes(b"synthetic verified artifact")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in result.iterdir()}
    (result / "colab-provenance.json").write_text(
        json.dumps(
            {
                "package_manifest_sha256": services.checksum(package / "manifest.json"),
                "result_sha256": hashes,
            }
        )
    )
    values = {key: "synthetic-value" for key in services.KEYS}
    values.update(HF_TOKEN="synthetic-write-token", HF_REPO="owner/private-model")
    monkeypatch.setattr(services, "load_credentials", lambda _: values)
    monkeypatch.setattr(services.os, "environ", {})
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    authorizations = []

    def handle(request):
        assert request.url.path == "/api/repos/create"
        authorizations.append(request.headers.get("authorization"))
        return httpx2.Response(200, json={"url": "https://huggingface.co/owner/private-model"})

    with httpx2.Client(transport=httpx2.MockTransport(handle)) as client:
        monkeypatch.setattr(sdk, "get_session", lambda: client)
        monkeypatch.setattr(
            sdk.HfApi, "model_info", lambda *a, **k: SimpleNamespace(private=True, sha="a" * 40)
        )
        monkeypatch.setattr(
            sdk.HfApi,
            "list_repo_files",
            lambda *a, **k: [f"runs/{result.name}/{p.name}" for p in result.iterdir()],
        )
        monkeypatch.setattr(sdk.HfApi, "upload_folder", lambda *a, **k: None)
        services.upload(
            tmp_path / "private.env",
            result,
            package,
            download=lambda **kwargs: result / Path(kwargs["filename"]).name,
        )
    assert authorizations == ["Bearer synthetic-write-token"]
