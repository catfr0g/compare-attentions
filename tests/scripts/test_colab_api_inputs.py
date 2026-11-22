"""Private inputs are files, never executable credential literals."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def module():
    assert importlib.util.find_spec("colab_api_inputs"), "inputs helper missing"
    import colab_api_inputs

    return colab_api_inputs


def test_render_allowlisted_literal_dotenv_roundtrip(tmp_path):
    inputs = module()
    from colab_services import KEYS, load_credentials

    values = dict.fromkeys(KEYS, "literal$VALUE${NO_EXPAND}\\tail\"quote'ки")
    values.update({key: "https://example.test" for key in KEYS if key.endswith("HOST")})
    values["HF_REPO"] = "fixture/private-model"
    data = inputs.render_env({**values, "UNAPPROVED": "DO_NOT_SEND"})
    path = tmp_path / "private.env"
    path.write_bytes(data)
    assert load_credentials(path) == values
    assert "UNAPPROVED" not in data.decode()
    assert {line.partition("=")[0] for line in data.decode().splitlines()} == set(KEYS)
    assert all(
        json.loads(line.partition("=")[2]) == values[line.partition("=")[0]]
        for line in data.decode().splitlines()
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://x.google.com/",
        "https://google.com.evil.test/",
        "https://user@x.googleusercontent.com/",
        "https://x.googleusercontent.com:444/",
        "https://x.googleusercontent.com/?token=secret",
        "https://x.googleusercontent.com/#secret",
    ],
)
def test_proxy_rejects_untrusted_endpoint(url):
    with pytest.raises(ValueError, match="INVALID_PROXY"):
        module().Contents({"url": url, "token": "proxy-fixture"})


def test_private_file_contents_put_readback_and_marker_mapping():
    import base64
    import io
    from urllib.parse import urlsplit

    inputs = module()
    files = {"content/marker": b"nonsecret-nonce"}
    seen = []

    class Response(io.BytesIO):
        status = 200

    def opened(request, timeout):
        assert timeout == 20
        assert request.get_header("Authorization") is None
        assert request.get_header("X-colab-runtime-proxy-token") == "proxy-fixture"
        key = urlsplit(request.full_url).path.removeprefix("/api/contents/")
        seen.append((request.method, request.full_url))
        if request.method == "PUT":
            model = json.loads(request.data)
            assert model["type"] == "file" and model["format"] == "base64"
            files[key] = base64.b64decode(model["content"])
        if key not in files:
            raise OSError("missing")
        return Response(
            json.dumps(
                {
                    "type": "file",
                    "format": "base64",
                    "content": base64.b64encode(files[key]).decode(),
                }
            ).encode()
        )

    client = inputs.Contents(
        {"url": "https://fixture.googleusercontent.com", "token": "proxy-fixture"}, opener=opened
    )
    prefix = client.discover_root("marker", "nonsecret-nonce")
    assert prefix == "content/"
    client.write(prefix + "private/input.env", b"SECRET=fixture-only")
    assert files[prefix + "private/input.env"] == b"SECRET=fixture-only"
    assert not any("SECRET" in url for _, url in seen)


def test_private_hf_publication_pins_atomic_commit_and_bytes(tmp_path):
    from types import SimpleNamespace

    inputs = module()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"trusted-fixture-zip")
    from colab_package import checksum

    sha = checksum(archive)
    calls = []

    class API:
        def __init__(self, *, endpoint, token):
            assert endpoint == "https://huggingface.co" and token == "HF_FIXTURE"

        def model_info(self, repo_id, *, revision):
            calls.append(("info", revision))
            return SimpleNamespace(private=True, sha="a" * 40)

        def list_repo_files(self, repo_id, *, revision, repo_type):
            return []

        def upload_file(self, **kwargs):
            calls.append(("upload", kwargs))
            return SimpleNamespace(oid="b" * 40)

        def get_paths_info(self, repo_id, paths, *, revision, repo_type):
            assert revision == "b" * 40
            return [
                SimpleNamespace(
                    path=paths[0], size=archive.stat().st_size, lfs=SimpleNamespace(sha256=sha)
                )
            ]

    def download(**kwargs):
        assert kwargs["revision"] == "b" * 40 and kwargs["token"] == "HF_FIXTURE"
        return archive

    receipt = inputs.publish_archive(
        archive,
        sha,
        {"HF_TOKEN": "HF_FIXTURE", "HF_REPO": "owner/private"},
        api_factory=API,
        download=download,
    )
    assert receipt == {
        "verified": True,
        "revision": "b" * 40,
        "sha256": sha,
        "bytes": archive.stat().st_size,
        "path": "launch-inputs/" + sha + "/colab-package.zip",
    }
    assert calls[1][0] == "upload" and calls[1][1]["parent_commit"] == "a" * 40
    assert "owner/private" not in json.dumps(receipt)


def test_downloader_is_standalone_path_only_and_strips_redirect_credentials():
    inputs = module()
    receipt = {
        "verified": True,
        "revision": "b" * 40,
        "sha256": "a" * 64,
        "bytes": 123,
        "path": "launch-inputs/" + "a" * 64 + "/colab-package.zip",
    }
    code = inputs.downloader_code(receipt)
    compile(code, "<download>", "exec")
    assert "HF_FIXTURE" not in code and "owner/private" not in code
    assert "kaggle-env.env" in code
    assert "Authorization" in code and "redirect_request" in code
    assert "hashlib" in code and "EXPECTED_SHA256" in code
