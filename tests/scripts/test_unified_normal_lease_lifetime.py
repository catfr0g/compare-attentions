"""Offline ownership tests for retained immutable-input leases."""

import contextvars
import hashlib
import os
from pathlib import Path

import pytest


@pytest.fixture
def leased_file(tmp_path):
    root = Path(os.environ["UNIFIED_LIFETIME_FIXTURES"]) / tmp_path.name
    root.mkdir(parents=True, exist_ok=False)
    path = root / "source.bin"
    path.write_bytes(b"original bytes")
    return path


def test_copied_context_cannot_keep_digest_after_owner_exit(leased_file):
    from unified_normal_live import held_digest, retained_lease

    expected = hashlib.sha256(leased_file.read_bytes()).hexdigest()
    with retained_lease([leased_file]):
        assert held_digest(leased_file) == expected
        copied = contextvars.copy_context()
    assert held_digest(leased_file) is None
    assert copied.run(held_digest, leased_file) is None


def test_copied_context_does_not_borrow_digest_in_another_thread(leased_file):
    from concurrent.futures import ThreadPoolExecutor

    from unified_normal_live import held_digest, retained_lease

    with retained_lease([leased_file]):
        copied = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            observed = pool.submit(copied.run, held_digest, leased_file).result(timeout=10)
        assert observed is None


def test_copied_context_is_invalidated_on_exception_exit(leased_file):
    from unified_normal_live import held_digest, retained_lease

    with pytest.raises(RuntimeError, match="^synthetic owner exit$"):
        with retained_lease([leased_file]):
            copied = contextvars.copy_context()
            raise RuntimeError("synthetic owner exit")
    assert copied.run(held_digest, leased_file) is None
    leased_file.write_bytes(b"changed bytes!")


def test_released_copy_cannot_accept_changed_mapped_bytes(leased_file):
    import mmap

    from seed202_pilot.prepared_files import identity
    from seed202_pilot.storage import digest
    from unified_normal_contract import verify_files
    from unified_normal_live import held_digest, retained_lease

    original = hashlib.sha256(leased_file.read_bytes()).hexdigest()
    expected = {
        "files": {str(leased_file): {"bytes": leased_file.stat().st_size, "sha256": original}},
        "roots": [str(leased_file.parent)],
    }
    with retained_lease([leased_file]):
        before = identity(leased_file)
        copied = contextvars.copy_context()
    with leased_file.open("r+b") as stream:
        view = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_WRITE)
    try:
        view[:] = b"changed bytes!"
        view.flush()
        actual = hashlib.sha256(leased_file.read_bytes()).hexdigest()
        assert actual != original
        assert identity(leased_file) == before
        assert held_digest(leased_file) is None
        assert digest(leased_file) == actual
        accepted = []

        def consume():
            with retained_lease([leased_file]) as selected:
                verify_files(expected, selected)
                accepted.append(True)

        with pytest.raises(ValueError, match="^CACHE_WRITE_DENIAL_UNAVAILABLE$"):
            copied.run(consume)
        assert not accepted
        assert copied.run(held_digest, leased_file) is None
    finally:
        view.close()


def test_nested_current_owner_reuses_actual_lease(leased_file, monkeypatch):
    import contextlib

    from seed202_pilot import prepared_files
    from unified_normal_live import held_digest, retained_lease

    actual_lease = prepared_files.lease
    acquisitions = []

    @contextlib.contextmanager
    def counted(paths):
        acquisitions.append(tuple(paths))
        with actual_lease(paths) as pins:
            yield pins

    monkeypatch.setattr(prepared_files, "lease", counted)
    expected = hashlib.sha256(leased_file.read_bytes()).hexdigest()
    with retained_lease([leased_file]):
        with retained_lease([leased_file]):
            assert held_digest(leased_file) == expected
            with pytest.raises(PermissionError):
                leased_file.write_bytes(b"changed bytes!")
        assert held_digest(leased_file) == expected
    assert len(acquisitions) == 1
    assert held_digest(leased_file) is None


def test_foreign_context_acquires_own_handle_lifetime(leased_file, monkeypatch):
    import contextlib
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from seed202_pilot import prepared_files
    from unified_normal_live import held_digest, retained_lease

    actual_lease = prepared_files.lease
    acquisitions = []
    ready = threading.Event()
    finish = threading.Event()
    expected = hashlib.sha256(leased_file.read_bytes()).hexdigest()

    @contextlib.contextmanager
    def counted(paths):
        acquisitions.append(threading.get_ident())
        with actual_lease(paths) as pins:
            yield pins

    monkeypatch.setattr(prepared_files, "lease", counted)

    def worker():
        with retained_lease([leased_file]):
            ready.set()
            assert finish.wait(timeout=10)
            assert held_digest(leased_file) == expected
        assert held_digest(leased_file) is None

    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            with retained_lease([leased_file]):
                copied = contextvars.copy_context()
                future = pool.submit(copied.run, worker)
                assert ready.wait(timeout=10)
            with pytest.raises(PermissionError):
                leased_file.write_bytes(b"changed bytes!")
        finally:
            finish.set()
        future.result(timeout=10)
    assert len(acquisitions) == 2
    assert len(set(acquisitions)) == 2
    leased_file.write_bytes(b"changed bytes!")
