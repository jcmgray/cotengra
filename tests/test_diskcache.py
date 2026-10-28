import os
import sys
import threading
import time
import warnings

import pytest

from cotengra.utils import DiskDict


def test_roundtrip(tmp_path):
    d = DiskDict(tmp_path)
    d["a"] = 1
    d[("bc", "def")] = {"x": 2}
    d2 = DiskDict(tmp_path)
    assert d2["a"] == 1
    assert d2[("bc", "def")] == {"x": 2}
    d["a"] = 3
    assert DiskDict(tmp_path)["a"] == 3
    assert sorted(p.name for p in tmp_path.rglob("*")) == ["a", "bc", "def"]


def test_missing(tmp_path):
    d = DiskDict(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(KeyError):
            d["a"]
        assert d.get("a") is None
        assert "a" not in d


@pytest.mark.parametrize("data", [b"", b"\x80\x05garbage"])
def test_unreadable_is_miss(tmp_path, data):
    (tmp_path / "a").write_bytes(data)
    d = DiskDict(tmp_path)
    with pytest.warns(UserWarning), pytest.raises(KeyError):
        d["a"]
    with pytest.warns(UserWarning):
        assert d.get("a") is None


def test_write_failure_warns(tmp_path, monkeypatch):
    def replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", replace)
    d = DiskDict(tmp_path)
    with pytest.warns(UserWarning):
        d["a"] = 1
    assert d["a"] == 1
    assert list(tmp_path.iterdir()) == []


def test_keys_values_items(tmp_path):
    d = DiskDict(tmp_path)
    d["a"] = 1
    d[("bc", "def")] = 2
    (tmp_path / "bc" / ".def.0123.tmp").write_bytes(b"")
    (tmp_path / "bad").write_bytes(b"")
    d2 = DiskDict(tmp_path)
    assert set(d2.keys()) == {"a", ("bc", "def"), "bad"}
    with pytest.warns(UserWarning):
        assert sorted(d2.values()) == [1, 2]
    with pytest.warns(UserWarning):
        assert dict(d2.items()) == {"a": 1, ("bc", "def"): 2}


def test_delitem(tmp_path):
    d = DiskDict(tmp_path)
    d["a"] = 1
    os.remove(tmp_path / "a")
    del d["a"]
    with pytest.raises(KeyError):
        del d["a"]
    DiskDict(tmp_path)["b"] = 2
    del d["b"]
    assert not (tmp_path / "b").exists()


def test_concurrent_read_write(tmp_path):
    value = list(range(1000))
    DiskDict(tmp_path)["a"] = value
    stop = threading.Event()
    write_errors = []
    read_errors = []

    def write():
        d = DiskDict(tmp_path)
        while not stop.is_set():
            try:
                d["a"] = value
            except Exception as e:
                write_errors.append(e)

    def read():
        while not stop.is_set():
            try:
                assert DiskDict(tmp_path)["a"] == value
            except Exception as e:
                read_errors.append(e)

    threads = [threading.Thread(target=f) for f in [write] * 4 + [read] * 4]
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        for t in threads:
            t.start()
        time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join()

    assert not write_errors
    assert not read_errors
    if sys.platform != "win32":  # Windows won't allow replacing open files
        assert not caught_warnings
