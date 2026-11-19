"""Tests for src/virtual_folders.py (pure logic, no Streamlit / Chroma needed).

Run with pytest or directly:
    python tests/test_virtual_folders.py
"""

import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import virtual_folders as vf
from src.virtual_folders import FolderError

SKIP = "rs_10-02-2026"


def sample_doc_list():
    return {
        "rs_10-02-2026/RS_2025-12 kpl.pdf": {"title": "RS", "virtual_path": None},
        "rs_10-02-2026/Richtlinien BVRS/TR Montage April 2024.pdf": {"title": "Montage", "virtual_path": None},
        "rs_10-02-2026/Richtlinien BVRS/TR_112_Laeden_Dezember_2019.pdf": {"title": "Läden", "virtual_path": None},
        "rs_10-02-2026/Richtlinien IVRSA/markisentuecher.pdf": {"title": "Tücher", "virtual_path": None},
        "uploads/ASR-A1-7.pdf": {"title": "ASR", "virtual_path": None},
    }


def raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except FolderError as e:
        return str(e)
    raise AssertionError(f"{fn.__name__} did not raise FolderError")


def test_validate_segment_rejects_unsafe_names():
    for bad in ["", "   ", ".", "..", "a/b", "a\\b", "x" * 101, "tab\there", "nl\nx"]:
        raises(vf.validate_segment, bad)
    assert vf.validate_segment("  Richtlinien BVRS  ") == "Richtlinien BVRS"
    # NFD (macOS file names) is normalised to NFC so names compare equal
    assert vf.validate_segment("Rollläden") == "Rollläden"


def test_normalize_path():
    assert vf.normalize_path(None) == ""
    assert vf.normalize_path("") == ""
    assert vf.normalize_path("/a/b/") == "a/b"
    raises(vf.normalize_path, "a//b")
    raises(vf.normalize_path, "a/../b")


def test_default_virtual_path_matches_previous_explorer_layout():
    assert vf.default_virtual_path("rs_10-02-2026/RS_2025-12 kpl.pdf", SKIP) == ""
    assert vf.default_virtual_path("rs_10-02-2026/Richtlinien BVRS/TR.pdf", SKIP) == "Richtlinien BVRS"
    assert vf.default_virtual_path("uploads/ASR-A1-7.pdf", SKIP) == "uploads"
    assert vf.default_virtual_path("/srv/app/uploads/x.pdf", SKIP) == "uploads"
    assert vf.default_virtual_path("document_index", SKIP) == ""
    assert vf.default_virtual_path("data/../x.pdf", None) == "data"  # ".." is dropped


def test_tree_and_folders():
    doc_list = sample_doc_list()
    sources = set(doc_list) | {"document_index"}
    tree = vf.build_tree(doc_list, {"Leer"}, SKIP, sources)
    assert set(tree["folders"]) == {"Richtlinien BVRS", "Richtlinien IVRSA", "uploads", "Leer"}
    assert set(tree["files"]) == {"rs_10-02-2026/RS_2025-12 kpl.pdf", "document_index"}
    assert len(tree["folders"]["Richtlinien BVRS"]["files"]) == 2


def test_same_file_name_in_one_folder_does_not_collide():
    doc_list = {"a/x.pdf": {"virtual_path": "Z"}, "b/x.pdf": {"virtual_path": "Z"}}
    tree = vf.build_tree(doc_list, set())
    assert len(tree["folders"]["Z"]["files"]) == 2


def test_create_folder():
    doc_list = sample_doc_list()
    _, folders = vf.create_folder(doc_list, set(), "", "Neu", SKIP)
    _, folders = vf.create_folder(doc_list, folders, "Neu", "Sub", SKIP)
    assert folders == {"Neu", "Neu/Sub"}
    raises(vf.create_folder, doc_list, folders, "", "Neu", SKIP)  # exists
    raises(vf.create_folder, doc_list, folders, "", "uploads", SKIP)  # exists implicitly
    raises(vf.create_folder, doc_list, folders, "Gibtsnicht", "x", SKIP)  # missing parent
    raises(vf.create_folder, doc_list, folders, "", "..", SKIP)


def test_rename_folder_moves_documents_and_subfolders():
    doc_list = sample_doc_list()
    folders = {"Richtlinien BVRS/Archiv"}
    new_doc_list, new_folders = vf.rename_folder(doc_list, folders, "Richtlinien BVRS", "BVRS", SKIP)
    assert new_doc_list["rs_10-02-2026/Richtlinien BVRS/TR Montage April 2024.pdf"]["virtual_path"] == "BVRS"
    assert new_doc_list["uploads/ASR-A1-7.pdf"]["virtual_path"] is None  # untouched
    assert "BVRS/Archiv" in new_folders and "Richtlinien BVRS/Archiv" not in new_folders
    assert "Richtlinien BVRS" not in vf.existing_folders(new_doc_list, new_folders, SKIP)
    # input was not mutated
    assert doc_list["rs_10-02-2026/Richtlinien BVRS/TR Montage April 2024.pdf"]["virtual_path"] is None


def test_move_folder_into_other_folder():
    doc_list = sample_doc_list()
    new_doc_list, new_folders = vf.move_folder(doc_list, set(), "Richtlinien IVRSA", "Richtlinien BVRS/IVRSA", SKIP)
    assert new_doc_list["rs_10-02-2026/Richtlinien IVRSA/markisentuecher.pdf"]["virtual_path"] == "Richtlinien BVRS/IVRSA"
    assert "Richtlinien BVRS/IVRSA" in vf.existing_folders(new_doc_list, new_folders, SKIP)


def test_move_folder_rejects_invalid_targets():
    doc_list = sample_doc_list()
    folders = {"Richtlinien BVRS/Archiv"}
    raises(vf.move_folder, doc_list, folders, "Richtlinien BVRS", "Richtlinien BVRS/Archiv/x", SKIP)  # into itself
    raises(vf.move_folder, doc_list, folders, "Richtlinien BVRS", "uploads", SKIP)  # exists
    raises(vf.move_folder, doc_list, folders, "", "x", SKIP)  # root
    raises(vf.move_folder, doc_list, folders, "Gibtsnicht", "x", SKIP)  # missing


def test_move_folder_refuses_files_without_doc_list_entry():
    doc_list = sample_doc_list()
    sources = set(doc_list) | {"uploads/orphan.pdf"}
    msg = raises(vf.move_folder, doc_list, set(), "uploads", "Uploads", SKIP, sources)
    assert "orphan.pdf" in msg


def test_delete_folder():
    doc_list = sample_doc_list()
    folders = {"Leer", "Richtlinien BVRS/Archiv"}
    _, new_folders = vf.delete_folder(doc_list, folders, "Leer", SKIP)
    assert "Leer" not in new_folders
    raises(vf.delete_folder, doc_list, folders, "uploads", SKIP)  # has files
    raises(vf.delete_folder, doc_list, folders, "Richtlinien BVRS", SKIP)  # has subfolder
    _, new_folders = vf.delete_folder(doc_list, folders, "Richtlinien BVRS/Archiv", SKIP)
    assert "Richtlinien BVRS" in vf.existing_folders(doc_list, new_folders, SKIP)


def test_move_file_keeps_source_folder():
    doc_list = sample_doc_list()
    new_doc_list, new_folders = vf.move_file(doc_list, set(), "uploads/ASR-A1-7.pdf", "Richtlinien BVRS", SKIP)
    assert new_doc_list["uploads/ASR-A1-7.pdf"]["virtual_path"] == "Richtlinien BVRS"
    assert "uploads" in vf.existing_folders(new_doc_list, new_folders, SKIP)  # now empty, but kept
    _, new_folders = vf.delete_folder(new_doc_list, new_folders, "uploads", SKIP)
    assert "uploads" not in vf.existing_folders(new_doc_list, new_folders, SKIP)
    raises(vf.move_file, doc_list, set(), "document_index", "", SKIP)  # no doc_list entry
    raises(vf.move_file, doc_list, set(), "uploads/ASR-A1-7.pdf", "Gibtsnicht", SKIP)


def test_locked_update_persists_and_backs_up():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "doc_list.json"), "w", encoding="utf-8") as handle:
            json.dump(sample_doc_list(), handle)

        vf.locked_update(tmp, lambda d, f: vf.create_folder(d, f, "", "Neu", SKIP))
        vf.locked_update(tmp, lambda d, f: vf.move_file(d, f, "uploads/ASR-A1-7.pdf", "Neu", SKIP))

        doc_list, folders = vf.load_state(tmp)
        assert doc_list["uploads/ASR-A1-7.pdf"]["virtual_path"] == "Neu"
        assert "Neu" in folders
        assert os.path.exists(os.path.join(tmp, "doc_list.json.bak"))
        assert not [name for name in os.listdir(tmp) if name.startswith(".tmp_")]

        # a failing operation leaves the files untouched
        before = open(os.path.join(tmp, "doc_list.json"), encoding="utf-8").read()
        raises(vf.locked_update, tmp, lambda d, f: vf.delete_folder(d, f, "Neu", SKIP))
        assert open(os.path.join(tmp, "doc_list.json"), encoding="utf-8").read() == before


def test_atomic_write_keeps_file_permissions():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "doc_list.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(sample_doc_list(), handle)
        os.chmod(path, 0o664)
        vf.locked_update(tmp, lambda d, f: vf.move_file(d, f, "uploads/ASR-A1-7.pdf", "", SKIP))
        assert oct(os.stat(path).st_mode & 0o777) == oct(0o664)


def test_locked_update_serialises_concurrent_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "doc_list.json"), "w", encoding="utf-8") as handle:
            json.dump(sample_doc_list(), handle)

        def worker(i):
            vf.locked_update(tmp, lambda d, f: vf.create_folder(d, f, "", f"Ordner {i}", SKIP))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        _, folders = vf.load_state(tmp)
        assert folders == {f"Ordner {i}" for i in range(20)}


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
