"""Virtual folder structure for the file explorer.

Folders exist only as metadata: every doc_list.json entry carries a ``virtual_path``
(e.g. ``"Richtlinien/BVRS"``, ``""`` = root), and empty folders are listed in
``folders.json`` next to doc_list.json. Moving or renaming therefore rewrites JSON only.
Physical files, their ``source`` paths and the vector store chunks are never touched, so
the RAG pipeline, re-embedding de-duplication and chat downloads keep working unchanged.

Entries whose ``virtual_path`` is ``None`` fall back to a folder derived from their
physical path (same rules the explorer used before), so existing data needs no migration.

All operations are pure functions on ``(doc_list, folders)`` that return new values.
``locked_update`` applies one of them to the files on disk under an exclusive lock.
"""

from __future__ import annotations

import copy
import json
import os
import unicodedata
from pathlib import PurePath
from typing import Any, Callable, Iterable

from src.document_utils import (
    DOC_LIST_FILENAME,
    FOLDERS_FILENAME,
    atomic_write_json,
    doc_list_lock,
    load_doc_list,
    save_doc_list,
)

ROOT = ""
MAX_SEGMENT_LENGTH = 100
MAX_DEPTH = 20

DocList = dict[str, dict[str, Any]]


class FolderError(ValueError):
    """Invalid folder operation. The message is shown to the user as is (German UI)."""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def validate_segment(name: str) -> str:
    """Validate a single folder name and return it normalised (trimmed, NFC)."""

    name = unicodedata.normalize("NFC", str(name or "")).strip()
    if not name:
        raise FolderError("Der Ordnername darf nicht leer sein.")
    if len(name) > MAX_SEGMENT_LENGTH:
        raise FolderError(f"Der Ordnername darf höchstens {MAX_SEGMENT_LENGTH} Zeichen lang sein.")
    if name in (".", ".."):
        raise FolderError("'.' und '..' sind als Ordnername nicht erlaubt.")
    if "/" in name or "\\" in name:
        raise FolderError("Der Ordnername darf keine Schrägstriche ('/' oder '\\') enthalten.")
    if any(unicodedata.category(char).startswith("C") for char in name):
        raise FolderError("Der Ordnername enthält unzulässige Steuerzeichen.")
    return name


def normalize_path(path: str | None) -> str:
    """Normalise a virtual folder path ("a/b"). Raises FolderError for invalid segments."""

    if path is None:
        return ROOT
    path = str(path).strip().strip("/")
    if not path:
        return ROOT
    segments = [validate_segment(segment) for segment in path.split("/")]
    if len(segments) > MAX_DEPTH:
        raise FolderError(f"Ordner dürfen höchstens {MAX_DEPTH} Ebenen tief verschachtelt sein.")
    return "/".join(segments)


def join_path(parent: str, name: str) -> str:
    name = validate_segment(name)
    return f"{parent}/{name}" if parent else name


def parent_of(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ROOT


def basename_of(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def split_path(path: str) -> list[str]:
    return path.split("/") if path else []


def is_within(path: str, ancestor: str) -> bool:
    """True if ``path`` equals ``ancestor`` or lies below it (root contains everything)."""

    return ancestor == ROOT or path == ancestor or path.startswith(ancestor + "/")


def _ancestors(path: str) -> list[str]:
    """All non-root folders on the way to ``path``, including ``path`` itself."""

    segments = split_path(path)
    return ["/".join(segments[: i + 1]) for i in range(len(segments))]


def default_virtual_path(source: str, skip_prefix: str | None = None) -> str:
    """Folder derived from the physical path (the explorer's behaviour before virtual folders).

    Everything up to and including ``skip_prefix`` is dropped, ``uploads`` paths start at
    ``uploads``; the result is the parent folder. Segments that are not valid virtual folder
    names are skipped or cleaned up instead of raising.
    """

    path_str = str(source)
    if skip_prefix and skip_prefix in path_str:
        relative_path = path_str.split(skip_prefix + "/", 1)[-1]
    elif "uploads" in path_str:
        parts = PurePath(path_str).parts
        relative_path = str(PurePath(*parts[parts.index("uploads"):])) if "uploads" in parts else path_str
    else:
        relative_path = path_str

    segments = []
    for part in PurePath(relative_path).parts[:-1]:
        part = part.replace("\\", "_")
        try:
            segments.append(validate_segment(part[:MAX_SEGMENT_LENGTH]))
        except FolderError:
            continue  # e.g. "/" of an absolute path, ".", ".."
    return "/".join(segments[:MAX_DEPTH])


def resolve_virtual_path(source: str, doc_list: DocList, skip_prefix: str | None = None) -> str:
    """Effective folder of a document: stored ``virtual_path`` or the derived default."""

    stored = (doc_list.get(source) or {}).get("virtual_path")
    if stored is not None:
        try:
            return normalize_path(stored)
        except FolderError:
            pass  # corrupted value: fall back instead of hiding the file
    return default_virtual_path(source, skip_prefix)


# ---------------------------------------------------------------------------
# Folder view
# ---------------------------------------------------------------------------

def _visible_sources(doc_list: DocList, sources: Iterable[str] | None) -> set[str]:
    return set(doc_list) if sources is None else set(sources)


def existing_folders(
    doc_list: DocList,
    folders: Iterable[str],
    skip_prefix: str | None = None,
    sources: Iterable[str] | None = None,
) -> set[str]:
    """All non-root folders: folders of visible documents, explicit folders and their parents."""

    result: set[str] = set()
    for source in _visible_sources(doc_list, sources):
        result.update(_ancestors(resolve_virtual_path(source, doc_list, skip_prefix)))
    for folder in folders:
        result.update(_ancestors(folder))
    return result


def build_tree(
    doc_list: DocList,
    folders: Iterable[str],
    skip_prefix: str | None = None,
    sources: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Nested ``{"folders": {name: node}, "files": {source: name}}`` tree for the explorer.

    Files are keyed by their unique ``source`` so equal file names in one folder don't
    collide, and folder names live in their own dict so they can't clash with files.
    """

    def new_node() -> dict[str, Any]:
        return {"folders": {}, "files": {}}

    def node_for(tree: dict[str, Any], path: str) -> dict[str, Any]:
        node = tree
        for segment in split_path(path):
            node = node["folders"].setdefault(segment, new_node())
        return node

    tree = new_node()
    for folder in existing_folders(doc_list, folders, skip_prefix, sources):
        node_for(tree, folder)
    for source in _visible_sources(doc_list, sources):
        folder = resolve_virtual_path(source, doc_list, skip_prefix)
        node_for(tree, folder)["files"][source] = PurePath(str(source)).name or str(source)
    return tree


# ---------------------------------------------------------------------------
# Operations (pure: return new doc_list / folders, never mutate the arguments)
# ---------------------------------------------------------------------------

def _require_folder(path: str, existing: set[str]) -> None:
    if path == ROOT:
        raise FolderError("Der Hauptordner kann nicht verändert werden.")
    if path not in existing:
        raise FolderError(f"Der Ordner '{path}' existiert nicht (mehr).")


def create_folder(
    doc_list: DocList,
    folders: set[str],
    parent: str,
    name: str,
    skip_prefix: str | None = None,
    sources: Iterable[str] | None = None,
) -> tuple[DocList, set[str]]:
    parent = normalize_path(parent)
    path = join_path(parent, name)
    existing = existing_folders(doc_list, folders, skip_prefix, sources)
    if parent != ROOT and parent not in existing:
        raise FolderError(f"Der übergeordnete Ordner '{parent}' existiert nicht (mehr).")
    if path in existing:
        raise FolderError(f"Der Ordner '{path}' existiert bereits.")
    if len(split_path(path)) > MAX_DEPTH:
        raise FolderError(f"Ordner dürfen höchstens {MAX_DEPTH} Ebenen tief verschachtelt sein.")
    return doc_list, set(folders) | {path}


def move_folder(
    doc_list: DocList,
    folders: set[str],
    old_path: str,
    new_path: str,
    skip_prefix: str | None = None,
    sources: Iterable[str] | None = None,
) -> tuple[DocList, set[str]]:
    """Move or rename a folder including its subfolders and documents."""

    old_path = normalize_path(old_path)
    new_path = normalize_path(new_path)
    visible = _visible_sources(doc_list, sources)
    existing = existing_folders(doc_list, folders, skip_prefix, visible)
    _require_folder(old_path, existing)
    if new_path == old_path:
        return doc_list, set(folders)
    if new_path == ROOT:
        raise FolderError("Ein Ordner kann nicht zum Hauptordner werden.")
    if is_within(new_path, old_path):
        raise FolderError("Ein Ordner kann nicht in sich selbst oder einen seiner Unterordner verschoben werden.")
    if new_path in existing:
        raise FolderError(f"Der Ordner '{new_path}' existiert bereits.")
    if len(split_path(new_path)) + max(
        (len(split_path(f)) - len(split_path(old_path)) for f in existing if is_within(f, old_path)), default=0
    ) > MAX_DEPTH:
        raise FolderError(f"Ordner dürfen höchstens {MAX_DEPTH} Ebenen tief verschachtelt sein.")

    # Files shown in the explorer without a doc_list entry (e.g. document_index) cannot carry
    # a virtual_path; moving their folder would leave them behind, so refuse.
    stranded = sorted(
        source for source in visible - set(doc_list)
        if is_within(resolve_virtual_path(source, doc_list, skip_prefix), old_path)
    )
    if stranded:
        raise FolderError(
            "Der Ordner enthält Dateien ohne Eintrag in doc_list.json und kann daher nicht "
            f"verschoben werden: {', '.join(stranded)}"
        )

    def remap(path: str) -> str:
        return new_path + path[len(old_path):]

    new_doc_list = copy.deepcopy(doc_list)
    for source, entry in new_doc_list.items():
        current = resolve_virtual_path(source, doc_list, skip_prefix)
        if is_within(current, old_path):
            entry["virtual_path"] = remap(current)

    new_folders = {remap(f) if is_within(f, old_path) else f for f in folders}
    new_folders.add(new_path)  # keep the moved folder even if it is empty
    if parent_of(old_path) != ROOT:
        new_folders.add(parent_of(old_path))  # like a file system: the old parent stays
    return new_doc_list, new_folders


def rename_folder(
    doc_list: DocList,
    folders: set[str],
    path: str,
    new_name: str,
    skip_prefix: str | None = None,
    sources: Iterable[str] | None = None,
) -> tuple[DocList, set[str]]:
    path = normalize_path(path)
    return move_folder(doc_list, folders, path, join_path(parent_of(path), new_name), skip_prefix, sources)


def delete_folder(
    doc_list: DocList,
    folders: set[str],
    path: str,
    skip_prefix: str | None = None,
    sources: Iterable[str] | None = None,
) -> tuple[DocList, set[str]]:
    """Delete an empty folder (no visible documents, no subfolders)."""

    path = normalize_path(path)
    visible = _visible_sources(doc_list, sources)
    existing = existing_folders(doc_list, folders, skip_prefix, visible)
    _require_folder(path, existing)
    if any(f != path and is_within(f, path) for f in existing):
        raise FolderError("Der Ordner enthält Unterordner. Bitte zuerst die Unterordner löschen oder verschieben.")
    if any(resolve_virtual_path(source, doc_list, skip_prefix) == path for source in visible):
        raise FolderError("Der Ordner ist nicht leer. Bitte zuerst alle Dateien verschieben oder löschen.")

    new_folders = set(folders) - {path}
    if parent_of(path) != ROOT:
        new_folders.add(parent_of(path))  # deleting a subfolder keeps its parent
    return doc_list, new_folders


def move_file(
    doc_list: DocList,
    folders: set[str],
    source: str,
    target: str,
    skip_prefix: str | None = None,
    sources: Iterable[str] | None = None,
) -> tuple[DocList, set[str]]:
    if source not in doc_list:
        raise FolderError(f"Für '{source}' gibt es keinen Eintrag in doc_list.json; die Datei kann nicht verschoben werden.")
    target = normalize_path(target)
    if target != ROOT and target not in existing_folders(doc_list, folders, skip_prefix, sources):
        raise FolderError(f"Der Zielordner '{target}' existiert nicht (mehr).")

    current = resolve_virtual_path(source, doc_list, skip_prefix)
    new_doc_list = copy.deepcopy(doc_list)
    new_doc_list[source]["virtual_path"] = target
    new_folders = set(folders)
    if current != ROOT:
        new_folders.add(current)  # the source folder stays, even when it becomes empty
    return new_doc_list, new_folders


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_folders(persist_directory: str) -> set[str]:
    """Read folders.json; invalid entries are dropped instead of breaking the explorer."""

    path = os.path.join(persist_directory, FOLDERS_FILENAME)
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    result = set()
    for entry in raw if isinstance(raw, list) else []:
        try:
            normalized = normalize_path(entry)
        except FolderError:
            continue
        if normalized:
            result.add(normalized)
    return result


def save_folders(persist_directory: str, folders: Iterable[str]) -> None:
    atomic_write_json(os.path.join(persist_directory, FOLDERS_FILENAME), sorted(set(folders)))


def load_state(persist_directory: str) -> tuple[DocList, set[str]]:
    doc_list_path = os.path.join(persist_directory, DOC_LIST_FILENAME)
    doc_list = load_doc_list(doc_list_path) if os.path.exists(doc_list_path) else {}
    return doc_list, load_folders(persist_directory)


def locked_update(
    persist_directory: str,
    operation: Callable[[DocList, set[str]], tuple[DocList, set[str]]],
) -> tuple[DocList, set[str]]:
    """Apply ``operation`` to the current on-disk state under an exclusive lock.

    The state is re-read inside the lock, so the operation always works on the latest data
    even if another session changed it meanwhile. doc_list.json is backed up to .bak first.
    """

    with doc_list_lock(persist_directory):
        doc_list, folders = load_state(persist_directory)
        new_doc_list, new_folders = operation(doc_list, folders)
        if new_doc_list != doc_list:
            save_doc_list(os.path.join(persist_directory, DOC_LIST_FILENAME), new_doc_list, backup=True)
        if new_folders != folders:
            save_folders(persist_directory, new_folders)
    return new_doc_list, new_folders
