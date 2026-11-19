import fcntl
import json
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from typing import Any, Iterator

from pydantic import BaseModel, Field

DOC_LIST_FILENAME = "doc_list.json"
FOLDERS_FILENAME = "folders.json"
_LOCK_FILENAME = ".doc_list.lock"


DOCUMENT_CLASSES = {
    "Klasse 1": 1,
    "Klasse 2": 2,
    "Klasse 3": 3,
}

def get_doc_class_info_text() -> str:
    """Get a Description of the Document classes.

    Returns:
        str: Document Calss Description text as Markdown.
    """

    return (
        "### Dokumentenklassen\n"
        "- **Klasse 1**: Regelwerke, Normen und Verbandsrichtlinien -- Dokumente mit Wahrheitsanspruch \n"
        "- **Klasse 2**: Fallbeispiele, Praxisberichte aus Verbandsarbeit -- Dokumente mit Auslegungsspielraum\n"
        "- **Klasse 3**: Zeitungsartikel, Werbung, Vorgangsberichte Produktvorstellungen...\n"
    )

class DocListEntry(BaseModel):
    """Validated metadata entry from doc_list.json."""

    source: str
    title: str = Field(default="No title")
    valid_from: int = Field(default=15000101, ge=0)
    valid_to: int = Field(default=99991231, ge=0)
    downloadable: bool = Field(default=False)
    date_edited: int | None = Field(default=None, ge=0)
    edited_by: str | None = Field(default=None)
    doc_class: int = Field(default=3, ge=1, le=3)  # 1: PDF, 2: DOCX, 3: TXT
    # Virtual folder shown in the file explorer ("" = root, None = derive from source path).
    # Purely organisational: the physical file and the vector store are never moved.
    virtual_path: str | None = Field(default=None)

    class Config:
        extra = "ignore"

    @classmethod
    def from_doc_list_item(cls, source: str, data: dict[str, Any] | None) -> "DocListEntry":
        """Build a validated entry from a doc_list.json key/value pair."""

        payload = dict(data or {})
        payload["source"] = source
        return cls(**payload)


def doc_list_entry_to_dict(source: str, data: dict[str, Any] | None) -> dict[str, Any]:
    """Validate a doc_list entry and return the stored JSON value shape."""

    entry = DocListEntry.from_doc_list_item(source, data)
    if hasattr(entry, "model_dump"):
        return entry.model_dump(exclude={"source"})
    return entry.dict(exclude={"source"})


def _current_umask() -> int:
    """Read the process umask without changing it (os.umask() would race between threads)."""

    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError):
        pass
    return 0o022


def atomic_write_json(path: str, data: Any, backup: bool = False) -> None:
    """Write JSON so readers never see a half-written file.

    The data is written to a temp file in the same directory and then swapped in with
    os.replace (atomic on POSIX). With backup=True the previous version is kept as <path>.bak.
    """

    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=4, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        # mkstemp creates 0600 files; keep the permissions a plain open(path, "w") would give
        if os.path.exists(path):
            mode = stat.S_IMODE(os.stat(path).st_mode)
        else:
            mode = 0o666 & ~_current_umask()
        os.chmod(tmp_path, mode)
        if backup and os.path.exists(path):
            shutil.copy2(path, f"{path}.bak")
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_doc_list(doc_list_path: str) -> dict[str, dict[str, Any]]:
    """Load and validate doc_list.json entries."""

    with open(doc_list_path, "r", encoding="utf-8") as handle:
        raw_doc_list = json.load(handle)

    return {
        source: doc_list_entry_to_dict(source, data)
        for source, data in raw_doc_list.items()
    }


def save_doc_list(doc_list_path: str, doc_list: dict[str, dict[str, Any]], backup: bool = False) -> None:
    """Validate doc_list entries and write them to disk atomically."""

    validated_doc_list = {
        source: doc_list_entry_to_dict(source, data)
        for source, data in doc_list.items()
    }
    atomic_write_json(doc_list_path, validated_doc_list, backup=backup)


@contextmanager
def doc_list_lock(persist_directory: str) -> Iterator[None]:
    """Exclusive inter-process lock for read-modify-write cycles on doc_list.json / folders.json.

    Streamlit serves every browser session in its own thread, so two admins editing at the
    same time would otherwise overwrite each other's changes.
    """

    os.makedirs(persist_directory, exist_ok=True)
    with open(os.path.join(persist_directory, _LOCK_FILENAME), "a") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
