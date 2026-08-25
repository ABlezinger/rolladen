from typing import Any

from pydantic import BaseModel, Field


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
    
    class Config:
        extra = "ignore"

    @classmethod
    def from_doc_list_item(cls, source: str, data: dict[str, Any] | None) -> "DocListEntry":
        """Build a validated entry from a doc_list.json key/value pair."""

        payload = dict(data or {})
        payload["source"] = source
        return cls(**payload)

