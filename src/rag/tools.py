import json
import datetime


def _format_validity_date(value):
    if value in (None, "", "unbekannt"):
        return "unbekannt"

    if isinstance(value, datetime.date):
        return value.strftime("%d.%m.%Y")

    if isinstance(value, str):
        value = value.strip()
        if value in ("", "unbekannt"):
            return "unbekannt"
        for fmt in ("%Y%m%d", "%Y-%m-%d", "%d.%m.%Y"):
            try:
                return datetime.datetime.strptime(value, fmt).strftime("%d.%m.%Y")
            except ValueError:
                continue
        return value

    if isinstance(value, int):
        try:
            return datetime.datetime.strptime(str(value), "%Y%m%d").strftime("%d.%m.%Y")
        except ValueError:
            return str(value)

    return str(value)


def get_list_of_available_docs(doc_info=None) -> str:
    if doc_info is None:
        with open("doc_info.json", "r", encoding="utf-8") as f:
            doc_info = json.load(f)

    text = "Zur Verügung stehende Dokumente:\n"

    for doc in doc_info:
        title = doc.get("title", "Kein Titel vorhanden")
        valid_from = _format_validity_date(doc.get("valid_from", "unbekannt"))
        valid_to = _format_validity_date(doc.get("valid_to", "unbekannt"))
        text += f"- {title} (gültig von {valid_from} bis {valid_to})\n"

    return text
