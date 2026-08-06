import json

def get_list_of_available_docs() -> str:
    doc_info = json.load(open("doc_info.json", "r"))
    
    text = "Zur Verügung stehende Dokumente:\n"
    
    for doc in doc_info:
        title = doc.get("title", "Kein Titel vorhanden")
        valid_from = doc.get("valid_from", "unbekannt")
        valid_to = doc.get("valid_to", "unbekannt")
        text += f"- {title} (gültig von {valid_from} bis {valid_to})\n"
    
    return text
    