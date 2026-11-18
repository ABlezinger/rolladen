import sys
sys.path.insert(0, ".")

import streamlit as st
import datetime
import os
import json

# Mock streamlit session state and secrets
class _S(dict):
    def __getattr__(self, k): return self[k]
    def __getitem__(self, k):
        val = super().__getitem__(k)
        if isinstance(val, dict):
            return _S(val)
        return val

import toml
st.secrets = _S(toml.load(".streamlit/secrets.toml"))

if "new_docs" not in st.session_state:
    st.session_state.new_docs = {}
if "docs_info" not in st.session_state:
    st.session_state.docs_info = {}

st.session_state.doc_list_editor = "Test Agent"

filename = "test_dummy_doc.pdf"
file_path = f"uploads/{filename}"

# Simulate widget interactions
st.session_state.new_docs[filename] = {}
st.session_state.new_docs[filename]["title"] = "Test Title"
st.session_state.new_docs[filename]["doc_class"] = 1
st.session_state.new_docs[filename]["downloadable"] = True

# Simulate the date computation from filemanagement.py
valid_from = datetime.date(2020, 1, 1)
valid_from_int = int(valid_from.strftime("%Y%m%d")) if valid_from else 15000101
st.session_state.new_docs[filename]["valid_from"] = valid_from_int

valid_to = datetime.date(2025, 12, 31)
valid_to_int = int(valid_to.strftime("%Y%m%d")) if valid_to else 99991231
st.session_state.new_docs[filename]["valid_to"] = valid_to_int

# Simulate saving to doc_list.json
from src.document_utils import DocListEntry
from src.rag.vector_store_management import save_doc_list

entry = DocListEntry.from_doc_list_item(
    file_path,
    {
        "title": st.session_state.new_docs[filename].get("title", filename),
        "valid_from": st.session_state.new_docs[filename].get("valid_from"),
        "valid_to": st.session_state.new_docs[filename].get("valid_to"),
        "doc_class": st.session_state.new_docs[filename].get("doc_class"),
        "downloadable": st.session_state.new_docs[filename].get("downloadable"),
        "date_edited": int(datetime.date.today().strftime("%Y%m%d")),
        "edited_by": st.session_state.doc_list_editor,
    },
)

entry_dump = entry.model_dump() if hasattr(entry, "model_dump") else entry.dict()
st.session_state.docs_info[file_path] = {
    key: value for key, value in entry_dump.items() if key != "source"
}

print(f"Created docs_info entry for {filename}:")
print(json.dumps(st.session_state.docs_info[file_path], indent=2))

