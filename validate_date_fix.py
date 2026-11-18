"""
Validation script for the date metadata fix on RS Auskunft RAG.
Run with:
  sudo -u alexander.blezinger /home/alexander.blezinger/miniconda3/envs/bbs/bin/python3 validate_date_fix.py
from /home/alexander.blezinger/rolladen/
"""

import sys
import os
import json

sys.path.insert(0, "/home/alexander.blezinger/rolladen")
os.chdir("/home/alexander.blezinger/rolladen")

# ── Patch streamlit secrets ──────────────────────────────────────────────────
import streamlit as st
import toml

class _Secrets(dict):
    def __getattr__(self, k): return self[k]
    def __getitem__(self, k):
        val = super().__getitem__(k)
        return _Secrets(val) if isinstance(val, dict) else val

st.secrets = _Secrets(toml.load(".streamlit/secrets.toml"))

# ─────────────────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results = []

def check(label, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((status, label, detail))
    print(f"  {status}  {label}")
    if detail:
        print(f"         {detail}")

print()
print("=" * 65)
print("  RS Auskunft — Date Fix Validation")
print("=" * 65)


# ── TEST 1: doc_list.json correct dates ──────────────────────────────────────
print("\n[1] doc_list.json — date entries")

DOC_LIST_PATH = "rsev_v2/doc_list.json"
with open(DOC_LIST_PATH) as f:
    doc_list = json.load(f)

EXPECTED = {
    "uploads/ZH 1-494 1989-04.pdf":  {"valid_from": 19890401, "valid_to": 20021231},
    "uploads/BGR 232 2003.pdf":       {"valid_from": 20030101, "valid_to": 20091031},
}

for key, expected in EXPECTED.items():
    entry = doc_list.get(key)
    if entry is None:
        check(f"{key} — exists", False, "Key not found in doc_list.json")
        continue
    for field, expected_val in expected.items():
        actual = entry.get(field)
        check(
            f"{os.path.basename(key)} — {field} = {expected_val}",
            actual == expected_val,
            f"Got: {actual}" if actual != expected_val else ""
        )

# ── TEST 2: No "No title" entries with dates set (sanity) ───────────────────
print("\n[2] doc_list.json — no-title entries sanity")
no_title_with_dates = [
    k for k, v in doc_list.items()
    if v.get("title") == "No title" and (v.get("valid_from", 15000101) != 15000101 or v.get("valid_to", 99991231) != 99991231)
]
check(
    "No 'No title' entries have non-default dates",
    len(no_title_with_dates) == 0,
    f"Suspicious entries: {no_title_with_dates}" if no_title_with_dates else ""
)

# ── TEST 3: ChromaDB vector store metadata ───────────────────────────────────
print("\n[3] ChromaDB vector store — metadata for ZH 1/494 and BGR 232")

try:
    from src.rag.vector_store_management import OpenAIEmbeddingsWrapper
    from src.llm_client import get_client
    from langchain_chroma import Chroma

    client = get_client(
        base_url="https://chat-ai.academiccloud.de/v1",
        api_key=st.secrets["KISSKI_API_KEY"]
    )
    embeddings = OpenAIEmbeddingsWrapper(client, st.secrets["embedding_id"])
    vector_store = Chroma(persist_directory="rsev_v2", embedding_function=embeddings)
    stored = vector_store.get()
    metadatas = stored.get("metadatas", []) or []

    # Group by source
    from collections import defaultdict
    by_source = defaultdict(list)
    for m in metadatas:
        if m and m.get("source"):
            by_source[m["source"]].append(m)

    for key, expected in EXPECTED.items():
        chunks = by_source.get(key, [])
        if not chunks:
            check(f"ChromaDB: {os.path.basename(key)} — has chunks", False, "Source not found in vector store")
            continue
        check(f"ChromaDB: {os.path.basename(key)} — has {len(chunks)} chunks", True)

        # All chunks should have consistent metadata
        sample = chunks[0]
        for field, expected_val in expected.items():
            actual = sample.get(field)
            all_match = all(c.get(field) == expected_val for c in chunks)
            check(
                f"ChromaDB: {os.path.basename(key)} — {field} = {expected_val} (all {len(chunks)} chunks)",
                all_match,
                f"Sample got: {actual}" if not all_match else ""
            )

    # Global: count chunks with default fallback dates (should be low - only truly undated docs)
    default_from = [m for m in metadatas if m and m.get("valid_from") == 15000101]
    default_to   = [m for m in metadatas if m and m.get("valid_to")   == 99991231]
    total = len(metadatas)
    print(f"\n  ℹ️  Total chunks in vector store: {total}")
    print(f"  ℹ️  Chunks with valid_from=15000101 (undated): {len(default_from)}")
    print(f"  ℹ️  Chunks with valid_to=99991231   (undated): {len(default_to)}")

except Exception as e:
    check("ChromaDB — load vector store", False, str(e))


# ── TEST 4: Code fix — redundant new_docs lines removed ──────────────────────
print("\n[4] Code fix — filemanagement.py patched correctly")

FM_PATH = "/home/alexander.blezinger/rolladen/src/filemanagement.py"
with open(FM_PATH) as f:
    fm_content = f.read()

bad_pattern_1 = (
    'valid_from_int = int(valid_from.strftime("%Y%m%d")) if valid_from else 15000101\n'
    '                    st.session_state.new_docs[uploaded_file.name] = st.session_state.new_docs.get(uploaded_file.name, {})\n'
    '                    st.session_state.new_docs[uploaded_file.name]["valid_from"] = valid_from_int'
)
bad_pattern_2 = (
    'valid_to_int = int(valid_to.strftime("%Y%m%d")) if valid_to else 99991231\n'
    '                    st.session_state.new_docs[uploaded_file.name] = st.session_state.new_docs.get(uploaded_file.name, {})\n'
    '                    st.session_state.new_docs[uploaded_file.name]["valid_to"] = valid_to_int'
)

check(
    "valid_from redundant reassignment removed",
    bad_pattern_1 not in fm_content,
    "Buggy pattern still present!" if bad_pattern_1 in fm_content else ""
)
check(
    "valid_to redundant reassignment removed",
    bad_pattern_2 not in fm_content,
    "Buggy pattern still present!" if bad_pattern_2 in fm_content else ""
)

# Confirm the correct pattern exists
good_pattern_1 = 'st.session_state.new_docs[uploaded_file.name]["valid_from"] = valid_from_int'
good_pattern_2 = 'st.session_state.new_docs[uploaded_file.name]["valid_to"] = valid_to_int'
check("valid_from assignment line present", good_pattern_1 in fm_content)
check("valid_to assignment line present", good_pattern_2 in fm_content)


# ── TEST 5: Service health ────────────────────────────────────────────────────
print("\n[5] Service health")
import subprocess
r = subprocess.run(["systemctl", "is-active", "rpluss.service"], capture_output=True, text=True)
check("rpluss.service is active", r.stdout.strip() == "active", r.stdout.strip())

import urllib.request
try:
    resp = urllib.request.urlopen("http://10.10.150.58:8501", timeout=5)
    check("HTTP 200 on port 8501", resp.status == 200, f"Got: {resp.status}")
except Exception as e:
    check("HTTP 200 on port 8501", False, str(e))


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 65)
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
print(f"  TOTAL: {passed} passed, {failed} failed out of {len(results)} checks")
print("=" * 65)
print()
if failed == 0:
    print("  🎉 All checks passed — fix is verified!")
else:
    print("  ⚠️  Some checks failed — review above output.")
print()
