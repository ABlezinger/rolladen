# AGENTS.md

## Running the app

This is a Streamlit chatbot. Always use the `chat_test` conda environment for this
project (not `bbs_chatbot` — its Python 3.14 is incompatible with the pinned Streamlit
version and fails at startup with `RuntimeError: There is no current event loop in
thread 'MainThread'`).

```bash
/home/alexander.blezinger/miniconda3/envs/chat_test/bin/python -m streamlit run src/chatbot.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Secrets (API keys, etc.) live in `.streamlit/secrets.toml` (gitignored, must exist locally).

## Layout

- `src/chatbot.py` — main Streamlit entrypoint
- `src/llm_client.py` — LLM completion wrapper
- `src/document_utils.py`, `src/filemanagement.py`, `src/handoff.py` — supporting app logic
- `src/rag/` — retrieval-augmented generation: vector store management
  (`vector_store_management.py`, `extend_vector_store.py`, `fix_vector_store.py`),
  prompts (`system_prompts.py`), tools (`tools.py`), safety (`llama_guard.py`)
- `src/addons/` — optional features (PDF generation, competency test generator, file chat)
- `setup_vector_store.py`, `smart_extend_vector_store.py`, `test_vector_store.py` — top-level
  scripts for building/maintaining the vector store
- `rsev.py` — standalone script (see file for purpose)

## Dependencies

Pinned in `requirements.txt` (Streamlit 1.61.0, LangChain 0.3.x, ChromaDB 1.0.19).
