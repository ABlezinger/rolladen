import os
import mimetypes
import streamlit as st
from src.rag.llama_guard import check_safety_llama_guard_3
from src.rag.utils import extract_thinking
from src.llm_client import safe_completion, safe_stream_iter, LLMServiceError, LLM_UNAVAILABLE_MESSAGE


def _get_stream_content(chunk):
    """Safely extract content string from a streaming chunk, or None if unavailable."""
    try:
        choices = getattr(chunk, "choices", None)
        if not choices:
            return None
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return None
        return getattr(delta, "content", None)
    except Exception:
        return None


def _get_available_files(vector_store):
    """Return a sorted list of distinct source documents stored in the vector store.

    Each entry contains the metadata needed to display and filter by a single document:
    source path, display title, folder and whether it is downloadable.
    """
    try:
        stored_data = vector_store.get()
    except Exception as e:
        print(f"Debug: Error reading vector store contents for file chat: {str(e)}")
        return []

    metadatas = stored_data.get("metadatas", []) or []
    files_by_source = {}
    for metadata in metadatas:
        if not isinstance(metadata, dict):
            continue
        source = metadata.get("source")
        # Skip chunks without a real source file (e.g. the generated document index).
        if not source or source == "document_index":
            continue
        if source not in files_by_source:
            files_by_source[source] = {
                "source": source,
                "title": metadata.get("title") or os.path.basename(source),
                "folder": metadata.get("folder", ""),
                "downloadable": metadata.get("downloadable", False),
            }

    return sorted(files_by_source.values(), key=lambda entry: entry["title"].lower())


def _file_label(file_entry):
    """Build a human readable label for a file entry, including its folder if known."""
    title = file_entry.get("title", "Unbekanntes Dokument")
    folder = file_entry.get("folder")
    return f"{title} ({folder})" if folder else title


def _build_file_system_prompt(file_entry):
    """Build a system prompt that restricts the assistant to the selected document."""
    return (
        "Du bist ein hilfreicher Assistent, der ausschließlich Fragen zu genau einem "
        "ausgewählten Dokument aus der Wissensdatenbank beantwortet.\n"
        f"Das aktuell ausgewählte Dokument heißt \"{file_entry['title']}\".\n\n"
        "**Rahmenbedingungen:**\n"
        "- Beantworte Fragen ausschließlich auf Basis der unten bereitgestellten Ausschnitte aus diesem Dokument.\n"
        "- Wenn die Antwort nicht aus den bereitgestellten Ausschnitten hervorgeht, sage das ehrlich "
        "und erfinde keine Informationen.\n"
        "- Gib, wenn möglich, die Seite an, auf der du eine Information gefunden hast.\n"
        "- Beziehe dich nicht auf andere Dokumente aus der Wissensdatenbank, auch wenn du sie kennst.\n"
        "- Kommuniziere klar, präzise und in einem freundlichen, professionellen Ton.\n"
        "- Antworte in der Sprache, in der die Nutzerin/der Nutzer die Frage stellt.\n"
    )


def run_file_chat(vector_store, client):
    st.title("📄 Datei-Chat")
    st.sidebar.markdown(
        "👋 **Willkommen beim Datei-Chat!**\n\n"
        "Wähle unten ein Dokument aus der Wissensdatenbank aus und stelle anschließend "
        "gezielt Fragen dazu. Ich beantworte deine Fragen ausschließlich auf Basis des "
        "ausgewählten Dokuments. 📚✨"
    )

    available_files = _get_available_files(vector_store)

    if not available_files:
        st.warning("⚠️ Es konnten keine Dokumente in der Wissensdatenbank gefunden werden.")
        st.stop()

    file_options = {file_entry["source"]: file_entry for file_entry in available_files}

    col1, col2 = st.columns([4, 1])
    with col1:
        selected_source = st.selectbox(
            "📁 Dokument auswählen",
            options=list(file_options.keys()),
            format_func=lambda source: _file_label(file_options[source]),
            key="file_chat_selected_source",
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Chat zurücksetzen"):
            st.session_state["file_chat_messages"] = []
            st.rerun()

    selected_file = file_options[selected_source]

    # Reset the conversation whenever a new document is selected.
    if st.session_state.get("file_chat_active_source") != selected_source:
        st.session_state["file_chat_active_source"] = selected_source
        st.session_state["file_chat_messages"] = []

    if selected_file.get("downloadable") and os.path.isfile(selected_source):
        try:
            with open(selected_source, "rb") as file_handle:
                file_bytes = file_handle.read()
            mime_type = mimetypes.guess_type(selected_source)[0] or "application/octet-stream"
            st.download_button(
                label="⬇️ Dokument herunterladen",
                data=file_bytes,
                file_name=os.path.basename(selected_source),
                mime=mime_type,
                key="file_chat_download_button",
            )
        except Exception as e:
            st.caption(f"Download nicht verfügbar: {str(e)}")

    st.divider()

    if "file_chat_messages" not in st.session_state:
        st.session_state["file_chat_messages"] = []

    # Re-render prior conversation on every rerun so widget actions do not clear the transcript.
    for message in st.session_state["file_chat_messages"]:
        with st.chat_message(message.get("role", "assistant")):
            st.markdown(message.get("content", ""))
            thinking_text = message.get("metadata", {}).get("thinking_text")
            if thinking_text:
                with st.expander("Gedankengang anzeigen"):
                    st.markdown(thinking_text)

    prompt = st.chat_input(f"Frage zu \"{selected_file['title']}\" stellen...")

    if prompt:
        st.session_state["file_chat_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.status("🔄 Verarbeitungsschritte anzeigen", expanded=True) as steps_container:
            # ===== Safety-Check =====
            with st.status("🔒 Führe Sicherheitsüberprüfung durch...", expanded=True) as status:
                is_safe, explanation = check_safety_llama_guard_3(prompt)
                if not is_safe:
                    status.update(label="⚠️ Sicherheitswarnung", state="error")
                    st.error(explanation)
                    st.session_state["file_chat_messages"].append({
                        "role": "assistant",
                        "content": explanation,
                        "metadata": {}
                    })
                    st.stop()
                status.update(label="✅ Sicherheitsüberprüfung erfolgreich", state="complete")

            # ===== Retrieve relevant excerpts from the selected file only =====
            with st.status("🔍 Durchsuche das Dokument...", expanded=True) as status:
                try:
                    retrieved_docs = vector_store.similarity_search(
                        prompt,
                        k=8,
                        filter={"source": selected_source},
                    )
                    print(f"Debug: Retrieved {len(retrieved_docs)} chunks from '{selected_source}' for file chat query")
                    status.update(label="✅ Relevante Textstellen gefunden", state="complete")
                except LLMServiceError:
                    status.update(label="⚠️ Server nicht erreichbar", state="error")
                    st.error(LLM_UNAVAILABLE_MESSAGE)
                    st.stop()
                except Exception as e:
                    print(f"Debug: Error in file chat similarity search: {str(e)}")
                    retrieved_docs = []
                    status.update(label="⚠️ Fehler bei der Dokumentsuche", state="error")

            try:
                retrieved_docs = sorted(retrieved_docs, key=lambda doc: doc.metadata.get("page_number", 0))
            except Exception:
                pass

            context = "\n\n".join(
                f"[Seite {doc.metadata.get('page_number', 'unbekannt')}]\n{doc.page_content}"
                for doc in retrieved_docs
            ) if retrieved_docs else "Keine relevanten Textstellen im Dokument gefunden."

            system_prompt_with_context = (
                _build_file_system_prompt(selected_file) +
                "\n\n=== Ausschnitte aus dem Dokument ===\n" +
                context +
                "\n=== Ende der Ausschnitte ===\n"
            )

            history = [
                {"role": message["role"], "content": message["content"]}
                for message in st.session_state["file_chat_messages"]
            ]
            messages = [{"role": "system", "content": system_prompt_with_context}] + history

            model = st.session_state.get("openai_model") or st.secrets.get("model_id", "gemma-3-27b-it")

            full_response = ""
            thinking_text = ""
            in_thinking_block = False

            try:
                with st.status("🤔 Denkt nach...", expanded=False) as status:
                    completion = safe_completion(
                        client,
                        model=model,
                        messages=messages,
                        max_tokens=16384,
                        temperature=0.3,
                        stream=True,
                    )
                    completion = safe_stream_iter(completion)

                    for chunk in completion:
                        content = _get_stream_content(chunk)
                        if content is None or content == "":
                            continue

                        if "<think>" in content:
                            in_thinking_block = True
                            content = content.replace("<think>", "")
                        if "</think>" in content:
                            in_thinking_block = False
                            thinking_text += content.replace("</think>", "")
                            continue

                        if in_thinking_block:
                            thinking_text += content
                            continue

                        full_response += content

                    status.update(label="✅ Antwort erstellt", state="complete")
            except LLMServiceError:
                st.error(LLM_UNAVAILABLE_MESSAGE)
                st.stop()

        thinking_text = extract_thinking(thinking_text) or thinking_text.strip()

        with st.chat_message("assistant"):
            st.markdown(full_response)
            if thinking_text:
                with st.expander("Gedankengang anzeigen"):
                    st.markdown(thinking_text)

        # Collapse the RAG-steps container now that the response is displayed.
        steps_container.update(state="complete", expanded=False)

        message_metadata = {"thinking_text": thinking_text} if thinking_text else {}

        st.session_state["file_chat_messages"].append({
            "role": "assistant",
            "content": full_response,
            "metadata": message_metadata,
        })  
