import streamlit as st
from openai import OpenAI
import mimetypes
import os
import re
from datetime import datetime
from utils import extract_thinking, extract_code, execute_code
from rag.llama_guard import check_safety_llama_guard_3

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

def _serialize_retrieved_docs(retrieved_docs):
    """Convert Document objects to serializable dictionaries."""
    if not retrieved_docs:
        return None
    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in retrieved_docs
    ]

def _render_retrieved_doc(doc, key_prefix="", doc_index=0):
    """Render one retrieved document entry."""
    metadata = doc.get("metadata", {})
    source_path = metadata.get("source")

    st.markdown(f"📄 **Dokument:** `{metadata.get('doc_id', 'Unbekannt')}`")
    st.markdown(f"📂 **Ordner:** `{metadata.get('folder', 'Unbekannt')}`")
    st.markdown(f"📜 **Quelle:** `{metadata.get('source', 'Unbekannt')}`")
    st.markdown(f"📑 **Seite:** `{metadata.get('page_number', 'Unbekannt')}`")

    if source_path and os.path.isfile(source_path):
        try:
            with open(source_path, "rb") as file_handle:
                file_bytes = file_handle.read()

            mime_type = mimetypes.guess_type(source_path)[0] or "application/octet-stream"
            st.download_button(
                label="⬇️ Dokument herunterladen",
                data=file_bytes,
                file_name=os.path.basename(source_path),
                mime=mime_type,
                key=f"download_{key_prefix}{metadata.get('doc_id', 00)}_{doc_index}",
            )
        except Exception as e:
            st.caption(f"Download nicht verfügbar: {str(e)}")
    else:
        st.caption("Download nicht verfügbar: Quelldatei nicht gefunden.")

    st.markdown("**🔍 Relevanter Ausschnitt:**")
    st.markdown(doc.get("page_content", ""))


@st.fragment
def _render_retrieved_docs_fragment(retrieved_docs, key_prefix=""):
    """Render retrieved documents inside a fragment so reruns stay local to this section."""
    if not retrieved_docs:
        return

    with st.expander("Dokumente anzeigen"):
        for doc_index, doc in enumerate(retrieved_docs):
            _render_retrieved_doc(doc, key_prefix=key_prefix, doc_index=doc_index)
            st.divider()
            
def new_query_needed(relevant_date, docs) -> bool:
    
    """Returns True if one document is not valid for the relevant timestamp, so a new query can get executed

    Returns:
        bool: True if one document is not valid for the relevant timestamp, False otherwise
    """
    for doc in docs:
        valid_from = doc.metadata.get("valid_from", "unknown")
        valid_to = doc.metadata.get("valid_to", "unknown")
        
        # date = datetime.strptime(rel_date, "%Y-%m-%d").date()
        valid_from = datetime.strptime(valid_from, "%d.%m.%Y").date() if valid_from != "unknown" else datetime.strptime("1800-01-01", "%Y-%m-%d").date()
        valid_to = datetime.strptime(valid_to, "%d.%m.%Y").date() if valid_to != "unknown" else datetime.strptime("9999-12-31", "%Y-%m-%d").date()  # Far future date if unknown

        if valid_from <= relevant_date <= valid_to:
            pass
        else:
            return True
    
    return False

def run_chatbot(vector_store, client, with_thinking=True):
    st.sidebar.markdown(
        "👋 **Willkommen beim R+S Auskunft ChatBot!**\n\n"
        "Dieser Chatbot unterstützt dich bei Fragen rund um Rollladen und Sonnenschutz. Stelle Fragen zu Rollladen und Sonnenschutz oder "
        "bestimmten Produkten und Services – ich helfe dir gerne weiter! 📚✨"
    )
    
    if "conversation_index" not in st.session_state:
        st.session_state["conversation_index"] = 0
    
    
    # tracing rag_steps for correct displays 
    # "initial" -> "awaiting_date" -> "date_received" -> "answering"
    steps = {
        "wait_for_input": 1,
        "wait_for_safety_check": 2,
        "safety_checked": 3,
        "retrieve_context": 4,
        "context_retrieved": 5,
        "awaiting_date": 6,
        "date_received": 7,
        "answering": 8
    }
    if "rag_step" not in st.session_state:
        st.session_state["rag_step"] = steps["wait_for_input"]
    
    # Re-render prior conversation on every rerun so widget actions do not clear the transcript.
    for message_index, message in enumerate(st.session_state.get("messages", [])):
        # if message_index < st.session_state["conversation_index"]:
        with st.chat_message(message.get("role", "assistant")):
            st.markdown(message.get("content", ""))

            metadata = message.get("metadata", {})
            retrieved_docs = metadata.get("retrieved_docs")

            if metadata.get("thinking_text"):
                with st.expander("Gedankengang anzeigen"):
                    st.markdown(metadata["thinking_text"])

            if retrieved_docs:
                _render_retrieved_docs_fragment(retrieved_docs, key_prefix=f"history_{message_index}_")


    if prompt := st.chat_input("Was möchtest du wissen?",  on_submit=lambda: st.session_state.update({"rag_step": steps["wait_for_safety_check"]})):
        # Append the user's message.
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state["active_prompt"] = prompt
        with st.chat_message("user",):
            st.markdown(prompt)
        st.rerun()
            
        
        # Safety-Check
    if st.session_state["rag_step"] == steps["wait_for_safety_check"]:
        with st.status("🔒 Führe Sicherheitsüberprüfung durch...", expanded=True) as status: 
            is_safe, explanation = check_safety_llama_guard_3(prompt)
            if not is_safe:
                status.update(label="⚠️ Sicherheitswarnung", state="error")
                st.error(explanation)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": explanation,
                    "metadata": {}  # No thinking or docs for safety errors
                })
                st.stop()
            st.session_state["rag_step"] = steps["safety_checked"]
            status.update(label="✅ Sicherheitsüberprüfung erfolgreich", state="complete")
    elif st.session_state["rag_step"] >= steps["safety_checked"]:
        with st.status(label="✅ Sicherheitsüberprüfung erfolgreich", state="complete") as status:
            print("SADOASDOAISjd")
        
        
    if st.session_state["rag_step"] == steps["safety_checked"]:
        # ===== Retrieve relevant context =====
        with st.status("🔍 Suche relevante Informationen...", expanded=True) as status:
            print("hAH")
            try:
                retrieved_docs = vector_store.similarity_search(prompt, k=5)
                print(f"Debug: Retrieved {len(retrieved_docs)} documents for query: '{prompt}'")
                
                if retrieved_docs:
                    for i, doc in enumerate(retrieved_docs):
                        print(f"Debug: Doc {i+1} - Folder: {doc.metadata.get('folder', 'Unknown')}, Source: {doc.metadata.get('source', 'Unknown')}")
                        print(f"Debug: Content preview: {doc.page_content[:100]}...")
                else:
                    print("Debug: No documents retrieved!")
                    
                status.update(label=f"✅ Relevante Informationen gefunden", state="complete")
            except Exception as e:
                print(f"Debug: Error in similarity search: {str(e)}")
                retrieved_docs = []
                status.update(label="⚠️ Fehler bei der Suche nach relevanten Informationen", state="error")

        context = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "Keine relevanten Dokumente gefunden."
        
        if retrieved_docs:
            st.session_state.retrieved_docs = retrieved_docs
        st.session_state["rag_step"] = steps["context_retrieved"]
        
        print(st.session_state["messages"])
        
    elif st.session_state["rag_step"] >= steps["context_retrieved"] and "retrieved_docs" in st.session_state:
        with st.status(label="✅ Relevante Informationen gefunden", state="complete") as status:
            pass
        
    # ===== Check for time relevant content =====
    if st.session_state["rag_step"] >= steps["context_retrieved"] and retrieved_docs:
        # if no document is time relevant skip to answering
        st.session_state["rag_step"] = steps["answering"] 
        for doc in st.session_state.retrieved_docs:
            if doc.metadata["valid_from"] != "unknown" or doc.metadata["valid_to"] != "unknown":
                st.session_state["rag_step"] = steps["awaiting_date"]
                break
            
    if st.session_state["rag_step"] == steps["awaiting_date"]:
        with st.status("Für einige der gefundenen Quellen sind Gültigkeitszeiträume hinterlegt. Welcher Zeitraum ist für deine Frage relevant?", expanded=True) as status:                
            with st.form("timestamp_form"):
                timestamp = st.date_input(
                    "Welcher Zeitraum ist für deine Frage relevant?"
                )   

                submitted = st.form_submit_button("Weiter", on_click=lambda: st.session_state.update({
                    "relevant_date": timestamp, 
                    "awaiting_date": False, 
                    "rag_step": steps["date_received"]
                }))

            if not submitted:
                st.stop()
            
            st.session_state.relevant_date = timestamp
            st.session_state["rag_step"] = steps["date_received"]

            status.update(state="complete")
            st.rerun()

            # Continue only after submit
    if st.session_state["rag_step"] >= steps["date_received"]:
        st.session_state["rag_step"] = steps["answering"]
        if "relevant_date" not in st.session_state:
            st.status("✅ Keine zeitlich relevanten Quellen gefunden", state="complete")
        else:
            # st.status("✅ Zeitlich relevante Quellen gefunden", state="complete")
                        
            with st.status(f"Überprüfe, ob die gefundenen Quellen für den Zeitpunkt {st.session_state['relevant_date']} relevant sind...", expanded=True) as status:
                if new_query_needed(st.session_state["relevant_date"], st.session_state.retrieved_docs):
                    prompt = f"Bitte berücksichtige nur Informationen, die für den Zeitpunkt {st.session_state['relevant_date']} relevant sind. {st.session_state['active_prompt']}"            
                    try:
                        retrieved_docs = vector_store.similarity_search(prompt, k=5)
                        print(f"Debug: Retrieved {len(retrieved_docs)} documents for query: '{prompt}'")
                        
                        if retrieved_docs:
                            for i, doc in enumerate(retrieved_docs):
                                print(f"Debug: Doc {i+1} - Folder: {doc.metadata.get('folder', 'Unknown')}, Source: {doc.metadata.get('source', 'Unknown')}")
                                print(f"Debug: Content preview: {doc.page_content[:100]}...")
                        else:
                            print("Debug: No documents retrieved!")
                            
                        status.update(label=f"✅ Relevante Informationen gefunden", state="complete", expanded=False)
                    except Exception as e:
                        print(f"Debug: Error in similarity search: {str(e)}")
                        retrieved_docs = []
                        status.update(label="⚠️ Fehler bei der Suche nach relevanten Informationen", state="error", expanded=False)
                    st.session_state.retrieved_docs = retrieved_docs
                else:
                    status.update(label=f"✅ Bereits gefundene Dokumente sind für den relevanten Zeitraum gültig", state="complete", expanded=False)
            
        
        # ===== Build an augmented system prompt from the base prompt and the newly retrieved context. =====
        if st.session_state["rag_step"] == steps["answering"]:
            
            context = "\n\n".join([doc.page_content for doc in st.session_state["retrieved_docs"]]) if st.session_state["retrieved_docs"] else "Keine relevanten Dokumente gefunden."
            with st.status("🔍 Initialisiert das Modell...", expanded=True) as status:
                system_prompt_with_context = (
                    st.session_state["base_system_prompt"] +
                    "\n\n=== Kontext aus Dokumenten ===\n" +
                    context +
                    "\n=== Ende Kontext ===\n"
                )

                # Build the messages list using the augmented prompt.
                messages = [{"role": "system", "content": system_prompt_with_context}] + st.session_state.messages
                # =============================================================

                # Call the API.
                completion = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=messages,
                    max_tokens=16384,
                    temperature=0.6,
                    stream=True  # Enable streaming
                )

                print("Query:", system_prompt_with_context)

                # Initialize variables to collect the full response
                full_response = ""
                thinking_text = ""
                in_thinking_block = True
                status.update(label="✅ Modell initialisiert", state="complete")

            with st.status("🤔 Denkt nach...", expanded=True) as status:
                # Get the first chunk to check if thinking starts immediately
                try:
                    first_chunk = next(completion)
                except StopIteration:
                    first_chunk = None
                # That chunk can be empty (apparently?!)
                while first_chunk is not None and (_get_stream_content(first_chunk) is None or _get_stream_content(first_chunk) == ''):
                    try:
                        first_chunk = next(completion)
                    except StopIteration:
                        first_chunk = None
                        break
                print(first_chunk)
                first_content = _get_stream_content(first_chunk) if first_chunk is not None else None
                if first_content is not None:
                    if first_content.startswith("<think>"):
                        # Thinking starts immediately
                        thinking_text += first_content
                        in_thinking_block = True
                    else:
                        # No thinking tag in first chunk
                        in_thinking_block = False
                        full_response += first_content
                
                # Process remaining chunks
                for chunk in completion:
                    content = _get_stream_content(chunk)
                    if content is not None:
                        
                        # Handle thinking block
                        if "</think>" in content:
                            thinking_text += content
                            in_thinking_block = False
                            status.update(label="✅ Gedankengang erstellt", state="complete")
                            break
                        elif in_thinking_block:
                            thinking_text += content
                            continue
                        else:
                            full_response += content

        # Continue processing remaining chunks outside of status block

        current_message = st.chat_message("assistant")
        message_placeholder = current_message.empty()

        # Create expanders before the message
        thinking_expander = st.expander("Gedankengang anzeigen")

        for chunk in completion:
            content = _get_stream_content(chunk)
            if content is not None:
                full_response += content
                message_placeholder.markdown(full_response + "▌")

        # Remove the cursor after completion
        message_placeholder.markdown(full_response)

        assistant_response = full_response
        code = extract_code(assistant_response)
        thinking_text = extract_thinking(thinking_text)

        print("Assistant response:", assistant_response)
        
        if code:
            # ... (Handle Python code extraction, execution, follow-up, etc.)
            execution_result = execute_code(code)
            follow_up_prompt = (
                f"Der generierte Python-Code wurde ausgeführt. Das Ergebnis lautet:\n\n"
                f"{execution_result}\n\n"
                "Bitte stelle den entsprechenden Python-Code in einem Codeblock bereit (eingeschlossen in drei Backticks mit 'python') und erkläre anschließend kurz das Ergebnis."
            )
            
            follow_up_messages = (
                [{"role": "system", "content": system_prompt_with_context}] +
                st.session_state.messages +
                [{"role": "user", "content": follow_up_prompt}]
            )
            
            # Handle follow-up response with streaming as well
            follow_up_completion = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=follow_up_messages,
                stream=True,
            )

            # Initialize variables for follow-up response
            follow_up_response = ""
            follow_up_thinking_text = ""
            in_follow_up_thinking_block = True
            follow_up_message = st.chat_message("assistant")
            follow_up_placeholder = follow_up_message.empty()

            # Process the streamed follow-up response
            for chunk in follow_up_completion:
                content = _get_stream_content(chunk)
                if content is not None:
                    
                    # Handle thinking block
                    if "</think>" in content:
                        follow_up_thinking_text += content
                        in_follow_up_thinking_block = False
                        follow_up_placeholder.markdown("")  # Clear the thinking indicator
                        continue
                    elif in_follow_up_thinking_block:
                        follow_up_thinking_text += content
                        # Show thinking animation
                        follow_up_placeholder.markdown("🤔 Denkt nach...")
                        continue
                    else:
                        # Only update the message placeholder with non-thinking content
                        follow_up_response += content
                        follow_up_placeholder.markdown(follow_up_response + "▌")

            # Remove the cursor after completion
            follow_up_placeholder.markdown(follow_up_response)
            
            new_thought = extract_thinking(follow_up_response)

            if new_thought and thinking_text:
                thinking_text += f"\n\n**Gedankengang zur Berechnung:**\n{new_thought}"

            if thinking_text:
                follow_up_response = re.sub(r"<think>.*?</think>", "", follow_up_response, flags=re.DOTALL)
                thinking_expander.markdown(thinking_text)
            
            # Store message with metadata for expanders
            message_metadata = {}
            if thinking_text:
                message_metadata["thinking_text"] = thinking_text
            if retrieved_docs:
                message_metadata["retrieved_docs"] = _serialize_retrieved_docs(retrieved_docs)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": follow_up_response,
                "metadata": message_metadata if message_metadata else {}
            })
        else:
            if thinking_text:
                assistant_response = re.sub(r"<think>.*?</think>", "", assistant_response, flags=re.DOTALL)
                thinking_expander.markdown(thinking_text)

            if retrieved_docs:
                _render_retrieved_docs_fragment(_serialize_retrieved_docs(retrieved_docs), key_prefix="live_")

            # Store message with metadata for expanders
            message_metadata = {}
            if thinking_text:
                message_metadata["thinking_text"] = thinking_text
            if retrieved_docs:
                message_metadata["retrieved_docs"] = _serialize_retrieved_docs(retrieved_docs)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": assistant_response,
                "metadata": message_metadata if message_metadata else {}
            })
            
            st.session_state.update({"rag_step": steps["wait_for_input"],
                                     "active_prompt": None,
                                     })
        # st.session_state["rag_step"] = steps["wait_for_input"]