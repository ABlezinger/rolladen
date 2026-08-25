import streamlit as st
from openai import OpenAI
import mimetypes
import os
import re
from datetime import datetime
from src.rag.utils import extract_thinking, extract_code, execute_code
from src.rag.llama_guard import check_safety_llama_guard_3
from src.rag.tools import get_list_of_available_docs
from src.document_utils import DOCUMENT_CLASSES, get_doc_class_info_text

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

    st.markdown(f"📄 **Dokument:** `{metadata.get('title', 'Unbekannt')}`")
    st.markdown(f"❗️ **Dokumentklasse:** `{metadata.get('doc_class', 'Unbekannt')}`")
    st.markdown(f"📂 **Ordner:** `{metadata.get('folder', 'Unbekannt')}`")
    st.markdown(f"📜 **Quelle:** `{metadata.get('source', 'Unbekannt')}`")
    st.markdown(f"📑 **Seite:** `{metadata.get('page_number', 'Unbkannt')}`")
    valid_from = metadata["valid_from"]
    if valid_from == 15000101:
        valid_from = "-"
    else: 
        valid_from = datetime.strptime(str(valid_from), "%Y%m%d").strftime("%d.%m.%Y")
    valid_to = metadata["valid_to"]
    if valid_to == 99991231:
        valid_to = "-"
    else:
        valid_to = datetime.strptime(str(valid_to), "%Y%m%d").strftime("%d.%m.%Y")
    st.markdown(f"📅 **Gültig von:** `{valid_from}`, **bis:** `{valid_to}`")

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
                disabled=not metadata.get("downloadable", False),
                help="Dieses Dokument steht nicht zum download zur Verfügung." if not metadata.get("downloadable", False) else "Klicke hier, um das Dokument herunterzuladen."
            )
        except Exception as e:
            st.caption(f"Download nicht verfügbar: {str(e)}")
    else:
        st.caption("Download nicht verfügbar: Quelldatei nicht gefunden.")

    st.markdown("**🔍 Relevanter Ausschnitt:**")
    st.markdown(doc.get("page_content", ""))
    
def _create_context_snippet(doc):
    """Create a context snippet for the system prompt from a Document object."""
    metadata = doc.metadata
    valid_from = metadata["valid_from"]
    if valid_from == 15000101:
        valid_from = "-"
    else: 
        valid_from = datetime.strptime(str(valid_from), "%Y%m%d").strftime("%d.%m.%Y")
    
    valid_to = metadata["valid_to"]
    if valid_to == 99991231:
        valid_to = "-"
    else:
        valid_to = datetime.strptime(str(valid_to), "%Y%m%d").strftime("%d.%m.%Y")
    return doc.page_content + "\nGültig von: " + valid_from + ", bis: " + valid_to


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
        valid_from = doc.metadata["valid_from"]
        valid_to = doc.metadata["valid_to"]
        int_date = int(relevant_date.strftime("%Y%m%d"))

        if valid_from <= int_date <= valid_to:
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
        
    if "search_config" not in st.session_state:
        st.session_state["search_config"] = {
            "doc_classes": list(DOCUMENT_CLASSES.keys()),  # Default to all classes
            "relevant_date": None
        }
    
    
    # tracing rag_steps for correct displays 
    steps = {
        "wait_for_input": 1,
        "wait_for_safety_check": 2,
        "planning": 3,
        "retrieval_execution": 4,
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


    with st.bottom:
        with st.container(horizontal=True, width="stretch", horizontal_alignment="right"):
            # Search Configurations
            with st.popover(icon=":material/tune:", label=""):
                class_filter = st.segmented_control("Dokumentklassen filtern", 
                                    ["Klasse 1", "Klasse 2", "Klasse 3"], 
                                    key="doc_class_filter", 
                                    selection_mode="multi",
                                    help=f"Dokumentklasse(n) zum Filtern auswählen \n{get_doc_class_info_text()}")
                # date_filter = st.date_input("Relevantes Datum (optional)", 
                #               key="relevant_date_filter", 
                #               value = None,
                #               help="Optionales Datum zum Filtern der Dokumente. Nur Dokumente, die zu diesem Zeitpunkt gültig sind, werden berücksichtigt.")
                st.session_state["search_config"] = {
                    "doc_classes": [DOCUMENT_CLASSES[cls] for cls in class_filter] if class_filter else list(DOCUMENT_CLASSES.values()),
                    # "relevant_date": date_filter
                }
                print("Search Config Updated:", st.session_state["search_config"])
                

    if prompt := st.chat_input("Was möchtest du wissen?",  on_submit=lambda: st.session_state.update({"rag_step": steps["wait_for_safety_check"]})):
        # Append the user's message.
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state["active_prompt"] = prompt
        
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["rag_step"] = steps["wait_for_safety_check"]    
        st.rerun()
            
        
    # ===== Safety-Check =====
    if st.session_state["rag_step"] == steps["wait_for_safety_check"]:
        with st.status("🔒 Führe Sicherheitsüberprüfung durch...", expanded=True) as status: 
            is_safe, explanation = check_safety_llama_guard_3(st.session_state["active_prompt"])
            if not is_safe:
                status.update(label="⚠️ Sicherheitswarnung", state="error")
                st.error(explanation)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": explanation,
                    "metadata": {}  # No thinking or docs for safety errors
                })
                st.stop()
            st.session_state["rag_step"] = steps["planning"]
            status.update(label="✅ Sicherheitsüberprüfung erfolgreich", state="complete")
    elif st.session_state["rag_step"] >= steps["planning"]:
        with st.status(label="✅ Sicherheitsüberprüfung erfolgreich", state="complete") as status:
            pass
    
    
    # ===== Planning Select Tools and Date Phase =====
    if st.session_state["rag_step"] == steps["planning"]:
        with st.status("📝 Plane das vorgehen...", expanded=True) as status:
            
            planning_prompt = st.session_state["planning_prompt"]
            
            print("MESSAGES:", st.session_state.messages)
            
            messages = [{"role": "system", "content": planning_prompt}] + st.session_state.messages
            planning_messages = [message.copy() for message in messages if message["role"] != "assistant"]

            for message in planning_messages:
                message.pop("metadata", None)            
            # print("Planning Prompt:")
            # print(planning_prompt)
            # print("Messages:")
            # print(messages)
            # Call the API.
            completion = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=planning_messages,
                max_tokens=16384,
                temperature=0,
                stream=False  # disable streaming
            )
            
            print("Planning Messages:", planning_messages)
            
            answer = completion.choices[0].message.content
            print("Planning Answer:", answer)
            
            print("STILL MESSAGES: ", st.session_state.messages)
            try:
                st.session_state.tool_match = re.search(r"<TOOL>(.*?)</TOOL>", answer).group(1)
            except AttributeError:
                st.session_state.tool_match = "SIMILARITY SEARCH"
            try:
                date_match = re.search(r"<DATE>(.*?)</DATE>", answer).group(1)
                if date_match == "n.a.":
                    st.session_state.relevant_date = None
                else:
                    st.session_state.relevant_date = datetime.strptime(date_match, "%Y-%m-%d")
            except AttributeError:
                st.session_state.relevant_date = None
            
            # print(answer)
            
            # Placeholder for planning logic
            if st.session_state.relevant_date is None:
                status.write("Kein relevantes Datum erkannt.")
            else:
                status.write(f"Relevantes Datum erkannt: {st.session_state.relevant_date.strftime('%d.%m.%Y')}")
            status.write(f"Erkannter Tool-Einsatz: {st.session_state.tool_match}")
            status.update(label=f"✅ Planung abgeschlossen", state="complete")
            st.session_state["rag_step"] = steps["retrieval_execution"]
    elif st.session_state["rag_step"] > steps["planning"]:
        print("WARUM: " + str(st.session_state["rag_step"]))
        with st.status(label=f"✅ Planung abgeschlossen", state="complete") as status:
            pass
        
    if st.session_state["rag_step"] == steps["retrieval_execution"]:
        # ===== Retrieve relevant context =====
        with st.status("🔍 Suche relevante Informationen...", expanded=True) as status:
            
                print("prompt:", st.session_state["active_prompt"])
                
                # Tool: List Documents 
                if st.session_state.tool_match == "LIST_DOCUMENTS":
                    retreived_context = get_list_of_available_docs(vector_store)
                    print("Debug: Retrieved context from LIST_DOCUMENTS:", retreived_context)
                    status.update(label=f"✅ Relevante Informationen gefunden", state="complete")
                
                # Default tool: Similarity Search
                else:
                    try:
                        if st.session_state.get("relevant_date") is not None:
                            int_date = int(st.session_state["relevant_date"].strftime("%Y%m%d"))
                            retrieved_docs = vector_store.similarity_search(
                                st.session_state["active_prompt"], 
                                k=5,
                                filter={"$and": [
                                    {"valid_from": {"$lte": int_date}}, 
                                    {"valid_to": {"$gte": int_date}},
                                    {"doc_class": {"$in": st.session_state["search_config"]["doc_classes"]}}]})
                            print(f"Debug: Retrieved {len(retrieved_docs)} documents for query: '{st.session_state['active_prompt']}' with date filter: {int_date}")
                        else:
                            retrieved_docs = vector_store.similarity_search(
                                st.session_state["active_prompt"], 
                                k=5,
                                filter={"doc_class": {"$in": st.session_state["search_config"]["doc_classes"]}})
                            print(f"Debug: Retrieved {len(retrieved_docs)} documents for query: '{st.session_state['active_prompt']}'")
                        
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
                    

        # context = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "Keine relevanten Dokumente gefunden."
        
        if retrieved_docs:
            st.session_state.retrieved_docs = retrieved_docs
        elif retreived_context:
            st.session_state.retrieved_context = retreived_context
        st.session_state["rag_step"] = steps["context_retrieved"]
        
        
    elif st.session_state["rag_step"] > steps["context_retrieved"] and "retrieved_docs" in st.session_state:
        with st.status(label="✅ Relevante Informationen gefunden", state="complete") as status:
            pass
        
    # ===== Check for time relevant content =====
    if st.session_state["rag_step"] == steps["context_retrieved"] and "retrieved_docs" in st.session_state:
        # if no document is time relevant skip to answering
        st.session_state["rag_step"] = steps["answering"] 
        
        print("HERER: " + str(st.session_state["relevant_date"]))
        print("STEP: " + str(st.session_state["rag_step"]))
        if st.session_state.get("relevant_date") is None:
            for doc in st.session_state.retrieved_docs:
                print("JAJAJA")
                if doc.metadata["valid_from"] != 15000101 or doc.metadata["valid_to"] != 99991231:
                    st.session_state["rag_step"] = steps["awaiting_date"]
                    break
            
    if st.session_state["rag_step"] == steps["awaiting_date"]:
        if "relevant_date" in st.session_state and st.session_state["relevant_date"] is not None:
            st.session_state["rag_step"] = steps["date_received"]
        else:
            with st.status("Für einige der gefundenen Quellen sind Gültigkeitszeiträume hinterlegt. Welcher Zeitraum ist für deine Frage relevant?", expanded=True) as status:                
                with st.form("timestamp_form"):
                    timestamp = st.date_input(
                        "Welcher Zeitpunkt ist für deine Frage relevant?",
                        min_value=datetime(1900, 1, 1),
                        format="DD.MM.YYYY",
                    )   
                    
                    submitted = st.form_submit_button("Weiter")
                    not_relevant = st.form_submit_button("Der Zeitpunkt ist nicht relevant")
                if submitted:
                    st.session_state["relevant_date"] = timestamp
                    st.session_state["awaiting_date"] = False
                    st.session_state["rag_step"] = steps["date_received"]
                    st.session_state["new_query_needed"] = new_query_needed(st.session_state["relevant_date"], st.session_state.retrieved_docs)
                    st.rerun()
                elif not_relevant:
                    st.session_state["relevant_date"] = None
                    st.session_state["awaiting_date"] = False
                    st.session_state["rag_step"] = steps["answering"]
                    st.rerun()
                else:
                    st.stop()
                
                # st.session_state.relevant_date = timestamp
                st.session_state["rag_step"] = steps["date_received"]

                status.update(state="complete")
                st.rerun()

            # Continue only after submit
            
    # ==== Answering the question with context =====
    if st.session_state["rag_step"] >= steps["date_received"] and st.session_state["rag_step"] < steps["answering"]:
        st.session_state["rag_step"] = steps["answering"]
        
        # Check if another time relevant query is necessary
        if "relevant_date" not in st.session_state or st.session_state["relevant_date"] is None:
            st.status("✅ Keine zeitlich relevanten Quellen gefunden", state="complete")
        else:                        
            with st.status(f"Überprüfe, ob die gefundenen Quellen für den Zeitpunkt {st.session_state['relevant_date']} relevant sind...", expanded=False) as status:
                if st.session_state.get("new_query_needed", False):
                    status.update(label=f"⚠️ Einige Quellen sind für den Zeitpunkt {st.session_state['relevant_date']} nicht relevant. Eine neue Suche wird durchgeführt...", state="running", expanded=False)
                    int_date = int(st.session_state["relevant_date"].strftime("%Y%m%d"))

                    try:
                        retrieved_docs = vector_store.similarity_search(
                            st.session_state["active_prompt"], 
                            k=5,
                            filter={"$and": [{"valid_from": {"$lte": int_date}}, {"valid_to": {"$gte": int_date}}]})
                        print(f"Debug: Retrieved {len(retrieved_docs)} documents for query: '{st.session_state['active_prompt']}' with date filter: {int_date}")
                        
                        if retrieved_docs:
                            for i, doc in enumerate(retrieved_docs):
                                print(f"Debug: Doc {i+1} - Folder: {doc.metadata.get('folder', 'Unknown')}, Source: {doc.metadata.get('source', 'Unknown')}")
                                print(f"Debug: Content preview: {doc.page_content[:100]}...")
                        else:
                            print("Debug: No documents retrieved!")
                            
                        status.update(label=f"✅ Neue zetilich passende Informationen gefunden", state="complete", expanded=False)
                    except Exception as e:
                        print(f"Debug: Error in similarity search: {str(e)}")
                        retrieved_docs = []
                        status.update(label="⚠️ Fehler bei der Suche nach relevanten Informationen", state="error", expanded=False)
                    st.session_state.retrieved_docs = retrieved_docs
                else:
                    status.update(label=f"✅ Bereits gefundene Dokumente sind für den relevanten Zeitraum gültig", state="complete", expanded=False)
            
        
    # ===== Build an augmented system prompt from the base prompt and the newly retrieved context. =====
    if st.session_state["rag_step"] == steps["answering"]:
        
        if "retrieved_docs" in st.session_state:
            context = "\n\n".join([
                _create_context_snippet(doc)
                for doc in st.session_state["retrieved_docs"]]) if (
                    st.session_state["retrieved_docs"]) else "Keine relevanten Dokumente gefunden."
        elif "retrieved_context" in st.session_state:
            context = st.session_state["retrieved_context"]
        with st.status("🔍 Initialisiert das Modell...", expanded=True) as status:
            system_prompt_with_context = (
                st.session_state["base_system_prompt"] +
                "\n\n=== Kontext aus Dokumenten ===\n" +
                context +
                "\n=== Ende Kontext ===\n"
                "\n Relevantes Datum: " + (st.session_state.get("relevant_date").strftime("%d.%m.%Y") if st.session_state.get("relevant_date") else "--")
            )

            
            
            # Build the messages list using the augmented prompt.
            messages = [{"role": "system", "content": system_prompt_with_context}] + st.session_state.messages
            # =============================================================
            # print("CALLING API WITH MESSAGES:")
            # print(messages)
            # Call the API.
            completion = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=messages,
                max_tokens=16384,
                temperature=0.6,
                stream=True  # Enable streaming
            )

            # print("Query:", system_prompt_with_context)

            # Initialize variables to collect the full response
            full_response = ""
            thinking_text = ""
            in_thinking_block = True
            status.update(label="✅ Modell initialisiert", state="complete")

        with st.status("🤔 Denkt nach...", expanded=False) as status:
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
        # thinking_expander = st.expander("Gedankengang anzeigen")

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
            if len(st.session_state["retrieved_docs"]) > 0:
                message_metadata["retrieved_docs"] = _serialize_retrieved_docs(st.session_state["retrieved_docs"])
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": follow_up_response,
                "metadata": message_metadata if message_metadata else {}
            })
        else:
            if thinking_text:
                assistant_response = re.sub(r"<think>.*?</think>", "", assistant_response, flags=re.DOTALL)
                thinking_expander.markdown(thinking_text)


            # Store message with metadata for expanders
            message_metadata = {}
            if thinking_text:
                message_metadata["thinking_text"] = thinking_text
            if len(st.session_state["retrieved_docs"]) > 0:
                message_metadata["retrieved_docs"] = _serialize_retrieved_docs(st.session_state["retrieved_docs"])
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": assistant_response,
                "metadata": message_metadata if message_metadata else {}
            })
            
            st.session_state.update({"rag_step": steps["wait_for_input"],
                                     "active_prompt": None,
                                     "relevant_date": None,
                                     "new_query_needed": False
                                     })
            
        
        if len(st.session_state["retrieved_docs"]) > 0 and st.session_state["rag_step"] == steps["wait_for_input"]:
            _render_retrieved_docs_fragment(_serialize_retrieved_docs(st.session_state["retrieved_docs"]), key_prefix="live_")
        # st.session_state["rag_step"] = steps["wait_for_input"]