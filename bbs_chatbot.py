import streamlit as st
from openai import OpenAI
import re
from datetime import datetime
from utils import extract_thinking, extract_code, execute_code
from llama_guard import check_safety_llama_guard_3

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

def run_chatbot(vector_store, client, with_thinking=True):
    st.sidebar.markdown(
        "👋 **Willkommen beim R+S Auskunft ChatBot!**\n\n"
        "Dieser Chatbot unterstützt dich bei Fragen rund um Rollladen und Sonnenschutz. Stelle Fragen zu Rollladen und Sonnenschutz oder "
        "bestimmten Produkten und Services – ich helfe dir gerne weiter! 📚✨"
    )


    if prompt := st.chat_input("Was möchtest du wissen?"):
        # Append the user's message.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        

        # Safety-Check
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
            status.update(label="✅ Sicherheitsüberprüfung erfolgreich", state="complete")

        # ===== Retrieve relevant context =====
        with st.status("🔍 Suche relevante Informationen...", expanded=True) as status:
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
        
        # ===== Build an augmented system prompt from the base prompt and the newly retrieved context. =====

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
        docs_expander = st.expander("Dokumente anzeigen")

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
                for doc in retrieved_docs:
                    metadata = doc.metadata  # Extract metadata dictionary
                    
                    # Display document metadata
                    docs_expander.markdown(f"📄 **Dokument:** `{metadata.get('doc_id', 'Unbekannt')}`")
                    docs_expander.markdown(f"📂 **Ordner:** `{metadata.get('folder', 'Unbekannt')}`")
                    docs_expander.markdown(f"📜 **Quelle:** `{metadata.get('source', 'Unbekannt')}`")
                    docs_expander.markdown(f"📑 **Seite:** `{metadata.get('page_number', 'Unbekannt')}`")

                    # Display content preview
                    docs_expander.markdown("**🔍 Relevanter Ausschnitt:**")
                    docs_expander.markdown(doc.page_content)

                    # Add a horizontal divider for better separation
                    docs_expander.divider()

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