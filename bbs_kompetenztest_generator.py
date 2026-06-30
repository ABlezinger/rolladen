import streamlit as st
from openai import OpenAI
import re
from system_prompts import system_prompt
from rag.llama_guard import check_safety_llama_guard_3
from pdf_generator import run_pdf_generator
from pdf_generator import run_pdf_download

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

def run_kompetenztest_generator(vector_store, client):
    st.title("🎯 Kompetenztest-Generator")
    st.sidebar.markdown(
        "👋 **Willkommen beim BBS Kompetenztest-Generator!**\n\n"
        "Dieser Generator hilft dir dabei, Kompetenztests zu erstellen, "
        "um dich für den Unterricht vorzubereiten. Lege fest, welches Themengebiet und "
        "welchen Umfang der Kompetenztest haben soll – ich helfe dir gerne weiter! 📚✨"
    )
    # Auswahl des Bereiches und des Fachs
    col1, col2 = st.columns(2)
    # Bereichsauswahl
    with col1:
        bereiche = {
            "Sprachen": ["Deutsch", "Englisch"],
            "Technik": ["Elektrotechnik", "Fahrzeugtechnik", "Metalltechnik"],
            "Mathematik": ["Mathematik (allgemein)", "Mathematik (Technik)"]
        }
        selected_bereich = st.selectbox("Bereich auswählen", list(bereiche.keys()))
    # Fachauswahl
    with col2:
        faecher = bereiche[selected_bereich]
        selected_fach = st.selectbox("Fach auswählen", faecher)
    
    # Aufgabentypen definieren
    aufgabentypen_mapping = {
        "Deutsch": ["Offene Frage", "Multiple Choice (einzeln)", "Multiple Choice (mehrfach)"],
        "Englisch": ["Offene Frage", "Multiple Choice (einzeln)", "Multiple Choice (mehrfach)"],
        "Elektrotechnik": ["Rechenaufgabe", "Offene Frage", "Multiple Choice (einzeln)", "Multiple Choice (mehrfach)"],
        "Fahrzeugtechnik": ["Rechenaufgabe", "Offene Frage", "Multiple Choice (einzeln)", "Multiple Choice (mehrfach)"],
        "Metalltechnik": ["Rechenaufgabe", "Offene Frage", "Multiple Choice (einzeln)", "Multiple Choice (mehrfach)"],
        "Mathematik (allgemein)": ["Rechenaufgabe", "Offene Frage", "Multiple Choice (einzeln)", "Multiple Choice (mehrfach)"],
        "Mathematik (Technik)": ["Rechenaufgabe", "Offene Frage", "Multiple Choice (einzeln)", "Multiple Choice (mehrfach)"]
    }

    # Konfiguration
    col1, col2 = st.columns(2)
    with col1:
        aufgabenanzahl = st.slider("Aufgabenanzahl", 3, 10)
    #with col2:
    #    niveau = st.select_slider("Schwierigkeitsgrad", ["leicht", "mittel", "schwer"])
    with col2:
        bearbeitungsdauer = st.slider("Bearbeitungsdauer", 15, 90, 45, 15)

    st.divider()

    # Liste für die gesammelten Aufgaben erstellen
    if 'aufgaben' not in st.session_state:
        st.session_state.aufgaben = []

    default_types = aufgabentypen_mapping.get(selected_fach, ["Multiple Choice"])

    aufgaben = []
    for i in range(aufgabenanzahl):
        # Zweispaltiges Layout für Typ und Niveau
        col1, col2 = st.columns(2)
        
        with col1:
            typ = st.selectbox(
                f"Aufgabentyp {i+1}",
                options=default_types,
                index=0,
                key=f"aufgabe_{i}_typ"
            )
        
        with col2:
            niveau = st.select_slider(
                f"Schwierigkeitsgrad {i+1}",
                options=["leicht", "mittel", "schwer"],
                value="mittel",
                key=f"aufgabe_{i}_niveau"
            )
        
        aufgaben.append({
            "Aufgabennummer": i+1,
            "Typ": typ,
            "Niveau": niveau
        })
    
    # Alte Aufgaben bereinigen wenn Anzahl reduziert wird
    st.session_state.aufgaben = aufgaben[:aufgabenanzahl]
    aufgaben = st.session_state.aufgaben

    st.divider()

    # Individuelle Themen
    col1, col2 = st.columns(2)
    agree = st.checkbox("Individuelle Themen")
    if agree:
        vorschlag = st.text_input("Individuelle Themen", "")
        # Sicherheitscheck nur bei aktiviertem Themenfeld
        if vorschlag.strip():
            with st.status("🔒 Führe Sicherheitsüberprüfung der Themen durch...", expanded=True) as status:
                is_safe, explanation = check_safety_llama_guard_3(vorschlag)
                if not is_safe:
                    status.update(label="⚠️ Sicherheitswarnung", state="error")
                    st.error(f"Unsicherer Inhalt erkannt: {explanation}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Sicherheitsblockierung: {explanation}"
                    })
                    st.stop()
                status.update(label="✅ Themen sicher", state="complete")

    # Response-Handling
    full_response = ""
    thinking_text = ""
    in_thinking_block = False
    status_placeholder = st.empty()

    # Generierungsschritt
    if st.button("🎯 Kompetenztest generieren", type="primary"):

        try:
            # Systemprompt erstellen
            syst_prompt = system_prompt(
                role="bbs_kompetenztest_generator",
                fach=selected_fach,
                niveau=niveau
            )

            # ===== Relevante Dokumente abrufen =====
            with st.status("🔍 Suche relevante Informationen...", expanded=True) as status:
                try:
                    retrieved_docs = vector_store.similarity_search(syst_prompt, k=5)
                    print(f"Debug: Retrieved {len(retrieved_docs)} documents for kompetenztest query")
                    
                    if retrieved_docs:
                        for i, doc in enumerate(retrieved_docs):
                            print(f"Debug: Doc {i+1} - Folder: {doc.metadata.get('folder', 'Unknown')}, Source: {doc.metadata.get('source', 'Unknown')}")
                            print(f"Debug: Content preview: {doc.page_content[:100]}...")
                    else:
                        print("Debug: No documents retrieved for kompetenztest!")
                        
                    status.update(label=f"✅ Relevante Informationen gefunden", state="complete")
                except Exception as e:
                    print(f"Debug: Error in kompetenztest similarity search: {str(e)}")
                    retrieved_docs = []
                    status.update(label="⚠️ Fehler bei der Suche nach relevanten Informationen", state="error")

            context = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "Keine relevanten Dokumente gefunden."

            with st.status("📚 Kompetenztest wird erstellt...", expanded=True) as status:
                # Systemprompt mit Kontext anreichern
                system_prompt_with_context = (
                    syst_prompt +
                    "\n\n=== Kontext aus Dokumenten ===\n" +
                    context +
                    "\n=== Ende Kontext ===\n"
                )

                # Prompt mit individuellen Angaben pro Aufgabe erstellen
                task_descriptions = "\n".join(
                    [f"- Aufgabe {task['Aufgabennummer']}: Typ '{task['Typ']}', Schwierigkeit '{task['Niveau']}'" 
                    for task in aufgaben]
                )

                # Optional: Individuelle Themen einbinden
                themen_hinweis = ""
                if agree and vorschlag.strip():
                    themen_hinweis = f"\nIndividuelle Themen: {vorschlag.strip()}"
                
                # eigentlicher Request
                completion = client.chat.completions.create(
                    model="gemma-3-27b-it",#"gemma3:27b-it-q8_0",
                    messages=[
                        {"role": "system", "content": system_prompt_with_context},
                        {"role": "user", "content": f"""
                            Erstelle eine Klausur für {selected_fach} mit folgenden {aufgabenanzahl} Aufgaben:
                            {task_descriptions}
                            {themen_hinweis}
                            
                            Gesamte Bearbeitungsdauer: {bearbeitungsdauer}
                            
                            Bitte beachte:
                            1. Aufgaben genau in der vorgegebenen Reihenfolge
                            2. Schwierigkeitsgrad pro Aufgabe individuell umsetzen
                            3. Aufgabentypen genau beachten
                            """
                        }
                    ],
                    stream=True,
                    temperature=0.7 if niveau == "mittel" else 1.0 if niveau == "schwer" else 0.3
                )

                for chunk in completion:
                    content = _get_stream_content(chunk)
                    if content is None or content == "":
                        continue

                    # Denkprozess-Block erkennen und sammeln
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

                    # Alles andere ist Klausurtext
                    full_response += content

                # Statusänderung
                status.update(label="✅ Kompetenztest erstellt", state="complete")

            # Trennung von Aufgaben und Lösung
            match = re.split(r"##\s*Lösung", full_response, maxsplit=1, flags=re.IGNORECASE)
            aufgaben_content = match[0].strip()
            loesung_content = match[1].strip() if len(match) > 1 else ""

            with st.expander("📘 Aufgaben anzeigen", expanded=False):
                st.markdown(aufgaben_content)

            with st.expander("📘 Lösungen anzeigen", expanded=False):
                st.markdown(loesung_content)
        
        except Exception as e:
            status_placeholder.error(f"❌ Fehler: {str(e)}")

        # PDF-Generierung
        pdf_bytes = run_pdf_generator(aufgaben_content, client)
        run_pdf_download(pdf_bytes, selected_fach)

    st.stop()  # Beendet die Ausführung hier