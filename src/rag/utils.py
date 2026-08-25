from openai import OpenAI
import streamlit as st
import re
import io
from contextlib import redirect_stdout
import json

def rerank_chunks(query, retrieved_docs, reranker_model="phi4:14b"):
    """
    Uses a small model to determine which retrieved chunks best answer the query.
    Returns the top 2 most relevant chunks.
    """
    client = OpenAI(
        base_url="https://interweb.l3s.uni-hannover.de/v1",  
        api_key=st.secrets["INTERWEB_API_KEY"]
    )

     # **Debug retrieved documents**
    print("\n📌 **Retrieved Keyword Matches:**")
    for doc in retrieved_docs:
        metadata = doc.metadata
        print(f"🔹 **Dokument:** `{metadata.get('doc_id', 'Unbekannt')}`")
        print(f"📜 **Quelle:** `{metadata.get('source', 'Unbekannt')}`")
        print(f"📂 **Ordner:** `{metadata.get('folder', 'Unbekannt')}`")
        print(f"📑 **Seite:** `{metadata.get('page_number', 'Unbekannt')}`")
        print(f"📖 **Inhalt:** {doc.page_content[:300]}...")
        print("-" * 50)
    
    ranking_prompt = f"""
    Du bist ein KI-Experte für Dokumentensuche. Die Nutzerfrage ist:
    "{query}"

    Unten sind 10 Textausschnitte aus Dokumenten. Ordne sie nach Relevanz für die Frage.
    Das relevanteste erhält die Nummer 1, das unwichtigste die Nummer 10.

    Antworte NUR mit einer Liste von Zahlen (z. B. "1, 3, 5, 2, 4, 7, 8, 9, 6, 10").

    Texte:
    {chr(10).join([f"[{i+1}] {doc.page_content[:500]}" for i, doc in enumerate(retrieved_docs)])}
    """

    # Call the small model for re-ranking
    completion = client.chat.completions.create(
        model=reranker_model,  # Use a small model like "mistral-7b" or "phi-2:2.7b"
        messages=[{"role": "user", "content": ranking_prompt}],
        stream=False,
    )

    ranking_response = completion.choices[0].message.content
    ranked_indices = [int(num) - 1 for num in ranking_response.split(", ") if num.isdigit()]

    # Select the top 2 chunks based on the new ranking
    top_chunks = [retrieved_docs[i] for i in ranked_indices[:3]]

    return top_chunks

def hybrid_search(query, vector_store, k=10):
    """
    First tries a keyword search in stored documents.
    If no exact matches are found, falls back to vector similarity search.
    """
    stored_data = vector_store._collection.get()  # Fetch stored data (metadata + content)

    keyword_hits = []
    for i in range(len(stored_data["documents"])):
        text = stored_data["documents"][i]  # Extract stored text
        metadata = stored_data["metadatas"][i]  # Extract metadata

        # If query phrase is found inside text, add it to keyword_hits
        if query.lower() in text.lower():
            keyword_hits.append({
                "content": text,
                "metadata": metadata
            })

    if keyword_hits:
        from langchain_core.documents import Document
        
        # **Prioritize correct document & sort by page number**
        keyword_hits.sort(
            key=lambda x: (
                0 if "Kriterien für die Lektüreauswahl" in x["content"] else 1,  # Boost title matches
                x["metadata"].get("page_number", 9999)  # Prioritize lower page numbers
            )
        )

        retrieved_docs = [Document(page_content=hit["content"], metadata=hit["metadata"]) for hit in keyword_hits[:k]]

        # **Debug retrieved documents**
        print("\n📌 **Retrieved Keyword Matches:**")
        for doc in retrieved_docs:
            metadata = doc.metadata
            print(f"🔹 **Dokument:** `{metadata.get('doc_id', 'Unbekannt')}`")
            print(f"📜 **Quelle:** `{metadata.get('source', 'Unbekannt')}`")
            print(f"📂 **Ordner:** `{metadata.get('folder', 'Unbekannt')}`")
            print(f"📑 **Seite:** `{metadata.get('page_number', 'Unbekannt')}`")
            print(f"📖 **Inhalt:** {doc.page_content[:300]}...")
            print("-" * 50)

        return retrieved_docs

    # **2nd Pass: Semantic Search (Vector Similarity)**
    retrieved_docs = vector_store.similarity_search(query, k=k)

    # **Debug retrieved documents**
    print("\n📌 **Retrieved Embedding Matches:**")
    for doc in retrieved_docs:
        metadata = doc.metadata
        print(f"🔹 **Dokument:** `{metadata.get('doc_id', 'Unbekannt')}`")
        print(f"📜 **Quelle:** `{metadata.get('source', 'Unbekannt')}`")
        print(f"📂 **Ordner:** `{metadata.get('folder', 'Unbekannt')}`")
        print(f"📑 **Seite:** `{metadata.get('page_number', 'Unbekannt')}`")
        print(f"📖 **Inhalt:** {doc.page_content[:300]}...")
        print("-" * 50)

    return retrieved_docs


# List of common LaTeX-esque patterns (using raw strings)
COMMON_LATEX_PATTERNS = [
    r"\\sqrt\{[^}]+\}",   # matches \sqrt{...}
    r"\\times",           # matches \times
    r"\\approx",          # matches \approx
    r"\\frac\{[^}]+\}\{[^}]+\}",  # matches \frac{...}{...}
    # You can add more patterns if needed:
    # r"\\infty",       # matches \infty
    # r"\\pi",          # matches \pi
]

def extract_latex(text):
    """
    Extracts all LaTeX expressions from text, both inline and display math.
    Returns a list of LaTeX expressions with their positions.
    """
    # Match both inline math \( ... \) and display math \[ ... \]
    inline_pattern = r"\\\((.*?)\\\)"
    display_pattern = r"\\\[(.*?)\\\]"
    
    inline_matches = [(m.group(1), m.start(), m.end(), "inline") 
                     for m in re.finditer(inline_pattern, text, re.DOTALL)]
    display_matches = [(m.group(1), m.start(), m.end(), "display") 
                      for m in re.finditer(display_pattern, text, re.DOTALL)]
    
    # Combine all matches and sort by position
    all_matches = sorted(inline_matches + display_matches, key=lambda x: x[1])
    return all_matches

def render_text_with_math(text):
    """
    Renders text with LaTeX math expressions properly using an inline approach.
    Processes the entire text as one unit, preserving paragraph structure
    and rendering math inline wherever possible.
    """
    # Process LaTeX delimiters to make them compatible with Streamlit's markdown
    # Convert display math \[ ... \] to $$ ... $$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$ \1 $$', text, flags=re.DOTALL)
    
    # Convert inline math \( ... \) to $ ... $
    text = re.sub(r'\\\((.*?)\\\)', r'$ \1 $', text, flags=re.DOTALL)
    
    # Add special handling for common LaTeX commands that might not render well
    # Convert \frac{}{} to the more compatible form if needed
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\\frac{\1}{\2}', text)
    
    # Use Streamlit's markdown which supports LaTeX with $ delimiters
    st.markdown(text)

def extract_thinking(text):
    """
    Extracts content between <think> and </think> tags.
    Returns the extracted text if found; otherwise, returns None.
    """
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_code(text):
    """
    Extracts a Python code block (enclosed in triple backticks) from the given text.
    Returns the code as a string, or None if not found.
    """
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None

def execute_code(code):
    """
    Executes the given Python code and captures its output.
    Returns the captured output or an error message if something goes wrong.
    """
    stdout = io.StringIO()
    try:
        with redirect_stdout(stdout):
            # Caution: using exec is risky; ensure that the code is trusted or sandboxed.
            exec(code, {})
    except Exception as e:
        return f"Error during execution: {e}"
    return stdout.getvalue()

def export_chat_history(chat_history):
    """
    Exports chat history to a JSON string.
    
    Args:
        chat_history: List of message dictionaries with 'role' and 'content' keys.
    
    Returns:
        JSON string representation of the chat history.
    """
    return json.dumps(chat_history, ensure_ascii=False, indent=2)

def import_chat_history(json_string):
    """
    Imports chat history from a JSON string.
    
    Args:
        json_string: JSON string containing chat history.
    
    Returns:
        List of message dictionaries, or None if parsing fails.
    """
    try:
        chat_history = json.loads(json_string)
        # Validate structure
        if isinstance(chat_history, list):
            for msg in chat_history:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    return None
                # Ensure metadata exists (for backward compatibility)
                if "metadata" not in msg:
                    msg["metadata"] = {}
            return chat_history
        return None
    except json.JSONDecodeError:
        return None