import streamlit as st
import os
from src.rag.system_prompts import system_prompt
from src.llm_client import get_client, LLMServiceError, LLM_UNAVAILABLE_MESSAGE

from src.chatbot import run_chatbot
from src.filemanagement import run_file_management
from src.addons.file_chat import run_file_chat
from src.handoff import require_login, logout

st.set_page_config(page_title="R+S Auskunft", page_icon="assets/images/rsev_favicon.ico")
st.title("R+S Auskunft – ein DAISEC-Projekt")
st.logo("assets/images/rsev_x_daisec.png", size="large")

st.html("""
  <style>
    [alt=Logo] {
      margin-top: 3rem;
      height: 9rem;
    }
  </style>
        """)

# --- Authentication ---

user = require_login()

# --- NAVIGATION ---
st.sidebar.space("small")
with st.sidebar:
    st.write(f"Sie sind über den R+S-Login angemeledet.")

    if user.get("via") == "trusted_network":
        # Shared network identity: a logout would be undone on the next rerun,
        # so we don't offer one. Flag the reduced accountability instead.
        st.caption("Zugang über vertrauenswürdiges Netz (keine persönliche Anmeldung).")
    else:
        if st.button("Abmelden"):
            logout()
            
            
module_list = ["💬 Chatbot", "📄 Dokument-Chat", "📂 Datei-Upload"] if '0070001' in user.get("groups", []) else [
    "💬 Chatbot", "📄 Dokument-Chat"]
st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.radio(
    "Modus wählen:",
    module_list,
    horizontal=False
)

client = get_client(
    base_url=st.secrets["base_url"],#"https://chat.daisec.eu/api",
    api_key=st.secrets["KISSKI_API_KEY"]#st.secrets["DAISEC_API_KEY"]
)

# Initialize model and conversation if they aren't already in session state.
if "kisski_model" not in st.session_state:
    st.session_state["kisski_model"] = st.secrets["model_id"] #"qwen3:30b" #"deepseek-r1:32b"

# Initialize the base system prompt and store it separately.
if "base_system_prompt" not in st.session_state:
    st.session_state["base_system_prompt"] = system_prompt(role=st.secrets["role"])
    st.session_state["planning_prompt"] = system_prompt(role=st.secrets["role"]+ "_plan")
    # Also store this as the current system prompt.
    st.session_state["system_prompt"] = st.session_state["base_system_prompt"]

if "messages" not in st.session_state:
    st.session_state.messages = []

persist_directory = st.secrets["persist_directory"]

# Initialize vector stores if they aren't already in session state
if "vector_stores" not in st.session_state:
    try:
        from src.rag.vector_store_management import load_unified_vector_store, OpenAIEmbeddingsWrapper
        from langchain_community.vectorstores import Chroma
        

        # Try to load the unified vector store
        st.session_state.vector_stores = load_unified_vector_store(persist_directory)
        
        if st.session_state.vector_stores is None:
            # Fallback: try to create embeddings and load manually
            st.warning("⚠️ Could not load vector store automatically. Trying manual loading...")
            
            try:
                # Create embeddings manually
                openai_client = get_client(
                    api_key=st.secrets["KISSKI_API_KEY"],
                    base_url=st.secrets["base_url"]
                )
                embeddings = OpenAIEmbeddingsWrapper(openai_client, st.secrets["embedding_id"])

                # Try to load from main directory
                if os.path.exists(persist_directory) and os.listdir(persist_directory):
                    st.session_state.vector_stores = Chroma(persist_directory=persist_directory, embedding_function=embeddings) # type: ignore
                    st.success("✅ Vector store loaded manually!")
                else:
                    st.error("❌ No vector store found. Please run setup_vector_store.py first.")
                    st.stop()

            except LLMServiceError:
                st.error(LLM_UNAVAILABLE_MESSAGE)
                st.stop()
            except Exception as e:
                st.error(f"❌ Error loading vector store: {str(e)}")
                st.error("Please run: python setup_vector_store.py")
                st.stop()

    except LLMServiceError:
        st.error(LLM_UNAVAILABLE_MESSAGE)
        st.stop()
    except Exception as e:
        st.error(f"❌ Critical error initializing vector store: {str(e)}")
        st.stop()

# Get the selected vector store
vector_store = st.session_state.vector_stores

if page == "📂 Datei-Upload" and '0070001' in user.get("groups", []):
    run_file_management(client, persist_directory=persist_directory, embedding_model=st.secrets["embedding_id"], skip_prefix=st.secrets["main_directory"])
elif page == "📄 Dokument-Chat":
    run_file_chat(vector_store, client)
else:
    run_chatbot(vector_store, client)