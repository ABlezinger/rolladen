import streamlit as st
import os
import json
import mimetypes
from pswd import verify_password
from rag.vector_store_management import (
    extend_existing_vector_store, 
    check_vector_store_status, 
    OpenAIEmbeddingsWrapper,
    load_unified_vector_store,
    delete_document_from_vector_store
)
from pathlib import Path
from langchain_community.vectorstores import Chroma
import math
from typing import List, Dict, Any
import datetime


class DatabaseFileExplorer:
    """
    File explorer for viewing and managing vector database contents.
    Based on StreamlitFileManager but adapted for virtual file paths.
    """
    def __init__(
        self,
        file_paths: set,
        client,
        persist_directory: str,
        embedding_model: str,
        skip_prefix: str = "drive_download_combined",
        key_prefix: str = "db_explorer_",
        items_per_page_options: List[int] = [25, 50, 100, 200]
    ):
        self.raw_file_paths = file_paths
        self.client = client
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.skip_prefix = skip_prefix
        self.key_prefix = key_prefix
        self.items_per_page_options = items_per_page_options
        if os.path.exists(persist_directory):
            self.doc_validity = json.load(open(f"{self.persist_directory}/doc_validity.json"))  # To store validity info for documents
        else:
            self.doc_validity = {}
        
        # Build virtual file tree
        self.file_tree = self._build_virtual_tree()
        self._init_session_state()
    
    def _build_virtual_tree(self) -> Dict[str, Any]:
        """Build a virtual directory tree from database file paths."""
        tree = {}
        
        for file_path in self.raw_file_paths:
            path_str = str(file_path)
            
            # Skip the prefix and get relative path
            if self.skip_prefix and self.skip_prefix in path_str:
                relative_path = path_str.split(self.skip_prefix + "/", 1)[-1]
            elif "uploads" in path_str:
                # Handle uploads folder - extract from uploads onwards
                parts = Path(path_str).parts
                try:
                    uploads_idx = parts.index("uploads")
                    relative_path = str(Path(*parts[uploads_idx:]))
                except ValueError:
                    relative_path = path_str
            else:
                relative_path = path_str
            
            # Build tree structure
            parts = Path(relative_path).parts
            current = tree
            
            # Navigate/create folders
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {"__type__": "folder", "__children__": {}}
                current = current[part]["__children__"]
            
            # Add file
            if parts:
                filename = parts[-1]
                current[filename] = {
                    "__type__": "file",
                    "__full_path__": file_path,
                    "__size__": 0  # We don't have size info from vector DB
                }
        
        return tree
    
    def _get_state_key(self, key: str) -> str:
        """Generate a unique session state key."""
        return f"{self.key_prefix}{key}"
    
    def _init_session_state(self):
        """Initialize session state variables."""
        state_vars = {
            'current_path_parts': [],  # List of path parts from root
            'current_page': 1,
            'items_per_page': 25,
        }
        
        for key, default_value in state_vars.items():
            state_key = self._get_state_key(key)
            if state_key not in st.session_state:
                st.session_state[state_key] = default_value
    
    def _get_current_directory(self) -> Dict[str, Any]:
        """Navigate to current directory in the tree."""
        current = self.file_tree
        path_parts = st.session_state[self._get_state_key('current_path_parts')]
        
        for part in path_parts:
            if part in current and current[part].get("__type__") == "folder":
                current = current[part]["__children__"]
            else:
                # Path doesn't exist, reset to root
                st.session_state[self._get_state_key('current_path_parts')] = []
                return self.file_tree
        
        return current
    
    def _get_items_in_current_directory(self) -> List[Dict[str, Any]]:
        """Get list of items in current directory."""
        current_dir = self._get_current_directory()
        items = []
        
        for name, data in current_dir.items():
            if name.startswith("__"):
                continue
            
            is_directory = data.get("__type__") == "folder"
            items.append({
                'name': name,
                'is_directory': is_directory,
                'size': data.get("__size__", 0) if not is_directory else 0,
                'full_path': data.get("__full_path__", ""),
                'valid_from': self.doc_validity.get(data.get("__full_path__", ""), {}).get("valid_from", "Unbekannt") if not is_directory else None,
                'valid_to': self.doc_validity.get(data.get("__full_path__", ""), {}).get("valid_to", "Unbekannt") if not is_directory else None,
            })
        
        # Sort: folders first, then files, alphabetically
        items.sort(key=lambda x: (not x['is_directory'], x['name'].lower()))
        return items
    
    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        if size == 0:
            return "--"
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def _get_current_path_display(self) -> str:
        """Get current path as a display string."""
        path_parts = st.session_state[self._get_state_key('current_path_parts')]
        if not path_parts:
            return "🏠 Root"
        return "🏠 Root / " + " / ".join(path_parts)
    
    def _delete_file(self, file_path: str) -> tuple[bool, str]:
        """
        Delete a file from both the filesystem and vector store.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Delete from vector store first
            embeddings = OpenAIEmbeddingsWrapper(self.client, self.embedding_model)
            success, deleted_count, error_msg = delete_document_from_vector_store(
                self.persist_directory, 
                embeddings, 
                file_path
            )
            
            if not success:
                return False, f"Fehler beim Löschen aus Vektorstore: {error_msg}"
            
            # Delete physical file if it exists
            if os.path.exists(file_path):
                os.remove(file_path)
                return True, f"Erfolgreich gelöscht: Datei und {deleted_count} Chunks aus Vektorstore entfernt"
            else:
                return True, f"Erfolgreich aus Vektorstore gelöscht ({deleted_count} Chunks). Physische Datei nicht gefunden."
                
        except Exception as e:
            return False, f"Fehler beim Löschen: {str(e)}"
    
    def _render_pagination(self, total_items: int):
        """Render pagination controls."""
        total_pages = math.ceil(total_items / st.session_state[self._get_state_key('items_per_page')])
        
        if total_pages > 1:
            col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])
            current_page = st.session_state[self._get_state_key('current_page')]
            
            with col1:
                if st.button("⏮️", disabled=current_page == 1, key=f"{self.key_prefix}first"):
                    st.session_state[self._get_state_key('current_page')] = 1
                    st.rerun()
            
            with col2:
                if st.button("◀️", disabled=current_page == 1, key=f"{self.key_prefix}prev"):
                    st.session_state[self._get_state_key('current_page')] -= 1
                    st.rerun()
            
            with col3:
                st.write(f"Seite {current_page} von {total_pages}")
            
            with col4:
                if st.button("▶️", disabled=current_page == total_pages, key=f"{self.key_prefix}next"):
                    st.session_state[self._get_state_key('current_page')] += 1
                    st.rerun()
            
            with col5:
                if st.button("⏭️", disabled=current_page == total_pages, key=f"{self.key_prefix}last"):
                    st.session_state[self._get_state_key('current_page')] = total_pages
                    st.rerun()
    def file_icon(self, name: str):
        n = name.lower()
        if n.endswith(".pdf"):
            return "📕"
        if n.endswith((".docx", ".doc")):
            return "📘"
        if n.endswith((".xlsx", ".xls")):
            return "📗"
        return "📄"


    def fmt(self, val):
        return str(val).replace("unknown", "--")
    
    def render(self):
        """Render the file explorer component."""
        # Custom styling
        st.html("""
            <style>
                .st-key-db_file_explorer_container {
                    padding: unset;
                    gap: unset;
                }
                .st-key-db_file_explorer_container .stButton button {
                    padding: unset;
                    border: 0px;
                }
                .st-key-db_file_explorer_container .stButton button:active {
                    padding: unset;
                    border: 0px;
                    background-color: unset;
                    color: unset;
                }
                .st-key-db_file_explorer_container hr {
                    margin-top: 15px;
                }
            </style>
        """)
        
        with st.container(border=True, key=f"{self.key_prefix}file_explorer_container"):
            # Top bar with items per page
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"**{self._get_current_path_display()}**")
            
            with col2:
                items_per_page = st.selectbox(
                    "Items pro Seite",
                    options=self.items_per_page_options,
                    index=0,
                    key=f"{self.key_prefix}items_per_page_selector",
                    label_visibility="collapsed"
                )
                if items_per_page != st.session_state[self._get_state_key('items_per_page')]:
                    st.session_state[self._get_state_key('items_per_page')] = items_per_page
                    st.session_state[self._get_state_key('current_page')] = 1
                    st.rerun()
            
            st.divider()
            
            # Up button (only if not at root)
            if st.session_state[self._get_state_key('current_path_parts')]:
                if st.button('⬆️ Zurück', key=f"{self.key_prefix}up"):
                    st.session_state[self._get_state_key('current_path_parts')].pop()
                    st.session_state[self._get_state_key('current_page')] = 1
                    st.rerun()
                st.divider()
            
            # File/Folder List
            items = self._get_items_in_current_directory()
            
            # Pagination
            start_idx = (st.session_state[self._get_state_key('current_page')] - 1) * \
                       st.session_state[self._get_state_key('items_per_page')]
            end_idx = start_idx + st.session_state[self._get_state_key('items_per_page')]
            paginated_items = items[start_idx:end_idx]
            
            if not paginated_items:
                st.info("Dieser Ordner ist leer.")
            else:
                # ---------- header ----------
                st.markdown(
                    """
                    <style>
                    .file-row {
                        display: flex;
                        align-items: center;
                        padding: 6px 8px;
                        border-bottom: 1px solid #eee;
                        font-size: 14px;
                    }
                    .file-col-name { flex: 7; }
                    .file-col-size { flex: 2; }
                    .file-col-from { flex: 2; }
                    .file-col-to { flex: 2; }
                    .file-col-actions { flex: 3; display: flex; gap: 6px; }
                    
                    [data-testid="column"] {
                        padding: 0rem 0.3rem;
                    }

                    div.stButton > button {
                        padding: 2px 6px;
                        font-size: 12px;
                        height: 28px;
                        min-height: 28px;
                        line-height: 1;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                header = st.columns([7, 2, 2, 2, 3])
                header[0].markdown("**Dateiname**")
                header[1].markdown("**Größe**")
                header[2].markdown("**Gültig von**")
                header[3].markdown("**Gültig bis**")
                header[4].markdown("**Aktionen**")

                st.divider()
                # col1, col2, col3, col4, col5, col6 = st.columns([7, 2, 2, 2, 1, 1])
                # with col1:
                #     st.caption("Dateiname")
                # with col2:
                #     st.caption("Größe")
                # with col3:
                #     st.caption("Gültig von")
                # with col4:
                #     st.caption("Gütlig bis")
                # with col5:
                #     st.caption("Löschen")
                # with col6:
                #     st.caption("Herunterladen")
                # for idx, item in enumerate(paginated_items):
                    
                #     # col1, col2, col3, col4, col5, col6 = st.columns([7, 2, 2, 2, 1, 1])
                #     with col1:
                #         if item['is_directory']:
                #             if st.button(
                #                 f"📁 {item['name']}", 
                #                 key=f"{self.key_prefix}dir_{idx}_{item['name']}"):
                #                 st.session_state[self._get_state_key('current_path_parts')].append(item['name'])
                #                 st.session_state[self._get_state_key('current_page')] = 1
                #                 st.rerun()
                #         else:
                #             # File icon based on extension
                #             file_icon = "📄"
                #             if item['name'].lower().endswith('.pdf'):
                #                 file_icon = "📕"
                #             elif item['name'].lower().endswith(('.docx', '.doc')):
                #                 file_icon = "📘"
                #             elif item['name'].lower().endswith(('.xlsx', '.xls')):
                #                 file_icon = "📗"
                #             st.text(f"{file_icon} {item['name']}")
                    
                #     with col2:
                #         if not item['is_directory']:
                #             st.text(self._format_size(item['size']))
                            
                #     with col3:
                #         if not item['is_directory']:
                #             # Edit button for files only (functionality can be implemented as needed)
                #             text = f"{item['valid_from']}".replace("unknown", "--")
                #             st.caption(body=text)
                    
                #     with col4:
                #         if not item['is_directory']:
                #             text = f"{item['valid_to']}".replace("unknown", "--")
                #             st.caption(body=text)
                #     with col5:
                #         if not item['is_directory'] and item['full_path']:
                #             # Delete button for files only
                #             if st.button('🗑️', key=f"{self.key_prefix}del_{idx}_{item['name']}", help="Datei löschen"):
                #                 # Confirmation in a separate dialog using session state
                #                 st.session_state[self._get_state_key('delete_confirm_file')] = item['full_path']
                #                 st.session_state[self._get_state_key('delete_confirm_name')] = item['name']
                #                 st.rerun()
                                
                #     with col6:
                #         if not item['is_directory']:
                #             print(item)
                #             source_path = item['full_path']
                #             if source_path and os.path.isfile(source_path):
                #                 try:
                #                     with open(source_path, "rb") as file_handle:
                #                         file_bytes = file_handle.read()

                #                     mime_type = mimetypes.guess_type(source_path)[0] or "application/octet-stream"
                #                     st.download_button(
                #                         label="⬇️",
                #                         data=file_bytes,
                #                         file_name=os.path.basename(source_path),
                #                         help="Datei herunterladen",
                #                         mime=mime_type,
                #                         key=f"download_{item.get('name', idx)}",
                #                     )
                #                 except Exception as e:
                #                     st.download_button(
                #                         label="⬇️", 
                #                         help=f"Download nicht verfügbar: {str(e)}",
                #                         data="", 
                #                         key=f"download_error_{idx}_{item['name']}",
                #                         disabled=True)
                                    
                #     st.divider()
                
                for idx, item in enumerate(paginated_items):

                    is_dir = item["is_directory"]
                    name = item["name"]
                    icon = "📁" if is_dir else self.file_icon(name)

                    col1, col2, col3, col4, col5 = st.columns([7, 2, 2, 2, 3])

                    # ---------- NAME / NAVIGATION ----------
                    with col1:
                        if is_dir:
                            if st.button(f"{icon} {name}", key=f"open_{idx}"):
                                st.session_state[self._get_state_key("current_path_parts")].append(name)
                                st.session_state[self._get_state_key("current_page")] = 1
                                st.rerun()
                        else:
                            st.write(f"{icon} {name}")

                    # ---------- SIZE ----------
                    with col2:
                        if not is_dir:
                            st.write(self._format_size(item["size"]))

                    # ---------- VALID FROM ----------
                    with col3:
                        if not is_dir:
                            valid_from = item.get("valid_from", 15000101)  # Default to a very old date if not set
                            date_string = f"{datetime.datetime.strptime(str(valid_from), '%Y%m%d').strftime('%d.%m.%Y')}" if valid_from != 15000101 else "--"
                            st.write(date_string)

                    # ---------- VALID TO ----------
                    with col4:
                        if not is_dir:
                            valid_to = item.get("valid_to", 99991231)  # Default to a far future date if not set
                            to_date_string = f"{datetime.datetime.strptime(str(valid_to), '%Y%m%d').strftime('%d.%m.%Y')}" if valid_to != 99991231 else "--"
                            st.write(to_date_string)

                    # ---------- ACTIONS ----------
                    with col5:
                        if not is_dir:
                            st.markdown(
                                "<div style='display:flex; gap:4px; align-items:center;'>",
                                unsafe_allow_html=True
                            )

                            # DELETE
                            if st.button("🗑️", key=f"del_{idx}", help="Löschen"):
                                st.session_state[self._get_state_key("delete_confirm_file")] = item["full_path"]
                                st.session_state[self._get_state_key("delete_confirm_name")] = name
                                st.rerun()

                            # DOWNLOAD
                            path = item.get("full_path")
                            if path and os.path.isfile(path):
                                try:
                                    with open(path, "rb") as f:
                                        data = f.read()

                                    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"

                                    st.download_button(
                                        "⬇️",
                                        data=data,
                                        file_name=os.path.basename(path),
                                        mime=mime,
                                        key=f"dl_{idx}"
                                    )

                                except Exception:
                                    st.caption("❌")
                        st.markdown("</div>", unsafe_allow_html=True)

            # Handle deletion confirmation
            if self._get_state_key('delete_confirm_file') in st.session_state:
                file_to_delete = st.session_state[self._get_state_key('delete_confirm_file')]
                file_name = st.session_state[self._get_state_key('delete_confirm_name')]
                
                st.divider()
                st.warning(f"⚠️ Möchten Sie die Datei **{file_name}** wirklich löschen?")
                st.caption("Diese Aktion löscht die Datei und alle zugehörigen Chunks aus dem Vektorstore.")
                
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button("✅ Ja, löschen", key=f"{self.key_prefix}confirm_delete", type="primary"):
                        success, message = self._delete_file(file_to_delete)
                        
                        if success:
                            st.success(message)
                            # Clear confirmation state
                            del st.session_state[self._get_state_key('delete_confirm_file')]
                            del st.session_state[self._get_state_key('delete_confirm_name')]
                            
                            # Reload vector store in session state
                            try:
                                st.session_state.vector_stores = load_unified_vector_store(self.persist_directory)
                                if st.session_state.vector_stores is None:
                                    embeddings = OpenAIEmbeddingsWrapper(self.client, self.embedding_model)
                                    st.session_state.vector_stores = Chroma(
                                        persist_directory=self.persist_directory,
                                        embedding_function=embeddings
                                    )
                            except Exception as e:
                                st.warning(f"Hinweis: Vektorstore konnte nicht neu geladen werden: {str(e)}")
                            
                            # Clear cache to refresh the view
                            if "db_file_list" in st.session_state:
                                del st.session_state.db_file_list
                            if "db_metadata_count" in st.session_state:
                                del st.session_state.db_metadata_count
                            # Force rebuild of tree
                            keys_to_remove = [k for k in st.session_state.keys() if k.startswith("db_explorer_")]
                            for key in keys_to_remove:
                                if key not in [self._get_state_key('delete_confirm_file'), self._get_state_key('delete_confirm_name')]:
                                    del st.session_state[key]
                            st.rerun()
                        else:
                            st.error(message)
                            del st.session_state[self._get_state_key('delete_confirm_file')]
                            del st.session_state[self._get_state_key('delete_confirm_name')]
                
                with col2:
                    if st.button("❌ Abbrechen", key=f"{self.key_prefix}cancel_delete"):
                        del st.session_state[self._get_state_key('delete_confirm_file')]
                        del st.session_state[self._get_state_key('delete_confirm_name')]
                        st.rerun()
            
            st.divider()
            
            # Show item count and pagination
            st.caption(f"Gesamt: {len(items)} Element(e)")
            self._render_pagination(len(items))
            
# add near top of file
@st.fragment
def _render_db_explorer_fragment(file_paths, client, persist_directory, embedding_model, skip_prefix, key_prefix):
    with st.expander(f"📂 Datei-Explorer ({len(file_paths)} Dateien)", expanded=False):
        st.caption("💡 Die Ansicht zeigt Ordner ab 'drive_download_combined' + den 'uploads' Ordner")
        explorer = DatabaseFileExplorer(
            file_paths=file_paths,
            client=client,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            skip_prefix=skip_prefix,
            key_prefix=key_prefix
        )
        explorer.render()


def run_file_management(client, persist_directory="kisski_db_v3", embedding_model="qwen3-embedding-4b", skip_prefix="drive_download_combined"):
    if skip_prefix is None:
        # Do not skip any prefixes
        skip_prefix = None

    # Add custom styling for login/logout buttons and error messages
    st.html("""
    <style>
        /* Style for login button */
        button[data-testid="baseButton-secondary"][aria-label="Anmelden"] {
            color: black !important;
        }
        
        /* Style for logout button */
        button[data-testid="baseButton-secondary"][aria-label="Abmelden"] {
            color: black !important;
        }
        
        /* Alternative selector using button text content */
        div[data-testid="column"] button:has-text("Anmelden"),
        button:has-text("🔓 Abmelden") {
            color: black !important;
        }
        
        /* More reliable selector - target buttons by their parent structure */
        .stButton > button {
            color: black !important;
        }
        
        /* Make error and warning messages wider */
        [data-testid="stAlert"] {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Ensure alert content containers are also wide */
        [data-testid="stAlert"] > div {
            width: 100% !important;
        }
    </style>
    """)
    # Initialize authentication state
    if "file_management_authenticated" not in st.session_state:
        st.session_state.file_management_authenticated = False
    
    # If not authenticated, show login form
    if not st.session_state.file_management_authenticated:
        st.title("🔒 Anmeldung erforderlich")
        st.markdown("Bitte geben Sie das Passwort ein, um auf die Datei-Upload-Funktion zuzugreifen.")
        
        # Store error/warning messages to display outside form
        error_message = None
        warning_message = None
        
        # Use form to enable Enter key submission
        with st.form("login_form", clear_on_submit=False):
            password = st.text_input(
                "Passwort:",
                type="password",
                key="file_management_password_input"
            )
            
            col1, _ = st.columns([1, 4])
            with col1:
                submitted = st.form_submit_button("Anmelden", use_container_width=True)
            
            if submitted:
                if password:
                    if verify_password(password):
                        st.session_state.file_management_authenticated = True
                        st.rerun()
                    else:
                        error_message = "❌ Falsches Passwort. Bitte versuchen Sie es erneut."
                else:
                    warning_message = "⚠️ Bitte geben Sie ein Passwort ein."
        
        # Display error/warning messages outside the form for full width
        if error_message:
            st.error(error_message)
        if warning_message:
            st.warning(warning_message)
        
        st.stop()
    
    # If authenticated, show file management interface
    st.title("Datei-Upload (Word & PDF)")
    
    # Logout button
    if st.button("🔓 Abmelden", key="file_management_logout_button"):
        st.session_state.file_management_authenticated = False
        st.rerun()
    
    st.divider()
    
    # Database file explorer section
    st.subheader("🗂️ Datenbank-Übersicht")
    st.markdown("Hier können Sie sehen, welche Dateien aktuell in der Vektordatenbank gespeichert sind.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔄 Datenbank aktualisieren", key="refresh_db_status"):
            # Clear any cached database info
            if "db_file_list" in st.session_state:
                del st.session_state.db_file_list
            if "db_metadata_count" in st.session_state:
                del st.session_state.db_metadata_count
            # Clear explorer session state
            keys_to_remove = [k for k in st.session_state.keys() if k.startswith("db_explorer_")]
            for key in keys_to_remove:
                del st.session_state[key]
            st.rerun()
    
    # Load database status if not already loaded
    if "db_file_list" not in st.session_state or "db_metadata_count" not in st.session_state:
        with st.spinner("📊 Lade Datenbank-Status..."):
            try:
                embeddings = OpenAIEmbeddingsWrapper(client, embedding_model)
                existing_sources, metadata_count = check_vector_store_status(persist_directory, embeddings)
                st.session_state.db_file_list = existing_sources
                st.session_state.db_metadata_count = metadata_count
            except Exception as e:
                st.error(f"❌ Fehler beim Laden des Datenbank-Status: {str(e)}")
                st.session_state.db_file_list = set()
                st.session_state.db_metadata_count = 0
    
    # Display database statistics
    file_count = len(st.session_state.db_file_list)
    metadata_count = st.session_state.db_metadata_count
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📁 Gespeicherte Dateien", file_count)
    with col2:
        st.metric("📄 Dokument-Chunks", metadata_count)
    with col3:
        avg_chunks = metadata_count // file_count if file_count > 0 else 0
        st.metric("📊 Durchschn. Chunks/Datei", avg_chunks)
    
    # Display file explorer
    if file_count > 0:
            _render_db_explorer_fragment(st.session_state.db_file_list, client, persist_directory, embedding_model, skip_prefix, "db_explorer_")
    else:
        st.info("ℹ️ Die Datenbank ist leer. Laden Sie Dateien hoch und fügen Sie sie zum Vektorstore hinzu.")
    
    st.divider()
    st.subheader("📤 Neue Dateien hochladen")
    
    # Initialize file uploader key for clearing after upload
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0
    
    uploaded_files = st.file_uploader(
        "Bitte lade eine oder mehrere Word- (.docx) oder PDF-Dateien hoch (Diese Dateien liegen anschließend im Ordner 'uploads'):",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )
    
    if uploaded_files:
        # Show file preview
        st.info(f"📋 {len(uploaded_files)} Datei(en) ausgewählt:")
        
        with st.expander("📁 Ausgewählte Dateien anzeigen", expanded=True):
            for uploaded_file in uploaded_files:
                file_size = uploaded_file.size
                st.markdown(f"📄 **{uploaded_file.name}** ({file_size:,} Bytes)")
                col1, col2 = st.columns([1, 1])
                with col1:
                    valid_from = st.date_input(
                        f"Gültig von (optional)",
                        value=None,
                        key=f"valid_from_{uploaded_file.name}",
                        format="DD.MM.YYYY")
                    valid_from_int = int(valid_from.strftime("%Y%m%d")) if valid_from else 0
                with col2:
                    valid_to = st.date_input(
                        f"Gültig bis (optional)",
                        value=None,
                        key=f"valid_to_{uploaded_file.name}",
                        format="DD.MM.YYYY")
                    valid_to_int = int(valid_to.strftime("%Y%m%d")) if valid_to else 99991231
                
                # Validation
                if valid_from and valid_to and valid_from > valid_to:
                    st.error("'Gültig ab' darf nicht nach 'Gültig bis' liegen.")
        
        st.divider()
        
        # Single confirmation button
        st.markdown("**Die Dateien werden gespeichert und automatisch zum Chatbot-Vektorstore hinzugefügt.**")
        
        if st.button("✅ Dateien hochladen und zum Vektorstore hinzufügen", key="confirm_upload", type="primary", use_container_width=True):
            uploads_dir = "uploads"
            os.makedirs(uploads_dir, exist_ok=True)
            
            with st.status("🔄 Verarbeite Dateien...", expanded=True) as status:
                try:
                    # Step 1: Save files to disk
                    status.update(label="💾 Speichere Dateien...", state="running")
                    saved_files = []
                    saved_count = 0
                    
                    progress_bar = st.progress(0)
                    for idx, uploaded_file in enumerate(uploaded_files, 1):
                        # Handle potential filename conflicts
                        file_path = os.path.join(uploads_dir, uploaded_file.name)
                        counter = 1
                        base_name, ext = os.path.splitext(uploaded_file.name)
                        while os.path.exists(file_path):
                            file_path = os.path.join(uploads_dir, f"{base_name}_{counter}{ext}")
                            counter += 1
                        
                        # Write file to disk
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        saved_files.append(file_path)
                        saved_count += 1
                        
                        # Update progress
                        progress = int(idx * 50 / len(uploaded_files))  # 0-50%
                        progress_bar.progress(progress)
                    
                    st.success(f"✅ {saved_count} Datei(en) erfolgreich gespeichert.")
                    
                    # Step 2: Add to vector store
                    status.update(label="🤖 Füge Dateien zum Vektorstore hinzu...", state="running")
                    
                    vector_store = extend_existing_vector_store(
                        data_folder=uploads_dir,
                        persist_directory=persist_directory,
                        model=embedding_model
                    )
                    
                    progress_bar.progress(100)  # 100%
                    
                    if vector_store is not None:
                        status.update(
                            label="✅ Erfolgreich abgeschlossen!",
                            state="complete"
                        )
                        
                        # Reload vector store in session state
                        st.session_state.vector_stores = load_unified_vector_store(persist_directory)
                        
                        if st.session_state.vector_stores is None:
                            # Fallback: load directly
                            embeddings = OpenAIEmbeddingsWrapper(client, embedding_model)
                            st.session_state.vector_stores = Chroma(
                                persist_directory=persist_directory,
                                embedding_function=embeddings
                            )
                        
                        # Clear database cache to refresh explorer
                        if "db_file_list" in st.session_state:
                            del st.session_state.db_file_list
                        if "db_metadata_count" in st.session_state:
                            del st.session_state.db_metadata_count
                        keys_to_remove = [k for k in st.session_state.keys() if k.startswith("db_explorer_")]
                        for key in keys_to_remove:
                            del st.session_state[key]
                        
                        # Increment file uploader key to clear the widget
                        st.session_state.file_uploader_key += 1
                        
                        # Store files for display outside status context
                        st.session_state['upload_success_files'] = saved_files
                    else:
                        status.update(
                            label="⚠️ Fehler beim Aktualisieren des Vektorstores",
                            state="error"
                        )
                        # Increment file uploader key to clear the widget
                        st.session_state.file_uploader_key += 1
                        
                        # Store error info for display outside status context
                        st.session_state['upload_partial_success'] = saved_files
                        
                except Exception as e:
                    status.update(
                        label=f"❌ Fehler: {str(e)}",
                        state="error"
                    )
                    st.error(f"❌ Fehler beim Verarbeiten der Dateien: {str(e)}")
                    st.exception(e)
            
            # Display success/error messages outside status context to avoid nested expander error
            if 'upload_success_files' in st.session_state:
                st.success("✅ Die Dateien wurden erfolgreich hochgeladen und sind jetzt im Chatbot verfügbar!")
                
                st.markdown("**📁 Hochgeladene Dateien:**")
                for file_path in st.session_state['upload_success_files']:
                    st.markdown(f"  ✅ {os.path.basename(file_path)}")
                
                st.info("💡 Tipp: Aktualisieren Sie die Datenbank-Übersicht oben, um die neuen Dateien im Explorer zu sehen.")
                
                # Clear the success files after displaying
                del st.session_state['upload_success_files']
            
            elif 'upload_partial_success' in st.session_state:
                st.error("❌ Dateien wurden gespeichert, aber es gab einen Fehler beim Hinzufügen zum Vektorstore.")
                
                st.markdown("**📁 Gespeicherte Dateien:**")
                for file_path in st.session_state['upload_partial_success']:
                    st.markdown(f"  💾 {os.path.basename(file_path)}")
                
                st.warning("⚠️ Diese Dateien sind auf der Festplatte gespeichert, aber nicht im Chatbot verfügbar. Bitte kontaktieren Sie den Administrator.")
                
                # Clear the partial success files after displaying
                del st.session_state['upload_partial_success']

