import streamlit as st
import os
import json
import mimetypes
from src.pswd import verify_password
from src.rag.vector_store_management import (
    extend_existing_vector_store, 
    check_vector_store_status, 
    OpenAIEmbeddingsWrapper,
    load_unified_vector_store,
    load_doc_list,
    save_doc_list,
    delete_document_from_vector_store,
    sync_metadata
)
import time
from src.document_utils import DocListEntry, DOCUMENT_CLASSES, DOC_LIST_FILENAME, doc_list_lock, get_doc_class_info_text
from src import virtual_folders as vf
from src.virtual_folders import FolderError
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
        # doc_list.json holds the metadata (incl. virtual_path), folders.json the empty folders
        self.docs_info, self.folders = vf.load_state(persist_directory)

        # Build virtual file tree
        self.file_tree = self._build_virtual_tree()
        self._init_session_state()

    def _build_virtual_tree(self) -> Dict[str, Any]:
        """Build the virtual folder tree (virtual_path metadata, fallback: physical path)."""
        return vf.build_tree(self.docs_info, self.folders, self.skip_prefix, self.raw_file_paths)

    def _current_path(self) -> str:
        """Virtual path of the folder currently shown ("" = root)."""
        return "/".join(st.session_state[self._get_state_key('current_path_parts')])

    def _all_folders(self) -> List[str]:
        return sorted(vf.existing_folders(self.docs_info, self.folders, self.skip_prefix, self.raw_file_paths), key=str.lower)

    @staticmethod
    def _folder_label(path: str) -> str:
        return "🏠 Hauptordner" if path == vf.ROOT else f"📁 {path}"

    def _apply_folder_operation(self, operation, success_message: str, new_path_parts: List[str] | None = None) -> bool:
        """Run a virtual_folders operation on the on-disk state and refresh the explorer.

        operation(doc_list, folders, skip_prefix, sources) -> (doc_list, folders)
        Returns False (and shows the error) if the operation was rejected.
        """
        try:
            new_doc_list, _ = vf.locked_update(
                self.persist_directory,
                lambda doc_list, folders: operation(doc_list, folders, self.skip_prefix, self.raw_file_paths),
            )
        except FolderError as e:
            st.error(str(e))
            return False
        except Exception as e:
            st.error(f"Fehler beim Speichern der Ordnerstruktur: {e}")
            return False

        # keep the rest of the page (upload form, replace selection) in sync
        st.session_state.docs_info = new_doc_list
        if new_path_parts is not None:
            st.session_state[self._get_state_key('current_path_parts')] = new_path_parts
            st.session_state[self._get_state_key('current_page')] = 1
        st.toast(success_message, icon="✅")
        return True

    @st.dialog("Neuer Ordner")
    def new_folder_dialog(self, parent: str):
        st.caption(f"Wird angelegt in: {self._folder_label(parent)}")
        with st.form(key=f"{self.key_prefix}new_folder_form", border=False):
            name = st.text_input("Ordnername", max_chars=vf.MAX_SEGMENT_LENGTH)
            if st.form_submit_button("📁 Anlegen", type="primary", width="stretch"):
                if self._apply_folder_operation(
                    lambda d, f, skip, src: vf.create_folder(d, f, parent, name, skip, src),
                    f"Ordner '{name.strip()}' angelegt",
                ):
                    st.rerun()

    @st.dialog("Ordner umbenennen")
    def rename_folder_dialog(self, path: str):
        st.caption(f"Ordner: {self._folder_label(path)}")
        with st.form(key=f"{self.key_prefix}rename_folder_form", border=False):
            name = st.text_input("Neuer Name", value=vf.basename_of(path), max_chars=vf.MAX_SEGMENT_LENGTH)
            if st.form_submit_button("✏️ Umbenennen", type="primary", width="stretch"):
                try:
                    new_path = vf.join_path(vf.parent_of(path), name)
                except FolderError as e:
                    st.error(str(e))
                    return
                if self._apply_folder_operation(
                    lambda d, f, skip, src: vf.move_folder(d, f, path, new_path, skip, src),
                    f"Ordner umbenannt in '{vf.basename_of(new_path)}'",
                    new_path_parts=self._remap_current_path(path, new_path),
                ):
                    st.rerun()

    @st.dialog("Ordner verschieben")
    def move_folder_dialog(self, path: str):
        st.caption(f"Ordner: {self._folder_label(path)}")
        # every folder except the folder itself, its subfolders and its current parent
        targets = [vf.ROOT] + [f for f in self._all_folders() if not vf.is_within(f, path)]
        targets = [t for t in targets if t != vf.parent_of(path)]
        if not targets:
            st.info("Es gibt keinen anderen Zielordner.")
            return
        target = st.selectbox("Zielordner", targets, format_func=self._folder_label, key=f"{self.key_prefix}move_folder_target")
        if st.button("📂 Verschieben", type="primary", width="stretch", key=f"{self.key_prefix}move_folder_confirm"):
            try:
                new_path = vf.join_path(target, vf.basename_of(path))
            except FolderError as e:
                st.error(str(e))
                return
            if self._apply_folder_operation(
                lambda d, f, skip, src: vf.move_folder(d, f, path, new_path, skip, src),
                f"Ordner verschoben nach '{self._folder_label(target)}'",
                new_path_parts=self._remap_current_path(path, new_path),
            ):
                st.rerun()

    @st.dialog("Ordner löschen")
    def delete_folder_dialog(self, path: str):
        st.warning(f"Soll der Ordner **{path}** gelöscht werden?")
        st.caption("Es können nur leere Ordner gelöscht werden. Dateien werden dabei nie gelöscht.")
        if st.button("🗑️ Ordner löschen", type="primary", width="stretch", key=f"{self.key_prefix}delete_folder_confirm"):
            current = self._current_path()
            new_parts = vf.split_path(vf.parent_of(path)) if vf.is_within(current, path) else None
            if self._apply_folder_operation(
                lambda d, f, skip, src: vf.delete_folder(d, f, path, skip, src),
                f"Ordner '{vf.basename_of(path)}' gelöscht",
                new_path_parts=new_parts,
            ):
                st.rerun()

    @st.dialog("Datei verschieben")
    def move_file_dialog(self, source: str, title: str):
        current = vf.resolve_virtual_path(source, self.docs_info, self.skip_prefix)
        st.caption(f"Datei: **{title}**  \nAktueller Ordner: {self._folder_label(current)}")
        targets = [t for t in [vf.ROOT] + self._all_folders() if t != current]
        if not targets:
            st.info("Es gibt keinen anderen Zielordner. Legen Sie zuerst einen Ordner an.")
            return
        target = st.selectbox("Zielordner", targets, format_func=self._folder_label, key=f"{self.key_prefix}move_file_target")
        st.caption("Die Datei selbst und der Vektorstore bleiben unverändert, nur die Einordnung ändert sich.")
        if st.button("📂 Verschieben", type="primary", width="stretch", key=f"{self.key_prefix}move_file_confirm"):
            if self._apply_folder_operation(
                lambda d, f, skip, src: vf.move_file(d, f, source, target, skip, src),
                f"Datei verschoben nach '{self._folder_label(target)}'",
            ):
                st.rerun()

    def _request_dialog(self, kind: str, *args) -> None:
        """Open a dialog at the end of render(), outside of popovers and expanders."""
        st.session_state[self._get_state_key("pending_dialog")] = (kind, args)

    def _remap_current_path(self, old_path: str, new_path: str) -> List[str] | None:
        """New breadcrumb if the folder currently shown is (inside) a moved folder."""
        current = self._current_path()
        if current and vf.is_within(current, old_path):
            return vf.split_path(new_path + current[len(old_path):])
        return None

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
            if part in current["folders"]:
                current = current["folders"][part]
            else:
                # Path doesn't exist (anymore), reset to root
                st.session_state[self._get_state_key('current_path_parts')] = []
                return self.file_tree

        return current

    def _get_items_in_current_directory(self) -> List[Dict[str, Any]]:
        """Get list of items in current directory."""
        current_dir = self._get_current_directory()
        current_path = self._current_path()
        items = []

        for name in current_dir["folders"]:
            items.append({
                'name': name,
                'is_directory': True,
                'size': 0,
                'full_path': vf.join_path(current_path, name),  # virtual path of the folder
            })

        for source, name in current_dir["files"].items():
            info = self.docs_info.get(source, {})
            items.append({
                'name': name,
                'is_directory': False,
                'size': 0,  # We don't have size info from vector DB
                'full_path': source,
                'virtual_path': current_path,
                'has_doc_list_entry': source in self.docs_info,
                'valid_from': info.get("valid_from", 15000101),
                'valid_to': info.get("valid_to", 99991231),
                'downloadable': info.get("downloadable", False),
                'date_edited': info.get("date_edited", None),
                'edited_by': info.get("edited_by", None),
                'title': info.get("title", "No title"),
                'doc_class': info.get("doc_class", "Unbekannt"),
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
            size /= 1024 # type: ignore
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
                message = f"Erfolgreich gelöscht: Datei und {deleted_count} Chunks aus Vektorstore entfernt"
            else:
                message = f"Erfolgreich aus Vektorstore gelöscht ({deleted_count} Chunks). Physische Datei nicht gefunden."

            # Remove the doc_list entry; the (virtual) folder stays visible even if now empty
            def remove_entry(doc_list, folders):
                folder = vf.resolve_virtual_path(file_path, doc_list, self.skip_prefix)
                remaining = {source: entry for source, entry in doc_list.items() if source != file_path}
                kept_folders = set(folders) | {folder} if folder else set(folders)
                return remaining, kept_folders

            try:
                new_doc_list, _ = vf.locked_update(self.persist_directory, remove_entry)
                st.session_state.docs_info = new_doc_list
            except Exception as e:
                message += f" Hinweis: Eintrag in doc_list.json konnte nicht entfernt werden: {e}"
            return True, message

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
    
    def _show_date_field(self, value) -> str: 
        try:
            return datetime.datetime.strptime(str(value), "%Y%m%d").strftime("%d.%m.%Y")
        except:
            return "corrupted"
        
    @st.dialog("Delete File")
    def delete_file_dialog(self, file_to_delete: str, file_name: str):
        
        st.warning(f"⚠️ Möchten Sie die Datei **{file_name}** wirklich löschen?")
        st.caption("Diese Aktion löscht die Datei und alle zugehörigen Chunks aus dem Vektorstore.")
        
        col1, col2, = st.columns([1, 1])
        with col1:
            if st.button("✅ Ja, löschen", key=f"{self.key_prefix}confirm_delete", type="primary", width="stretch"):
                progr = st.progress(0.0, text="Löschen...")
                success, message = self._delete_file(file_to_delete)
                time.sleep(0.2)
                progr.progress(0.2, text="Löschen...")
                
                if success:
                    st.success(message)
                    time.sleep(0.2)
                    progr.progress(0.5, text="Datenbank laden...")
                    
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
                    progr.progress(1.0, text="Datei gelöscht")
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
            if st.button("❌ Abbrechen", key=f"{self.key_prefix}cancel_delete", width="stretch"):
                del st.session_state[self._get_state_key('delete_confirm_file')]
                del st.session_state[self._get_state_key('delete_confirm_name')]
                st.rerun()

    def render(self):
        """Render the file explorer component."""
        # Custom styling
        st.html(
            """
            <style>
                .st-key-db_explorer_file_explorer_container {
                    padding: 1rem;
                    gap: 2px;
                }
                .st-key-db_explorer_file_explorer_container .stButton button {
                    padding: unset;
                    border: 0px;
                }
                .st-key-db_explorer_file_explorer_container .stButton button:active {
                    padding: unset;
                    border: 0px;
                    background-color: unset;
                    color: unset;
                }
                .st-key-db_explorer_file_explorer_container .stDownloadButton button {
                    padding: unset;
                    border: 0px;
                }
                .st-key-db_explorer_file_explorer_container .stDownloadButton button:active {
                    padding: unset;
                    border: 0px;
                    background-color: unset;
                    color: unset;
                }
                .st-key-db_explorer_file_explorer_container hr {
                    margin-top: 2px;
                }
                .st-key-db_explorer_file_explorer_container [data-testid="stVerticalBlock"] > div {
                    gap: 0.1rem !important;
                }
                .st-key-db_explorer_file_explorer_container [data-testid="stVerticalBlockBorderWrapper"] [data-testid="column"] {
                    padding: 0rem 0.1rem;
                }
                .st-key-db_explorer_file_explorer_container [data-testid="stExpander"] {
                    margin-bottom: 0.2rem;
                }
                .st-key-db_explorer_file_explorer_container .streamlit-expanderHeader {
                    padding: 0.2rem 0.5rem;
                    min-height: 0;
                    margin: 0;
                }
                .st-key-db_explorer_file_explorer_container .streamlit-expanderContent {
                    padding-top: 0.15rem;
                    padding-bottom: 0.05rem;
                }
                .st-key-db_explorer_file_explorer_container .stButton {
                    margin: 0;
                }
                .st-key-db_explorer_file_explorer_container .stButton > button,
                .st-key-db_explorer_file_explorer_container .file-actions-row button {
                    padding: 0.1rem 0.3rem;
                    min-height: 2.2rem;
                    height: 2.2rem;
                    line-height: 1;
                    font-size: 1rem;
                    margin: 0;
                }
                .st-key-db_explorer_file_explorer_container .file-actions-row {
                    display: flex;
                    gap: 0.1rem;
                    align-items: center;
                }
                
                .st-key-db_explorer_confirm_delete button,
                .st-key-db_explorer_cancel_delete button {
                    height: 40px;
                    min-height: 40px;
                    padding: 8px 12px;
                    white-space: nowrap;
                }
            </style>
        """)
        
        with st.container(border=True, key=f"{self.key_prefix}file_explorer_container"):
            # Top bar: current path, new folder, items per page
            col1, col_new, col2 = st.columns([3, 1, 1], vertical_alignment="center")

            with col1:
                st.markdown(f"**{self._get_current_path_display()}**")

            with col_new:
                if st.button("📁 Neuer Ordner", key=f"{self.key_prefix}new_folder", help="Ordner im aktuellen Ordner anlegen"):
                    self._request_dialog("new_folder", self._current_path())

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

                    div.stButton:not(.st-key-db_explorer_confirm_delete)
                                :not(.st-key-db_explorer_cancel_delete) > button {
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
                # header[1].markdown("**Größe**")
                # header[2].markdown("**Gültig von**")
                # header[3].markdown("**Gültig bis**")
                # header[4].markdown("**Aktionen**")

                st.divider()
                
                for idx, item in enumerate(paginated_items):

                    is_dir = item["is_directory"]
                    name = item["name"]
                    icon = "📁" if is_dir else self.file_icon(name)

                    col1, col2, col3, col4, col5 = st.columns([7, 2, 2, 2, 3], vertical_alignment="center")
                    
                    if is_dir:
                        folder_path = item["full_path"]
                        c_open, c_actions = st.columns([12, 1], vertical_alignment="center")
                        with c_open:
                            if st.button(f"{icon} {name}", key=f"open_{idx}"):
                                st.session_state[self._get_state_key("current_path_parts")].append(name)
                                st.session_state[self._get_state_key("current_page")] = 1
                                st.rerun()
                        with c_actions:
                            with st.popover("⋯", help="Ordner bearbeiten", key=f"{self.key_prefix}folder_actions_{idx}"):
                                if st.button("Umbenennen", icon="✏️", key=f"{self.key_prefix}rename_folder_{idx}", width="stretch"):
                                    self._request_dialog("rename_folder", folder_path)
                                if st.button("Verschieben", icon="📂", key=f"{self.key_prefix}move_folder_{idx}", width="stretch"):
                                    self._request_dialog("move_folder", folder_path)
                                if st.button("Löschen", icon="🗑️", key=f"{self.key_prefix}delete_folder_{idx}", width="stretch"):
                                    self._request_dialog("delete_folder", folder_path)
                    else:
                        with st.expander(f"{icon} {item.get('title', name)}", expanded=False, type="default") as doc_expander:
                            with st.container() as doc_container:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    rows = [
                                        ("Titel:", self.fmt(item.get("title", "No title"))),
                                        ("Ordner:", self._folder_label(item.get("virtual_path", vf.ROOT))),
                                        ("Pfad:", item.get("full_path", "--")),
                                        ("Dokumentklasse:", f"{item.get('doc_class', 'n.a.')}"),
                                        ("Gültig von:", f"{self._show_date_field(str(item.get('valid_from', 15000101))) if item.get('valid_from', 15000101) != 15000101 else '--'}"),
                                        ("Gültig bis:", f"{self._show_date_field(str(item.get('valid_to', 99991231))) if item.get('valid_to', 99991231) != 99991231 else '--'}"),
                                        ("Herunterladbar:", "Ja" if item.get("downloadable", False) else "Nein"),
                                        ("Zuletzt bearbeitet:", f"{datetime.datetime.strptime(str(item.get('date_edited')), '%Y%m%d').strftime('%d.%m.%Y') if item.get('date_edited') else 'Unbekannt'}"),
                                        ("Bearbeitet von:", self.fmt(item.get("edited_by", "System"))),
                                    ]

                                    for label, value in rows:
                                        c1, c2 = st.columns([2, 4])
                                        with c1:
                                            st.caption(f"**{label}**")
                                        with c2:
                                            st.caption(value)
                                with col2:
                                    
                                    if item["name"] != "document_index":
                                        if st.button("Bearbeiten", key=f"edit_{idx}", icon="✏️", help="Bearbeiten", disabled=True, width="stretch"):
                                            st.warning("Bearbeiten ist derzeit deaktiviert.")
                                        if st.button("Löschen", key=f"del_{idx}", icon="🗑️", help="Löschen", width="stretch"):
                                            st.session_state[self._get_state_key("delete_confirm_file")] = item["full_path"]
                                            st.session_state[self._get_state_key("delete_confirm_name")] = name
                                            st.rerun()
                                        if st.button(
                                            "Verschieben", key=f"mv_{idx}", icon="📂", width="stretch",
                                            disabled=not item.get("has_doc_list_entry"),
                                            help="In einen anderen Ordner verschieben" if item.get("has_doc_list_entry")
                                            else "Kein Eintrag in doc_list.json, daher nicht verschiebbar",
                                        ):
                                            self._request_dialog("move_file", item["full_path"], self.fmt(item.get("title", name)))
                                        if st.download_button("Herunterladen", icon="📥", key=f"dl_{idx}", disabled=True, data="test"):
                                            st.toast("Download gestartet.", icon="📥")
            # Open a requested folder/move dialog (one per run; popped so a later rerun closes it)
            pending = st.session_state.pop(self._get_state_key("pending_dialog"), None)
            if pending:
                kind, args = pending
                # Streamlit allows only one dialog per run: drop a lingering delete confirmation
                st.session_state.pop(self._get_state_key('delete_confirm_file'), None)
                st.session_state.pop(self._get_state_key('delete_confirm_name'), None)
                {
                    "new_folder": self.new_folder_dialog,
                    "rename_folder": self.rename_folder_dialog,
                    "move_folder": self.move_folder_dialog,
                    "delete_folder": self.delete_folder_dialog,
                    "move_file": self.move_file_dialog,
                }[kind](*args)

            # Handle deletion confirmation
            if self._get_state_key('delete_confirm_file') in st.session_state:
                self.delete_file_dialog(file_to_delete=st.session_state[self._get_state_key('delete_confirm_file')], 
                                 file_name=st.session_state[self._get_state_key('delete_confirm_name')])
                # file_to_delete = st.session_state[self._get_state_key('delete_confirm_file')]
                # file_name = st.session_state[self._get_state_key('delete_confirm_name')]
                
                # st.divider()
                # st.warning(f"⚠️ Möchten Sie die Datei **{file_name}** wirklich löschen?")
                # st.caption("Diese Aktion löscht die Datei und alle zugehörigen Chunks aus dem Vektorstore.")
                
                # col1, col2, col3 = st.columns([1, 1, 3])
                # with col1:
                #     if st.button("✅ Ja, löschen", key=f"{self.key_prefix}confirm_delete", type="primary"):
                #         success, message = self._delete_file(file_to_delete)
                        
                #         if success:
                #             st.success(message)
                #             # Clear confirmation state
                #             del st.session_state[self._get_state_key('delete_confirm_file')]
                #             del st.session_state[self._get_state_key('delete_confirm_name')]
                            
                #             # Reload vector store in session state
                #             try:
                #                 st.session_state.vector_stores = load_unified_vector_store(self.persist_directory)
                #                 if st.session_state.vector_stores is None:
                #                     embeddings = OpenAIEmbeddingsWrapper(self.client, self.embedding_model)
                #                     st.session_state.vector_stores = Chroma(
                #                         persist_directory=self.persist_directory,
                #                         embedding_function=embeddings
                #                     )
                #             except Exception as e:
                #                 st.warning(f"Hinweis: Vektorstore konnte nicht neu geladen werden: {str(e)}")
                            
                #             # Clear cache to refresh the view
                #             if "db_file_list" in st.session_state:
                #                 del st.session_state.db_file_list
                #             if "db_metadata_count" in st.session_state:
                #                 del st.session_state.db_metadata_count
                #             # Force rebuild of tree
                #             keys_to_remove = [k for k in st.session_state.keys() if k.startswith("db_explorer_")]
                #             for key in keys_to_remove:
                #                 if key not in [self._get_state_key('delete_confirm_file'), self._get_state_key('delete_confirm_name')]:
                #                     del st.session_state[key]
                #             st.rerun()
                #         else:
                #             st.error(message)
                #             del st.session_state[self._get_state_key('delete_confirm_file')]
                #             del st.session_state[self._get_state_key('delete_confirm_name')]
                
                # with col2:
                #     if st.button("❌ Abbrechen", key=f"{self.key_prefix}cancel_delete"):
                #         del st.session_state[self._get_state_key('delete_confirm_file')]
                #         del st.session_state[self._get_state_key('delete_confirm_name')]
                #         st.rerun()
            
            st.divider()
            
            # Show item count and pagination
            st.caption(f"Gesamt: {len(items)} Element(e)")
            self._render_pagination(len(items))
            
def save_metadata_changes() -> None:
    pass
# add near top of file
@st.fragment
def _render_db_explorer_fragment(file_paths, client, persist_directory, embedding_model, skip_prefix, key_prefix):
    with st.container(border=True, key=f"{key_prefix}file_explorer_fragment"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"📂 Datei-Explorer ({len(file_paths)} Dateien)")
        with col2:
            with st.container(horizontal_alignment="right"):
                if st.session_state.get("metadata_changes") !=  {} or True:
                    if st.button("💾 Änderungen speichern", key=f"{key_prefix}save_metadata_changes"):
                        save_metadata_changes()
                    st.html(
                        f"""
                        <div style="
                            width: 100%; text-align: right; color: #ff0000; font-size: 0.8rem; line-height: 1.2;
                        ">
                            {len(st.session_state.get('metadata_changes', {}))}
                            ungespeicherte Änderungen
                        </div>
                        """
                    )
        st.caption("💡 Die Ordner sind virtuell: Anlegen, Umbenennen und Verschieben ändert nur die Einordnung im Explorer, nicht die Dateien selbst oder den Vektorstore.")
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
    # # Initialize authentication state
    # if "file_management_authenticated" not in st.session_state:
    #     st.session_state.file_management_authenticated = False
    
    # # If not authenticated, show login form
    # if not st.session_state.file_management_authenticated:
    #     st.title("🔒 Anmeldung erforderlich")
    #     st.markdown("Bitte geben Sie das Passwort ein, um auf die Datei-Upload-Funktion zuzugreifen.")
        
    #     # Store error/warning messages to display outside form
    #     error_message = None
    #     warning_message = None
        
    #     # Use form to enable Enter key submission
    #     with st.form("login_form", clear_on_submit=False):
    #         password = st.text_input(
    #             "Passwort:",
    #             type="password",
    #             key="file_management_password_input"
    #         )
            
    #         col1, _ = st.columns([1, 4])
    #         with col1:
    #             submitted = st.form_submit_button("Anmelden", use_container_width=True)
            
    #         if submitted:
    #             if password:
    #                 if verify_password(password):
    #                     st.session_state.file_management_authenticated = True
    #                     st.rerun()
    #                 else:
    #                     error_message = "❌ Falsches Passwort. Bitte versuchen Sie es erneut."
    #             else:
    #                 warning_message = "⚠️ Bitte geben Sie ein Passwort ein."
        
    #     # Display error/warning messages outside the form for full width
    #     if error_message:
    #         st.error(error_message)
    #     if warning_message:
    #         st.warning(warning_message)
        
    #     st.stop()
    
    # If authenticated, show file management interface
    st.title("Datei-Upload (Word & PDF)")
    
    if "docs_info" not in st.session_state:
        if os.path.exists(f"{persist_directory}/doc_list.json"):
            docs_info = json.load(open(f"{persist_directory}/doc_list.json"))
        else:
            docs_info = {}
        st.session_state.docs_info = docs_info

    # # Logout button
    # if st.button("🔓 Abmelden", key="file_management_logout_button"):
    #     st.session_state.file_management_authenticated = False
    #     st.rerun()
    
    st.divider()
    
    # Database file explorer section
    st.subheader("🗂️ Datenbank-Übersicht")
    st.markdown("Hier können Sie sehen, welche Dateien aktuell in der Vektordatenbank gespeichert sind.")
    
    if "metadata_changes" not in st.session_state:
        st.session_state.metadata_changes = {}
    
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

    if "doc_list_editor" not in st.session_state:
        st.session_state.doc_list_editor = "System"

    st.session_state.doc_list_editor = st.text_input(
        "Bearbeitet von",
        value=st.session_state.doc_list_editor,
        key="doc_list_editor_input",
    ).strip() or "System"

    # Target folder in the explorer (virtual only: files are always stored flat in 'uploads/')
    upload_doc_list, upload_folders = vf.load_state(persist_directory)
    folder_options = [vf.ROOT] + sorted(
        vf.existing_folders(upload_doc_list, upload_folders, skip_prefix, st.session_state.get("db_file_list", set())),
        key=str.lower,
    )
    default_folder = "uploads" if "uploads" in folder_options else vf.ROOT
    upload_target_folder = st.selectbox(
        "Zielordner im Datei-Explorer",
        folder_options,
        index=folder_options.index(default_folder),
        format_func=DatabaseFileExplorer._folder_label,
        key="upload_target_folder",
        help="Neue Ordner legen Sie oben im Datei-Explorer an. Die Einordnung lässt sich später jederzeit ändern.",
    )

    # Initialize file uploader key for clearing after upload
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0
    
    uploaded_files = st.file_uploader(
        "Bitte lade eine oder mehrere Word- (.docx) oder PDF-Dateien hoch (sie erscheinen anschließend im gewählten Zielordner):",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )
    # st.session_state.uploaded_files = uploaded_files  # Store in session state for access in other parts
    
    if uploaded_files:
        # Show file preview
        st.info(f"📋 {len(uploaded_files)} Datei(en) ausgewählt:")
        
        with st.expander("📁 Ausgewählte Dateien anzeigen", expanded=True):
            for uploaded_file in uploaded_files:
                
                ## Dokumnt Upload Maske 
                if "new_docs" not in st.session_state:
                    st.session_state.new_docs = {}
                st.session_state.new_docs[uploaded_file.name] = st.session_state.new_docs.get(uploaded_file.name, {})
                
                file_size = uploaded_file.size
                st.markdown(f"📄 **{uploaded_file.name}** ({file_size:,} Bytes)")
                #Title
                doc_title = st.text_input("Geben Sie der Datei einen Namen", key=f"doc_name_{uploaded_file.name}")
                if doc_title:
                    st.session_state.new_docs[uploaded_file.name]["title"] = doc_title if doc_title else uploaded_file.name
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    cls =st.segmented_control(
                        "Dateiklasse", 
                        DOCUMENT_CLASSES.keys(), 
                        key=f"doc_class_{uploaded_file.name}", 
                        help=f"Wählen Sie die passende Kategorie für die Datei aus. \n{get_doc_class_info_text()}")
                    st.session_state.new_docs[uploaded_file.name]["doc_class"] = DOCUMENT_CLASSES.get(cls)
                with c2:
                    #Downloadable
                    downloadable = st.checkbox("Datei ist für den Download verfügbar", value=False, key=f"downloadable_{uploaded_file.name}")
                    st.session_state.new_docs[uploaded_file.name]["downloadable"] = downloadable
                col1, col2, = st.columns([1, 1])
                with col1:
                    #valid From
                    valid_from = st.date_input(
                        f"Gültig von (optional)",
                        value=None,
                        min_value=datetime.date(1980, 1, 1),
                        key=f"valid_from_{uploaded_file.name}",
                        format="DD.MM.YYYY")
                    valid_from_int = int(valid_from.strftime("%Y%m%d")) if valid_from else 15000101
                    st.session_state.new_docs[uploaded_file.name]["valid_from"] = valid_from_int
                with col2:
                    #Valid To
                    valid_to = st.date_input(
                        f"Gültig bis (optional)",
                        value=None,
                        min_value=datetime.date(1980, 1, 1),
                        key=f"valid_to_{uploaded_file.name}",
                        format="DD.MM.YYYY")
                    valid_to_int = int(valid_to.strftime("%Y%m%d")) if valid_to else 99991231
                    st.session_state.new_docs[uploaded_file.name]["valid_to"] = valid_to_int
                    
                # Validation
                if valid_from and valid_to and valid_from > valid_to:
                    st.error("'Gültig ab' darf nicht nach 'Gültig bis' liegen.")
                # Replace existing document option
                if replace_box := st.checkbox("Das Dokument ersetzt ein existierendes Dokument", value=False, key=f"replace_document_{uploaded_file.name}"):
                    existing_docs = {os.path.basename(doc.get("title", source)): source for source, doc in st.session_state.docs_info.items()}
                    replace_select = st.selectbox("Bitte wählen Sie das zu ersetzende Dokument aus:",
                        options=list(existing_docs.keys()),
                        key=f"replace_select_{uploaded_file.name}",
                        disabled=not replace_box
                    )
                    st.session_state.new_docs[uploaded_file.name]["replace_document"] = replace_select
        
        st.divider()
        
        # Single confirmation button
        st.markdown("**Die Dateien werden gespeichert und automatisch zum Chatbot-Vektorstore hinzugefügt.**")
        
        if st.button("✅ Dateien hochladen und zum Vektorstore hinzufügen", key="confirm_upload", type="primary", use_container_width=True):
            #titlecheck
            for uploaded_file in uploaded_files:
                if not st.session_state.new_docs[uploaded_file.name].get("title"):
                    st.error(f"❌ Bitte geben Sie einen Titel für die Datei '{uploaded_file.name}' ein.")
                    return
                if not st.session_state.new_docs[uploaded_file.name].get("doc_class"):
                    st.error(f"❌ Bitte wählen Sie eine Dokumentklasse für die Datei '{uploaded_file.name}' aus.")
                    return
            
            
            uploads_dir = "uploads"
            os.makedirs(uploads_dir, exist_ok=True)
            
            if os.path.exists(persist_directory):
                st.info(f"ℹ️ Der Vektorstore-Ordner '{persist_directory}' existiert bereits. Neue Dateien werden hinzugefügt.")
            # load Doc_info file
            if "docs_info" not in st.session_state:
                if os.path.exists(f"{persist_directory}/doc_list.json"):
                    docs_info = load_doc_list(f"{persist_directory}/doc_list.json")
                else:
                    docs_info = {}
                st.session_state.docs_info = docs_info                
            
            with st.status("🔄 Verarbeite Dateien...", expanded=True) as status:
                try:
                    # Step 1: Save files to disk
                    status.update(label="💾 Speichere Dateien...", state="running")
                    saved_files = []
                    saved_count = 0
                    doc_list_updates = {}  # source -> fields added/changed by this upload
                    
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
                        
                        # Add new file to doc_list.json with validated metadata
                        entry = DocListEntry.from_doc_list_item(
                            file_path,
                            {
                                "title": st.session_state.new_docs[uploaded_file.name].get("title", uploaded_file.name),
                                "valid_from": st.session_state.new_docs[uploaded_file.name].get("valid_from"),
                                "valid_to": st.session_state.new_docs[uploaded_file.name].get("valid_to"),
                                "doc_class": st.session_state.new_docs[uploaded_file.name].get("doc_class"),
                                "downloadable": st.session_state.new_docs[uploaded_file.name].get("downloadable"),
                                "date_edited": int(datetime.date.today().strftime("%Y%m%d")),
                                "edited_by": st.session_state.doc_list_editor,
                                "virtual_path": upload_target_folder,
                            },
                        )
                        entry_dump = entry.model_dump() if hasattr(entry, "model_dump") else entry.dict()
                        st.session_state.docs_info[file_path] = {
                            key: value for key, value in entry_dump.items() if key != "source"
                        }
                        doc_list_updates[file_path] = dict(st.session_state.docs_info[file_path])
                        
                        # handle replace document option
                        if old_file := st.session_state.new_docs[uploaded_file.name].get("replace_document"):
                            existing_docs = {os.path.basename(doc.get("title", source)): source for source, doc in st.session_state.docs_info.items()}
                            st.session_state.docs_info[existing_docs[old_file]]["valid_to"] = st.session_state.new_docs[uploaded_file.name].get("valid_from") - 1
                            st.session_state.docs_info[existing_docs[old_file]]["date_edited"] = int(datetime.date.today().strftime("%Y%m%d"))
                            st.session_state.docs_info[existing_docs[old_file]]["edited_by"] = st.session_state.doc_list_editor
                            doc_list_updates.setdefault(existing_docs[old_file], {}).update({
                                field: st.session_state.docs_info[existing_docs[old_file]][field]
                                for field in ("valid_to", "date_edited", "edited_by")
                            })

                    # Merge only this upload's entries into the current file (under the lock), so
                    # folder changes made meanwhile in the explorer or other sessions are kept.
                    doc_list_path = os.path.join(persist_directory, DOC_LIST_FILENAME)
                    with doc_list_lock(persist_directory):
                        current_docs_info = load_doc_list(doc_list_path) if os.path.exists(doc_list_path) else {}
                        for source, fields in doc_list_updates.items():
                            current_docs_info[source] = {**current_docs_info.get(source, {}), **fields}
                        save_doc_list(doc_list_path, current_docs_info)
                    st.session_state.docs_info = current_docs_info
                    st.success(f"✅ {saved_count} Datei(en) erfolgreich gespeichert.")
                    
                    # Step 2: Add to vector store
                    status.update(label="🤖 Füge Dateien zum Vektorstore hinzu...", state="running")
                    
                    vector_store = extend_existing_vector_store(
                        data_folder=uploads_dir,
                        persist_directory=persist_directory,
                        model=embedding_model
                    )
                    
                    progress_bar.progress(95)  # 95%
                    
                    # sync metadata
                    status.update(label="🔄 Synchronisiere Metadaten...", state="running")
                    
                    sync_metadata(persist_directory=persist_directory, client=client, model=embedding_model)
                    
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
                del st.session_state['new_docs']
            
            elif 'upload_partial_success' in st.session_state:
                st.error("❌ Dateien wurden gespeichert, aber es gab einen Fehler beim Hinzufügen zum Vektorstore.")
                
                st.markdown("**📁 Gespeicherte Dateien:**")
                for file_path in st.session_state['upload_partial_success']:
                    st.markdown(f"  💾 {os.path.basename(file_path)}")
                
                st.warning("⚠️ Diese Dateien sind auf der Festplatte gespeichert, aber nicht im Chatbot verfügbar. Bitte kontaktieren Sie den Administrator.")
                
                # Clear the partial success files after displaying
                del st.session_state['upload_partial_success']
