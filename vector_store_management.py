import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from openai import OpenAI
import streamlit as st
from tqdm import tqdm
import re

def clean_text(text):
    """
    Cleans up text by:
    - Removing hyphenated line breaks (e.g., "verglei-\nchenden" -> "vergleichenden")
    - Collapsing multiple spaces and newlines into single spaces
    """
    text = re.sub(r"-\s*\n\s*", "", text)  # Fix hyphenated words
    text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces/newlines
    return text.strip()

# OpenAIEmbeddingsWrapper is a wrapper around the OpenAI embeddings API.
# It is used to embed documents and queries through the OpenAI API.
# As of now, the existing prototype uses OpenAI's "text-embedding-3-large" model.
# We'll need to update this to a local model to avoid GDPR related issues.
class OpenAIEmbeddingsWrapper:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def embed_documents(self, texts):
        """
        Accepts a list of texts and returns a list of embedding vectors.
        """
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text):
        """
        Accepts a single text string and returns its embedding vector.
        """
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding

# A helper function that loads a document from file_path based on its extension.
def load_document(file_path):
    try:
        # Skip Microsoft Office temporary files
        filename = os.path.basename(file_path)
        if filename.startswith('~$') or filename.startswith('.~'):
            print(f"Skipping temporary file: {file_path}")
            return []
        
        ext = os.path.splitext(file_path)[1].lower()
        docs = []
        
        try:
            if ext == ".pdf":
                # Use PyPDFLoader with error handling
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()
            elif ext == ".xlsx":
                loader = UnstructuredExcelLoader(file_path)
                docs = loader.load()
            else:
                docs = []  # Unsupported file type
        except UnicodeDecodeError as e:
            error_msg = f"UTF-8 decode error reading file content: {str(e)}"
            print(f"❌ {error_msg} for file: {file_path}")
            raise  # Re-raise to be caught by outer handler
        except UnicodeEncodeError as e:
            error_msg = f"UTF-8 encode error processing file: {str(e)}"
            print(f"❌ {error_msg} for file: {file_path}")
            raise  # Re-raise to be caught by outer handler
            
        # Add metadata to each Document: store its full path, its parent folder name, and an ID.
        folder_name = os.path.basename(os.path.dirname(file_path))
        for i, doc in enumerate(docs):
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            
            # Sanitize metadata to avoid encoding issues
            try:
                safe_file_path = file_path.encode('utf-8', errors='replace').decode('utf-8')
                safe_folder_name = folder_name.encode('utf-8', errors='replace').decode('utf-8')
                safe_basename = os.path.basename(file_path).encode('utf-8', errors='replace').decode('utf-8')
                
                doc.metadata.update({
                    "source": safe_file_path,
                    "folder": safe_folder_name,
                    "doc_id": f"{safe_folder_name}_{safe_basename}_{i}",
                    "page_number": doc.metadata.get("page", i + 1)  # Store PDF page number
                })
            except Exception as e:
                # Fall back to simple metadata if encoding fails
                doc.metadata.update({
                    "source": "unknown_source",
                    "folder": "unknown_folder",
                    "doc_id": f"doc_{i}",
                    "page_number": doc.metadata.get("page", i + 1)
                })
                print(f"Warning: Could not properly set metadata for document: {str(e)}")
            
            # Ensure page_content is a string and handle encoding issues
            if hasattr(doc, 'page_content'):
                try:
                    if isinstance(doc.page_content, bytes):
                        doc.page_content = doc.page_content.decode('utf-8', errors='replace')
                    elif not isinstance(doc.page_content, str):
                        doc.page_content = str(doc.page_content)
                except (UnicodeDecodeError, UnicodeEncodeError) as e:
                    print(f"⚠️ Encoding issue with page_content in {file_path}, attempting to fix: {str(e)}")
                    try:
                        if isinstance(doc.page_content, bytes):
                            doc.page_content = doc.page_content.decode('latin-1', errors='replace').encode('utf-8', errors='replace').decode('utf-8')
                        else:
                            doc.page_content = str(doc.page_content).encode('utf-8', errors='replace').decode('utf-8')
                    except Exception as e2:
                        print(f"⚠️ Could not fix encoding for page_content, using empty string: {str(e2)}")
                        doc.page_content = ""
        
        return docs
    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        error_msg = f"UTF-8 encoding error: {str(e)}"
        print(f"❌ {error_msg} for file: {file_path}")
        return []  # Return empty list instead of propagating the exception
    except Exception as e:
        error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
        print(f"❌ Error loading document {file_path}: {error_msg}")
        return []  # Return empty list instead of propagating the exception


# Function 1: setupVectorStore
# This function initializes (or loads) a persistent vector store based on the documents in data_folder.
# It uses the OpenAI embeddings API to embed the documents and queries.
# The vector store is persisted in the directory specified by persist_directory.
def setupVectorStore(data_folder: str, persist_directory: str, client: OpenAI, model: str):
    """
    Initialize (or load) a persistent vector store based on the documents in data_folder.
    """
    from langchain_community.vectorstores import Chroma

    # Set up the embedding function
    embeddings = OpenAIEmbeddingsWrapper(client, model)
    
    # If a persistent store exists, load it; otherwise, create a new one.
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        vector_store = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

    # Walk the data folder recursively and load all supported documents.
    all_docs = []
    for root, dirs, files in tqdm(os.walk(data_folder), desc="Loading documents"):
        for file in files:
            if file.lower().endswith((".pdf", ".docx", ".xlsx")):
                try:
                    file_path = os.path.join(root, file)
                    # Load and process documents
                    docs = load_document(file_path)
                    if docs:  # Check if documents were loaded successfully
                        # Clean the text content
                        for doc in docs:
                            if hasattr(doc, 'page_content'):
                                doc.page_content = clean_text(doc.page_content)
                        all_docs.extend(docs)
                except Exception as e:
                    # Safely print error message with encoding handling
                    safe_error = str(e).encode('utf-8', errors='replace').decode('utf-8')
                    try:
                        safe_path = file_path.encode('utf-8', errors='replace').decode('utf-8')
                        print(f"Error processing {safe_path}: {safe_error}. Skipping file.")
                    except:
                        print(f"Error processing a file: {safe_error}. Skipping file.")
                    continue
    
    if not all_docs:
        print("No documents were loaded. Please check the data folder and file formats.")
        return vector_store

    # Use a text splitter to break large documents into chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    
    # Process documents in smaller batches to stay within token limits
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Split documents into {len(split_docs)} chunks")
    
    if split_docs:
        batch_size = 100  # Adjust batch size based on your token limit
        successful_batches = 0
        
        for i in tqdm(range(0, len(split_docs), batch_size), desc="Processing batches"):
            batch = split_docs[i:i + batch_size]
            try:
                vector_store.add_documents(batch)
                successful_batches += 1
                print(f"Successfully added batch {i // batch_size + 1} with {len(batch)} documents")
            except Exception as e:
                print(f"Error adding batch {i // batch_size + 1}: {str(e)}")
                return None
        
        print(f"Successfully processed and stored {len(split_docs)} document chunks in {successful_batches} batches.")
        
        # Note: In newer versions of Chroma, documents are automatically persisted
        # No need to call vector_store.persist() manually
    else:
        print("No documents were split. Please check the text splitter configuration.")
    
    return vector_store


# Function 2: updateVectorStore
# This function checks the data_folder for new documents (i.e. files whose full path is not yet in the vector store).
# For any new files, it loads, splits, and adds their content (with metadata) to the vector store,
# then persists the updated store.
def test_embeddings_search(persist_directory: str, embeddings, query: str, k: int = 5):
    """
    Test the embeddings search functionality to see what documents are retrieved for a given query.
    
    Args:
        persist_directory: Path to the vector store directory
        embeddings: Embedding function to use
        query: Search query string
        k: Number of documents to retrieve (default: 5)
    
    Returns:
        List of retrieved documents with metadata
    """
    try:
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        print(f"\n=== Testing Embeddings Search ===")
        print(f"Query: '{query}'")
        print(f"Retrieving top {k} documents...")
        
        # Perform similarity search
        retrieved_docs = vector_store.similarity_search(query, k=k)
        
        print(f"\nFound {len(retrieved_docs)} documents:")
        print("=" * 80)
        
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.metadata
            content = doc.page_content
            
            print(f"\n📄 Document {i}:")
            print(f"   📂 Folder: {metadata.get('folder', 'Unknown')}")
            print(f"   📜 Source: {metadata.get('source', 'Unknown')}")
            print(f"   🆔 Doc ID: {metadata.get('doc_id', 'Unknown')}")
            print(f"   📑 Page: {metadata.get('page_number', 'Unknown')}")
            print(f"   📝 Content Preview ({len(content)} chars):")
            print(f"   {content[:200]}{'...' if len(content) > 200 else ''}")
            print("-" * 80)
        
        return retrieved_docs
        
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return []

def test_embeddings_search_with_scores(persist_directory: str, embeddings, query: str, k: int = 5):
    """
    Test the embeddings search functionality with similarity scores.
    
    Args:
        persist_directory: Path to the vector store directory
        embeddings: Embedding function to use
        query: Search query string
        k: Number of documents to retrieve (default: 5)
    
    Returns:
        List of tuples (document, score)
    """
    try:
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        print(f"\n=== Testing Embeddings Search with Scores ===")
        print(f"Query: '{query}'")
        print(f"Retrieving top {k} documents...")
        
        # Perform similarity search with scores
        retrieved_docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
        
        print(f"\nFound {len(retrieved_docs_with_scores)} documents:")
        print("=" * 80)
        
        for i, (doc, score) in enumerate(retrieved_docs_with_scores, 1):
            metadata = doc.metadata
            content = doc.page_content
            
            print(f"\n📄 Document {i} (Score: {score:.4f}):")
            print(f"   📂 Folder: {metadata.get('folder', 'Unknown')}")
            print(f"   📜 Source: {metadata.get('source', 'Unknown')}")
            print(f"   🆔 Doc ID: {metadata.get('doc_id', 'Unknown')}")
            print(f"   📑 Page: {metadata.get('page_number', 'Unknown')}")
            print(f"   📝 Content Preview ({len(content)} chars):")
            print(f"   {content[:200]}{'...' if len(content) > 200 else ''}")
            print("-" * 80)
        
        return retrieved_docs_with_scores
        
    except Exception as e:
        print(f"Error during search with scores: {str(e)}")
        return []

def interactive_search_test(persist_directory: str, embeddings):
    """
    Interactive function to test embeddings search with user input.
    """
    print("\n=== Interactive Embeddings Search Test ===")
    print("Enter queries to test the vector store search functionality.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Check if we have subfolders or a single directory
    if os.path.exists(persist_directory):
        subfolders = [f for f in os.listdir(persist_directory) if os.path.isdir(os.path.join(persist_directory, f))]
        
        if subfolders:
            print(f"Found subfolders: {subfolders}")
            print("You can search in:")
            print("1. All subfolders combined")
            print("2. Individual subfolders")
            search_mode = input("Choose search mode (1/2, default 1): ").strip() or "1"
        else:
            search_mode = "1"  # Single directory
            subfolders = []
    else:
        print(f"Directory {persist_directory} does not exist!")
        return
    
    while True:
        try:
            query = input("Enter your search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Exiting search test...")
                break
            
            if not query:
                print("Please enter a non-empty query.")
                continue
            
            # Ask for number of results
            try:
                k = int(input("Number of results to retrieve (default 5): ") or "5")
            except ValueError:
                k = 5
            
            # Ask if user wants scores
            show_scores = input("Show similarity scores? (y/n, default n): ").lower().startswith('y')
            
            if search_mode == "1" and subfolders:
                # Search all subfolders
                print(f"\n=== Searching all subfolders for: '{query}' ===")
                all_results = []
                
                for subfolder in subfolders:
                    subfolder_path = os.path.join(persist_directory, subfolder)
                    print(f"\n--- Results from '{subfolder}' ---")
                    
                    if show_scores:
                        results = test_embeddings_search_with_scores(subfolder_path, embeddings, query, k)
                        all_results.extend(results)
                    else:
                        results = test_embeddings_search(subfolder_path, embeddings, query, k)
                        all_results.extend(results)
                
                print(f"\n=== Total results across all subfolders: {len(all_results)} ===")
                
            elif search_mode == "2" and subfolders:
                # Search individual subfolders
                print(f"Available subfolders: {subfolders}")
                selected_subfolder = input(f"Enter subfolder name (or press Enter for '{subfolders[0]}'): ").strip()
                
                if not selected_subfolder:
                    selected_subfolder = subfolders[0]
                
                if selected_subfolder in subfolders:
                    subfolder_path = os.path.join(persist_directory, selected_subfolder)
                    print(f"\n=== Searching '{selected_subfolder}' for: '{query}' ===")
                    
                    if show_scores:
                        test_embeddings_search_with_scores(subfolder_path, embeddings, query, k)
                    else:
                        test_embeddings_search(subfolder_path, embeddings, query, k)
                else:
                    print(f"Subfolder '{selected_subfolder}' not found!")
                    continue
            else:
                # Single directory search
                if show_scores:
                    test_embeddings_search_with_scores(persist_directory, embeddings, query, k)
                else:
                    test_embeddings_search(persist_directory, embeddings, query, k)
            
            print("\n" + "="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting search test...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def check_vector_store_status(persist_directory: str, embeddings):
    """
    Check the current status of the vector store and return information about existing sources.
    """
    existing_sources = set()
    
    # Check if vector store directory exists and has content
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        print(f"Vector store directory '{persist_directory}' is empty or doesn't exist")
        return existing_sources, 0
    
    try:
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        stored_data = vector_store._collection.get()
        
        metadata_count = len(stored_data.get('metadatas', []))
        print(f"Vector store contains {metadata_count} metadata entries")
        
        if stored_data and "metadatas" in stored_data:
            for meta in stored_data["metadatas"]:
                if isinstance(meta, dict) and "source" in meta:
                    existing_sources.add(meta["source"])
                elif isinstance(meta, list):
                    for m in meta:
                        if isinstance(m, dict) and "source" in m:
                            existing_sources.add(m["source"])
        
        print(f"Found {len(existing_sources)} unique source files")
        return existing_sources, metadata_count
        
    except Exception as e:
        print(f"Error accessing vector store: {str(e)}")
        return existing_sources, 0


def delete_document_from_vector_store(persist_directory: str, embeddings, file_path: str):
    """
    Delete all chunks associated with a specific file from the vector store.
    
    Args:
        persist_directory: Path to the vector store directory
        embeddings: Embedding function to use
        file_path: Full path of the file to delete from vector store
    
    Returns:
        Tuple of (success: bool, deleted_count: int, error_message: str)
    """
    try:
        # Check if vector store exists
        if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            return False, 0, "Vector store directory doesn't exist or is empty"
        
        # Load vector store
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        # Get all documents
        stored_data = vector_store._collection.get()
        
        if not stored_data or "metadatas" not in stored_data or "ids" not in stored_data:
            return False, 0, "No documents found in vector store"
        
        # Find IDs of chunks that match this file path
        ids_to_delete = []
        for i, meta in enumerate(stored_data["metadatas"]):
            if isinstance(meta, dict) and meta.get("source") == file_path:
                ids_to_delete.append(stored_data["ids"][i])
        
        if not ids_to_delete:
            return False, 0, f"No chunks found for file: {file_path}"
        
        # Delete the chunks
        vector_store._collection.delete(ids=ids_to_delete)
        
        print(f"Successfully deleted {len(ids_to_delete)} chunks for file: {file_path}")
        return True, len(ids_to_delete), ""
        
    except Exception as e:
        error_msg = f"Error deleting from vector store: {str(e)}"
        print(error_msg)
        return False, 0, error_msg

def updateVectorStore(data_folder: str, persist_directory: str, client: OpenAI, model: str):
    """
    Check the data_folder for new documents (i.e. files whose full path is not yet in the vector store).
    For any new files, load, split, and add their content (with metadata) to the vector store,
    then persist the updated store.
    """
    # Set up the embedding function
    embeddings = OpenAIEmbeddingsWrapper(client, model)
    
    # Get the list of already-indexed file paths from the vector store metadata.
    existing_sources, metadata_count = check_vector_store_status(persist_directory, embeddings)
    print("existing_sources: ", existing_sources)
    
    # Create vector store instance for adding new documents
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    new_docs = []
    for root, dirs, files in tqdm(os.walk(data_folder), desc="Checking for new documents"):
        for file in files:
            if file.lower().endswith((".pdf", ".docx", ".xlsx")):
                file_path = os.path.join(root, file)
                if file_path not in existing_sources:
                    print("file_path: ", file_path)
                    docs = load_document(file_path)
                    new_docs.extend(docs)
    
    if new_docs:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        split_docs = text_splitter.split_documents(new_docs)
        print(len(split_docs))
        try:
            vector_store.add_documents(split_docs)
            print(f"Successfully added {len(split_docs)} document chunks to vector store")
        except Exception as e:
            print(f"Error during document splitting or storage: {str(e)}")
            print("Trying again but with multiple subsequent API calls")
            batch_size = 100  # Adjust batch size based on your token limit
            for i in tqdm(range(0, len(split_docs), batch_size), desc="Processing batches"):
                batch = split_docs[i:i + batch_size]
                try:
                    vector_store.add_documents(batch)
                except Exception as e:
                    print(f"Error adding batch {i // batch_size + 1}: {str(e)}")
                    return None
            print(f"Successfully added {len(split_docs)} document chunks in batches")
            # Note: In newer versions of Chroma, documents are automatically persisted
    return vector_store

def create_unified_vector_store(original_data_folder: str, original_persist_directory: str, client: OpenAI, model: str):
    """
    Create a unified vector store that combines all documents into one searchable index.
    Handles documents both directly in the folder and in subfolders recursively.
    This is useful for the chatbot which expects a single vector store.
    """
    print("\n=== Creating Unified Vector Store ===")
    
    # Set up the embedding function
    embeddings = OpenAIEmbeddingsWrapper(client, model)
    
    # Create unified persist directory
    unified_persist_directory = os.path.join(original_persist_directory, "unified")
    if not os.path.exists(unified_persist_directory):
        os.makedirs(unified_persist_directory)
    
    # Collect all documents from the data folder (handles both files directly in folder and in subfolders)
    all_docs = []
    for root, dirs, files in os.walk(original_data_folder):
        for file in files:
            if file.lower().endswith((".pdf", ".docx", ".xlsx")):
                try:
                    file_path = os.path.join(root, file)
                    print(f"Loading document: {file_path}")
                    docs = load_document(file_path)
                    if docs:
                        # Clean the text content
                        for doc in docs:
                            if hasattr(doc, 'page_content'):
                                doc.page_content = clean_text(doc.page_content)
                        all_docs.extend(docs)
                except UnicodeDecodeError as e:
                    print(f"🔒 Skipping encrypted/corrupted file: {file_path}")
                    print(f"   Reason: Unicode decode error - file may be encrypted or corrupted")
                    continue
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(keyword in error_msg for keyword in ['encrypted', 'password', 'protected', 'corrupted', 'decode', 'utf-8']):
                        print(f"🔒 Skipping encrypted/corrupted file: {file_path}")
                        print(f"   Reason: {str(e)}")
                    else:
                        print(f"❌ Error loading {file_path}: {str(e)}")
                    continue
    
    if not all_docs:
        print("No documents were loaded for unified store.")
        return None
    
    print(f"Loaded {len(all_docs)} documents from all subfolders")
    
    # Create unified vector store
    vector_store = Chroma(embedding_function=embeddings, persist_directory=unified_persist_directory)
    
    # Use a text splitter to break large documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Split documents into {len(split_docs)} chunks")
    
    if split_docs:
        batch_size = 100
        successful_batches = 0
        
        for i in tqdm(range(0, len(split_docs), batch_size), desc="Processing unified batches"):
            batch = split_docs[i:i + batch_size]
            try:
                vector_store.add_documents(batch)
                successful_batches += 1
            except Exception as e:
                print(f"Error adding batch {i // batch_size + 1}: {str(e)}")
                return None
        
        print(f"Successfully created unified vector store with {len(split_docs)} document chunks")
        return vector_store
    else:
        print("No documents were split for unified store.")
        return None

def create_fresh_unified_vector_store(data_folder: str, persist_directory: str = "kisski_db_v2", model: str = "qwen3-embedding-4b"):
    """
    Create a fresh unified vector store from scratch. This is the main function for new users.
    
    Args:
        data_folder: Path to the folder containing documents (can have subfolders)
        persist_directory: Where to store the vector database (default: "kisski_db_v2")
        model: Embedding model to use (default: "qwen3-embedding-4b")
    
    Returns:
        Chroma vector store instance or None if failed
    """
    print(f"\n=== Creating Fresh Unified Vector Store ===")
    print(f"Data folder: {data_folder}")
    print(f"Output directory: {persist_directory}")
    print(f"Embedding model: {model}")
    
    try:
        # Initialize OpenAI client
        client = OpenAI(
            base_url="https://chat-ai.academiccloud.de/v1",
            api_key=st.secrets["KISSKI_API_KEY"]
        )
        
        # Check if data folder exists
        if not os.path.exists(data_folder):
            print(f"❌ Data folder does not exist: {data_folder}")
            return None
        
        # Create the unified vector store
        vector_store = create_unified_vector_store(data_folder, persist_directory, client, model)
        
        if vector_store:
            # Copy to main directory for easy access
            unified_persist_directory = os.path.join(persist_directory, "unified")
            main_vector_store_path = persist_directory
            
            if os.path.exists(main_vector_store_path):
                # Clean main directory
                import shutil
                for item in os.listdir(main_vector_store_path):
                    item_path = os.path.join(main_vector_store_path, item)
                    if os.path.isdir(item_path) and item != "unified":
                        shutil.rmtree(item_path)
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                
                # Copy unified store to main directory
                print(f"Copying unified vector store to main directory...")
                shutil.copytree(unified_persist_directory, main_vector_store_path, dirs_exist_ok=True)
                print("✅ Fresh unified vector store created and ready for use!")
                
                # Test the vector store
                embeddings = OpenAIEmbeddingsWrapper(client, model)
                print("\n=== Testing Fresh Vector Store ===")
                test_embeddings_search(main_vector_store_path, embeddings, "test query", k=2)
                
                return vector_store
        else:
            print("❌ Failed to create unified vector store")
            return None
            
    except Exception as e:
        print(f"❌ Error creating fresh unified vector store: {str(e)}")
        return None

def extend_existing_vector_store(data_folder: str, persist_directory: str = "kisski_db_v2", model: str = "qwen3-embedding-4b"):
    """
    Extend an existing unified vector store with new documents.
    
    Args:
        data_folder: Path to the folder containing NEW documents to add
        persist_directory: Path to existing vector store directory
        model: Embedding model to use (default: "qwen3-embedding-4b")
    
    Returns:
        Updated Chroma vector store instance or None if failed
    """
    print(f"\n=== Extending Existing Vector Store ===")
    print(f"New data folder: {data_folder}")
    print(f"Existing vector store: {persist_directory}")
    print(f"Embedding model: {model}")
    
    try:
        # Initialize OpenAI client
        client = OpenAI(
            base_url="https://chat-ai.academiccloud.de/v1",
            api_key=st.secrets["KISSKI_API_KEY"]
        )
        embeddings = OpenAIEmbeddingsWrapper(client, model)
        
        # Check if existing vector store exists
        if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            print(f"❌ No existing vector store found at {persist_directory}")
            print("💡 Use create_fresh_unified_vector_store() to create a new one")
            return None
        
        # Load existing vector store
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        # Get existing sources to avoid duplicates
        existing_sources, metadata_count = check_vector_store_status(persist_directory, embeddings)
        print(f"Found existing vector store with {metadata_count} documents from {len(existing_sources)} sources")
        
        # Collect new documents
        new_docs = []
        failed_files = []
        for root, dirs, files in tqdm(os.walk(data_folder), desc="Scanning for new documents"):
            for file in files:
                if file.lower().endswith((".pdf", ".docx", ".xlsx")):
                    file_path = os.path.join(root, file)
                    if file_path not in existing_sources:
                        # Safely log file path even if it contains non-UTF-8 bytes
                        try:
                            print(f"Found new document: {file_path}")
                        except UnicodeEncodeError:
                            safe_path = file_path.encode("utf-8", errors="replace").decode("utf-8")
                            print(f"Found new document: {safe_path}")
                        try:
                            docs = load_document(file_path)
                            if docs:
                                # Clean the text content
                                for doc in docs:
                                    if hasattr(doc, 'page_content'):
                                        try:
                                            doc.page_content = clean_text(doc.page_content)
                                        except (UnicodeDecodeError, UnicodeEncodeError) as e:
                                            print(f"⚠️ Encoding error cleaning content from {file_path}: {str(e)}")
                                            # Try to fix encoding issues
                                            try:
                                                if isinstance(doc.page_content, bytes):
                                                    doc.page_content = doc.page_content.decode('utf-8', errors='replace')
                                                else:
                                                    doc.page_content = doc.page_content.encode('utf-8', errors='replace').decode('utf-8')
                                                doc.page_content = clean_text(doc.page_content)
                                            except Exception as e2:
                                                print(f"⚠️ Could not fix encoding for {file_path}, skipping document: {str(e2)}")
                                                continue
                                new_docs.extend(docs)
                            else:
                                print(f"⚠️ No documents loaded from {file_path} (file may be empty or corrupted)")
                        except UnicodeDecodeError as e:
                            error_msg = f"UTF-8 decode error reading {file_path}: {str(e)}"
                            print(f"❌ {error_msg}")
                            failed_files.append((file_path, error_msg))
                        except UnicodeEncodeError as e:
                            error_msg = f"UTF-8 encode error processing {file_path}: {str(e)}"
                            print(f"❌ {error_msg}")
                            failed_files.append((file_path, error_msg))
                        except Exception as e:
                            error_msg = f"Error processing {file_path}: {str(e)}"
                            print(f"❌ {error_msg}")
                            failed_files.append((file_path, error_msg))
                    else:
                        print(f"Document already exists: {file_path}")
        
        if failed_files:
            print(f"\n⚠️ Failed to process {len(failed_files)} file(s):")
            for file_path, error in failed_files:
                print(f"  - {file_path}: {error}")
        
        if not new_docs:
            print("✅ No new documents found to add")
            return vector_store
        
        print(f"Found {len(new_docs)} new documents to add")
        
        # Split and add new documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        
        split_docs = text_splitter.split_documents(new_docs)
        print(f"Split new documents into {len(split_docs)} chunks")
        
        if split_docs:
            batch_size = 100
            successful_batches = 0
            
            for i in tqdm(range(0, len(split_docs), batch_size), desc="Adding new documents"):
                batch = split_docs[i:i + batch_size]
                try:
                    vector_store.add_documents(batch)
                    successful_batches += 1
                except Exception as e:
                    print(f"Error adding batch {i // batch_size + 1}: {str(e)}")
                    return None
            
            print(f"✅ Successfully added {len(split_docs)} new document chunks in {successful_batches} batches")
            
            # Test the updated vector store
            print("\n=== Testing Updated Vector Store ===")
            test_embeddings_search(persist_directory, embeddings, "test query", k=2)
            
            return vector_store
        else:
            print("❌ No documents were split for addition")
            return None
            
    except Exception as e:
        # Safely encode error message to avoid truncation
        try:
            error_msg = str(e)
            if isinstance(error_msg, bytes):
                error_msg = error_msg.decode('utf-8', errors='replace')
            else:
                error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
        except:
            error_msg = "Unknown error (could not decode error message)"
        
        print(f"❌ Error extending vector store: {error_msg}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return None

def load_unified_vector_store(persist_directory: str = "kisski_db_v2"):
    """
    Load the unified vector store for use in chatbot and other applications.
    
    Args:
        persist_directory: Path to the vector store directory (default: "kisski_db_v2")
    
    Returns:
        Chroma vector store instance or None if not found
    """
    try:
        client = OpenAI(
            base_url="https://chat-ai.academiccloud.de/v1",
            api_key=st.secrets["KISSKI_API_KEY"]
        )
        embeddings = OpenAIEmbeddingsWrapper(client, "qwen3-embedding-4b")
        
        # Try to load from main directory first
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            try:
                vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                # Test if the vector store actually has data
                test_results = vector_store.similarity_search("test", k=1)
                print(f"✅ Loaded unified vector store from {persist_directory} (contains {len(test_results)}+ documents)")
                return vector_store
            except Exception as e:
                print(f"⚠️ Main directory exists but vector store is corrupted: {str(e)}")
        
        # If not found in main directory, try unified subdirectory
        unified_path = os.path.join(persist_directory, "unified")
        if os.path.exists(unified_path) and os.listdir(unified_path):
            try:
                vector_store = Chroma(persist_directory=unified_path, embedding_function=embeddings)
                # Test if the vector store actually has data
                test_results = vector_store.similarity_search("test", k=1)
                print(f"✅ Loaded unified vector store from {unified_path} (contains {len(test_results)}+ documents)")
                return vector_store
            except Exception as e:
                print(f"⚠️ Unified directory exists but vector store is corrupted: {str(e)}")
        
        print(f"❌ No working unified vector store found in {persist_directory}")
        print(f"   Main directory exists: {os.path.exists(persist_directory)}")
        print(f"   Main directory has content: {os.path.exists(persist_directory) and os.listdir(persist_directory)}")
        print(f"   Unified directory exists: {os.path.exists(unified_path)}")
        print(f"   Unified directory has content: {os.path.exists(unified_path) and os.listdir(unified_path)}")
        return None
        
    except Exception as e:
        print(f"❌ Error loading unified vector store: {str(e)}")
        return None

def smart_update_vector_store(main_directory: str, persist_directory: str = "kisski_db_v2", model: str = "qwen3-embedding-4b", dry_run: bool = False):
    """
    Smart update function that scans main directory + subfolders and adds only new documents.
    This is the main function for incremental updates.
    
    Args:
        main_directory: Main directory containing subfolders with documents
        persist_directory: Path to existing vector store directory
        model: Embedding model to use
        dry_run: If True, only scan and report what would be added without actually doing it
    
    Returns:
        Updated Chroma vector store instance or None if failed (or dry run results if dry_run=True)
    """
    print(f"\n=== Smart Vector Store Update ===")
    print(f"Main directory: {main_directory}")
    print(f"Vector store: {persist_directory}")
    
    try:
        # Initialize OpenAI client
        client = OpenAI(
            base_url="https://chat-ai.academiccloud.de/v1",
            api_key=st.secrets["KISSKI_API_KEY"]
        )
        embeddings = OpenAIEmbeddingsWrapper(client, model)
        
        # Check if main directory exists
        if not os.path.exists(main_directory):
            print(f"❌ Main directory not found: {main_directory}")
            return None
        
        # Check if existing vector store exists
        if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            print(f"❌ No existing vector store found at {persist_directory}")
            print("💡 Use create_fresh_unified_vector_store() to create a new one")
            return None
        
        # Load existing vector store
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        # Get existing sources to avoid duplicates
        existing_sources, metadata_count = check_vector_store_status(persist_directory, embeddings)
        print(f"📊 Current vector store: {metadata_count} documents from {len(existing_sources)} sources")
        
        # Scan for new documents in main directory and all subfolders
        new_docs = []
        scanned_files = 0
        new_files = 0
        
        print(f"🔍 Scanning main directory and subfolders...")
        for root, dirs, files in tqdm(os.walk(main_directory), desc="Scanning"):
            for file in files:
                if file.lower().endswith((".pdf", ".docx", ".xlsx")):
                    scanned_files += 1
                    file_path = os.path.join(root, file)
                    
                    if file_path not in existing_sources:
                        new_files += 1
                        print(f"📄 New: {file_path}")
                        
                        docs = load_document(file_path)
                        if docs:
                            for doc in docs:
                                if hasattr(doc, 'page_content'):
                                    doc.page_content = clean_text(doc.page_content)
                            new_docs.extend(docs)
        
        print(f"📋 Scan results: {scanned_files} files scanned, {new_files} new files found")
        
        if not new_docs:
            print("✅ No new documents to add!")
            return vector_store
        
        # Process and add new documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        
        split_docs = text_splitter.split_documents(new_docs)
        print(f"📄 Processing {len(split_docs)} new document chunks...")
        
        batch_size = 100
        for i in tqdm(range(0, len(split_docs), batch_size), desc="Adding documents"):
            batch = split_docs[i:i + batch_size]
            try:
                vector_store.add_documents(batch)
            except Exception as e:
                print(f"❌ Error adding batch {i // batch_size + 1}: {str(e)}")
                return None
        
        print(f"✅ Successfully added {len(split_docs)} new document chunks!")
        return vector_store
        
    except Exception as e:
        print(f"❌ Error in smart update: {str(e)}")
        return None

def quick_search_test(persist_directory: str, query: str, k: int = 5, show_scores: bool = False):
    """
    Quick function to test embeddings search - can be imported and used from other scripts.
    
    Args:
        persist_directory: Path to the vector store directory
        query: Search query string
        k: Number of documents to retrieve (default: 5)
        show_scores: Whether to show similarity scores (default: False)
    
    Returns:
        List of retrieved documents (with or without scores)
    """
    client = OpenAI(
        base_url="https://chat-ai.academiccloud.de/v1",
        api_key=st.secrets["KISSKI_API_KEY"]
    )
    embeddings = OpenAIEmbeddingsWrapper(client, "qwen3-embedding-4b")
    
    if show_scores:
        return test_embeddings_search_with_scores(persist_directory, embeddings, query, k)
    else:
        return test_embeddings_search(persist_directory, embeddings, query, k)




if __name__ == "__main__":
    client = OpenAI(
        base_url="https://chat-ai.academiccloud.de/v1",
        api_key=st.secrets["KISSKI_API_KEY"]
    )
    #data_folder = "../../BBS/data/Projekt KI Demonstrator - Unterlagen/Unterrichtsmaterialien"
    persist_directory = "rsev_v2"
    embeddings = OpenAIEmbeddingsWrapper(client, "qwen3-embedding-4b")

    # Check vector store status
    existing_sources, metadata_count = check_vector_store_status(persist_directory, embeddings)
    print(f"Found {metadata_count} documents from {len(existing_sources)} sources")
    for source in existing_sources:
        print(source)
        print("-"*100)


    # if not os.path.exists(persist_directory):
    #     os.makedirs(persist_directory)
    

    # Store original paths
    # original_data_folder = data_folder
    # original_persist_directory = persist_directory
    
    # # Example usage for new users - create fresh unified vector store
    # print("\n=== EXAMPLE: Creating Fresh Unified Vector Store ===")
    # print("This is what new users should run to create their vector store from scratch:")
    # print(f"create_fresh_unified_vector_store('{original_data_folder}', '{original_persist_directory}')")
    
    # # Create the unified vector store using the new function
    # unified_vector_store = create_fresh_unified_vector_store(original_data_folder, original_persist_directory)
    
    # if unified_vector_store:
    #     print("\n=== EXAMPLE: Testing Search Functionality ===")
    #     test_embeddings_search(original_persist_directory, embeddings, "Kalthämmern Metalltechnik", k=3)
        
    #     print("\n=== EXAMPLE: Extending with New Documents ===")
    #     print("To add new documents later, users can run:")
    #     print(f"extend_existing_vector_store('/path/to/new/documents', '{original_persist_directory}')")
        
    #     # Example of extending (commented out since we don't have new documents)
    #     # extend_existing_vector_store("/path/to/new/documents", original_persist_directory)
    # else:
    #     print("❌ Failed to create unified vector store")
     
    # Uncomment the line below to start interactive search testing
    # interactive_search_test(original_persist_directory, embeddings)
    
    # Later, to update the vector store with new documents:
    #vector_store = updateVectorStore(data_folder, persist_directory, client, model="text-embedding-3-large")
