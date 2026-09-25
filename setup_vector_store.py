#!/usr/bin/env python3
"""
CLI tool for managing BBS vector stores.

This script provides functionality to:
1. Create a fresh unified vector store from documents
2. Extend an existing vector store with new documents
3. Test vector store functionality

Usage Examples:
    # Create a fresh vector store
    python setup_vector_store.py fresh --data-folder /path/to/documents --output-dir my_vector_store
    
    # Extend existing vector store with new documents
    python setup_vector_store.py extend --data-folder /path/to/new/documents --vector-store my_vector_store
    
    # Use default paths
    python setup_vector_store.py fresh
    python setup_vector_store.py extend

    # Sync vector store metadata from doc_list.json
    python setup_vector_store.py sync-metadata --vector-store rsev_v2
"""

import os
import sys
import argparse

def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="BBS Vector Store Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s fresh --data-folder /path/to/documents --output-dir my_vector_store
  %(prog)s extend --data-folder /path/to/new/documents --vector-store my_vector_store
  %(prog)s fresh  # Use default paths
  %(prog)s extend --dry-run  # Preview what would be added without actually doing it
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Fresh command
    fresh_parser = subparsers.add_parser(
        'fresh',
        help='Create a fresh unified vector store from documents'
    )
    fresh_parser.add_argument(
        '--data-folder',
        type=str,
        default="../../data/bbs3/drive_download_combined",
        help='Path to folder containing documents (default: ../../data/bbs3/drive_download_combined)'
    )
    fresh_parser.add_argument(
        '--output-dir',
        type=str,
        default="rsev_v2",
        help='Directory to save the vector store (default: rsev_v2)'
    )
    fresh_parser.add_argument(
        '--model',
        type=str,
        default="qwen3-embedding-4b",
        help='Embedding model to use (default: qwen3-embedding-4b)'
    )
    
    # Extend command
    extend_parser = subparsers.add_parser(
        'extend',
        help='Extend existing vector store with new documents'
    )
    extend_parser.add_argument(
        '--data-folder',
        type=str,
        required=True,
        help='Path to folder containing NEW documents to add'
    )
    extend_parser.add_argument(
        '--vector-store',
        type=str,
        default="rsev_v2",
        help='Path to existing vector store directory (default: rsev_v2)'
    )
    extend_parser.add_argument(
        '--model',
        type=str,
        default="qwen3-embedding-4b",
        help='Embedding model to use (default: qwen3-embedding-4b)'
    )
    extend_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be added without actually doing it'
    )
    
    # Update command (smart update)
    update_parser = subparsers.add_parser(
        'update',
        help='Smart update: scan main directory and add only new documents'
    )
    update_parser.add_argument(
        '--main-directory',
        type=str,
        required=True,
        help='Main directory containing subfolders with documents'
    )
    update_parser.add_argument(
        '--vector-store',
        type=str,
        default="rsev_v2",
        help='Path to existing vector store directory (default: rsev_v2)'
    )
    update_parser.add_argument(
        '--model',
        type=str,
        default="qwen3-embedding-4b",
        help='Embedding model to use (default: qwen3-embedding-4b)'
    )
    update_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be added without actually doing it'
    )

    # Metadata sync command
    metadata_parser = subparsers.add_parser(
        'sync-metadata',
        help='Sync title, downloadable and validity metadata from doc_list.json'
    )
    metadata_parser.add_argument(
        '--vector-store',
        type=str,
        default="rsev_v2",
        help='Path to existing vector store directory (default: rsev_v2)'
    )

    # Document index command
    doc_index_parser = subparsers.add_parser(
        'doc-index',
        help='Create or refresh only the document index in the vector store'
    )
    doc_index_parser.add_argument(
        '--vector-store',
        type=str,
        default="rsev_v2",
        help='Path to existing vector store directory (default: rsev_v2)'
    )
    
    return parser

def validate_paths(data_folder, vector_store_path=None):
    """Validate that required paths exist."""
    if not os.path.exists(data_folder):
        print(f"❌ Data folder not found: {data_folder}")
        print("Please check the path and try again.")
        return False
    
    if vector_store_path and not os.path.exists(vector_store_path):
        print(f"❌ Vector store directory not found: {vector_store_path}")
        print("Please check the path or create a fresh vector store first.")
        return False
    
    return True

def create_fresh_vector_store(data_folder, output_dir, model):
    """Create a fresh unified vector store."""
    from src.rag.vector_store_management import create_fresh_unified_vector_store, load_unified_vector_store

    print("🚀 BBS Vector Store Setup - Fresh Creation")
    print("=" * 50)
    
    if not validate_paths(data_folder):
        return False
    
    print(f"📁 Data folder: {data_folder}")
    print(f"💾 Vector store will be saved to: {output_dir}")
    print(f"🤖 Embedding model: {model}")
    
    # Create the unified vector store
    print("\n🔄 Creating unified vector store...")
    vector_store = create_fresh_unified_vector_store(data_folder, output_dir, model)
    
    if vector_store:
        print("\n✅ Vector store created successfully!")
        
        # Test loading the vector store
        print("\n🧪 Testing vector store loading...")
        loaded_store = load_unified_vector_store(output_dir)
        
        if loaded_store:
            print("✅ Vector store can be loaded successfully!")
            print("\n🎉 Setup complete! Your chatbot is ready to use.")
            print(f"\nTo use in your chatbot, the vector store is available at: {output_dir}")
            return True
        else:
            print("❌ Failed to load the created vector store")
            return False
    else:
        print("❌ Failed to create vector store")
        return False

def extend_vector_store(data_folder, vector_store_path, model, dry_run=False):
    """Extend existing vector store with new documents."""
    from src.rag.vector_store_management import extend_existing_vector_store, load_unified_vector_store

    print("🚀 BBS Vector Store Setup - Extension")
    print("=" * 50)
    
    if not validate_paths(data_folder, vector_store_path):
        return False
    
    print(f"📁 New documents folder: {data_folder}")
    print(f"💾 Existing vector store: {vector_store_path}")
    print(f"🤖 Embedding model: {model}")
    if dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
    
    # Extend the vector store
    print("\n🔄 Extending vector store...")
    vector_store = extend_existing_vector_store(data_folder, vector_store_path, model)
    
    if vector_store:
        print("\n✅ Vector store extended successfully!")
        
        # Test the updated vector store
        print("\n🧪 Testing updated vector store...")
        loaded_store = load_unified_vector_store(vector_store_path)
        
        if loaded_store:
            print("✅ Extended vector store can be loaded successfully!")
            print("\n🎉 Extension complete! Your chatbot is ready to use.")
            return True
        else:
            print("❌ Failed to load the extended vector store")
            return False
    else:
        print("❌ Failed to extend vector store")
        return False

def smart_update_vector_store_cmd(main_directory, vector_store_path, model, dry_run=False):
    """Smart update: scan main directory and add only new documents."""
    from src.rag.vector_store_management import load_unified_vector_store, smart_update_vector_store

    print("🚀 BBS Vector Store Setup - Smart Update")
    print("=" * 50)
    
    if not validate_paths(main_directory, vector_store_path):
        return False
    
    print(f"📁 Main directory: {main_directory}")
    print(f"💾 Vector store: {vector_store_path}")
    print(f"🤖 Embedding model: {model}")
    if dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
    
    # Smart update the vector store
    print("\n🔄 Smart updating vector store...")
    vector_store = smart_update_vector_store(main_directory, vector_store_path, model, dry_run)
    
    if vector_store:
        print("\n✅ Vector store updated successfully!")
        
        # Test the updated vector store
        print("\n🧪 Testing updated vector store...")
        loaded_store = load_unified_vector_store(vector_store_path)
        
        if loaded_store:
            print("✅ Updated vector store can be loaded successfully!")
            print("\n🎉 Update complete! Your chatbot is ready to use.")
            return True
        else:
            print("❌ Failed to load the updated vector store")
            return False
    else:
        print("❌ Failed to update vector store")
        return False


def sync_vector_store_metadata_from_doc_list(vector_store_path: str):
    """Update selected metadata fields in the vector store from doc_list.json."""
    from src.rag.vector_store_management import load_doc_list, sync_metadata

    print("🚀 BBS Vector Store Metadata Sync")
    print("=" * 50)
    if not validate_paths(vector_store_path, vector_store_path):
        return False

    doc_list_path = os.path.join(vector_store_path, "doc_list.json")
    if not os.path.exists(doc_list_path):
        print(f"❌ doc_list.json not found: {doc_list_path}")
        return False

    try:
        load_doc_list(doc_list_path)
    except Exception as e:
        print(f"❌ Failed to validate doc_list.json: {str(e)}")
        return False

    
    result = sync_metadata(vector_store_path, client =None, model="qwen3-embedding-4b")
    return bool(result.get("success"))


def refresh_document_index_only(vector_store_path: str):
    """Rebuild only the document index chunk(s) and store them in the vector store."""
    from langchain_community.vectorstores import Chroma
    from src.llm_client import get_client
    import streamlit as st
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from src.rag.vector_store_management import (
        OpenAIEmbeddingsWrapper,
        check_vector_store_status,
        create_doc_index_document_chunks,
        delete_document_from_vector_store,
    )

    print("🚀 BBS Vector Store Document Index Refresh")
    print("=" * 50)

    if not validate_paths(vector_store_path, vector_store_path):
        return False

    doc_list_path = os.path.join(vector_store_path, "doc_list.json")
    if not os.path.exists(doc_list_path):
        print(f"❌ doc_list.json not found: {doc_list_path}")
        return False

    client = get_client(
        base_url="https://chat-ai.academiccloud.de/v1",
        api_key=st.secrets["KISSKI_API_KEY"],
    )
    model = "qwen3-embedding-4b"
    embeddings = OpenAIEmbeddingsWrapper(client, model)
    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)

    _, metadata_count = check_vector_store_status(vector_store_path, embeddings)
    print(f"📊 Vector store currently contains {metadata_count} metadata entries")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    index_chunks = create_doc_index_document_chunks(
        __import__("json").load(open(doc_list_path, "r", encoding="utf-8")),
        text_splitter,
    )

    if not index_chunks:
        print("❌ No document index chunks were created")
        return False

    delete_document_from_vector_store(
        vector_store_path,
        embeddings,
        index_chunks[0].metadata.get("source", "document_index"),
    )

    try:
        vector_store.add_documents(index_chunks)
    except Exception as e:
        print(f"❌ Failed to add document index chunks: {str(e)}")
        return False

    print(f"✅ Added {len(index_chunks)} document index chunk(s) to the vector store")
    return True



def main():
    """Main CLI entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return False
    
    try:
        if args.command == 'fresh':
            return create_fresh_vector_store(args.data_folder, args.output_dir, args.model)
        elif args.command == 'extend':
            return extend_vector_store(args.data_folder, args.vector_store, args.model, args.dry_run)
        elif args.command == 'update':
            return smart_update_vector_store_cmd(args.main_directory, args.vector_store, args.model, args.dry_run)
        elif args.command == 'sync-metadata':
            return sync_vector_store_metadata_from_doc_list(args.vector_store)
        elif args.command == 'doc-index':
            return refresh_document_index_only(args.vector_store)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            return False
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    print("\n🚀 Ready to run your BBS chatbot!")
