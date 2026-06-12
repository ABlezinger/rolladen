#!/usr/bin/env python3
"""
Script to extend an existing unified vector store with new documents.

Usage:
    python extend_vector_store.py

This script will:
1. Check if an existing vector store exists
2. Scan for new documents in the specified folder
3. Add only new documents (avoiding duplicates)
4. Test the updated vector store

For command line usage:
    python extend_vector_store.py /path/to/new/documents [vector_store_path]
"""

import os
import sys
import argparse
from vector_store_management import extend_existing_vector_store, load_unified_vector_store, test_embeddings_search, OpenAIEmbeddingsWrapper
from openai import OpenAI
import streamlit as st

def main():
    parser = argparse.ArgumentParser(description='Extend existing vector store with new documents')
    parser.add_argument('data_folder', nargs='?', default='../../BBS/data/Projekt KI Demonstrator - Unterlagen/Unterrichtsmaterialien',
                       help='Path to folder containing new documents to add')
    parser.add_argument('vector_store_path', nargs='?', default='kisski_db_v2',
                       help='Path to existing vector store directory')
    parser.add_argument('--test', action='store_true',
                       help='Test the vector store after extension')
    
    args = parser.parse_args()
    
    print("🔄 BBS Vector Store Extension")
    print("=" * 50)
    
    # Check if data folder exists
    if not os.path.exists(args.data_folder):
        print(f"❌ Data folder not found: {args.data_folder}")
        print("Please provide a valid path to the folder containing new documents.")
        return False
    
    # Check if vector store exists
    if not os.path.exists(args.vector_store_path):
        print(f"❌ Vector store not found: {args.vector_store_path}")
        print("Please create a vector store first using setup_vector_store.py")
        return False
    
    print(f"📁 New documents folder: {args.data_folder}")
    print(f"💾 Existing vector store: {args.vector_store_path}")
    
    # Test current vector store status
    print("\n📊 Current vector store status:")
    try:
        current_store = load_unified_vector_store(args.vector_store_path)
        if current_store:
            # Test a simple search to see current document count
            test_results = current_store.similarity_search("test", k=1)
            print(f"   Current documents: {len(test_results)}+ (estimated)")
        else:
            print("   ⚠️ Could not load current vector store")
    except Exception as e:
        print(f"   ⚠️ Error checking current status: {str(e)}")
    
    # Extend the vector store
    print(f"\n🔄 Extending vector store with new documents...")
    updated_store = extend_existing_vector_store(args.data_folder, args.vector_store_path)
    
    if updated_store:
        print("\n✅ Vector store extended successfully!")
        
        # Test the updated vector store
        if args.test:
            print("\n🧪 Testing updated vector store...")
            try:
                client = OpenAI(
                    base_url="https://chat-ai.academiccloud.de/v1",
                    api_key=st.secrets["KISSKI_API_KEY"]
                )
                embeddings = OpenAIEmbeddingsWrapper(client, "qwen3-embedding-4b")
                
                # Test with a few queries
                test_queries = [
                    "Kalthämmern Metalltechnik",
                    "Elektrotechnik Grundlagen",
                    "Fahrzeugtechnik Motor"
                ]
                
                for query in test_queries:
                    print(f"\n🔍 Testing query: '{query}'")
                    results = updated_store.similarity_search(query, k=2)
                    print(f"   Found {len(results)} results")
                    
                    if results:
                        for i, doc in enumerate(results, 1):
                            folder = doc.metadata.get('folder', 'Unknown')
                            content_preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
                            print(f"   Result {i}: {folder} - {content_preview}")
                    else:
                        print("   ⚠️ No results found")
                        
            except Exception as e:
                print(f"❌ Error testing updated vector store: {str(e)}")
        
        print(f"\n🎉 Extension complete! Your chatbot can now access the new documents.")
        return True
    else:
        print("❌ Failed to extend vector store")
        return False

def interactive_extend():
    """
    Interactive mode for extending vector store.
    """
    print("🔄 Interactive Vector Store Extension")
    print("=" * 50)
    
    # Get data folder
    data_folder = input("Enter path to folder with new documents (or press Enter for default): ").strip()
    if not data_folder:
        data_folder = "../../BBS/data/Projekt KI Demonstrator - Unterlagen/Unterrichtsmaterialien"
    
    # Get vector store path
    vector_store_path = input("Enter path to existing vector store (or press Enter for default): ").strip()
    if not vector_store_path:
        vector_store_path = "kisski_db_v2"
    
    # Ask if user wants to test
    test_choice = input("Test the vector store after extension? (y/n): ").lower().startswith('y')
    
    # Run extension
    success = extend_existing_vector_store(data_folder, vector_store_path)
    
    if success:
        print("\n✅ Extension completed!")
        
        if test_choice:
            print("\n🧪 Testing vector store...")
            try:
                client = OpenAI(
                    base_url="https://chat-ai.academiccloud.de/v1",
                    api_key=st.secrets["KISSKI_API_KEY"]
                )
                embeddings = OpenAIEmbeddingsWrapper(client, "qwen3-embedding-4b")
                
                test_embeddings_search(vector_store_path, embeddings, "test query", k=3)
            except Exception as e:
                print(f"❌ Error testing: {str(e)}")
    else:
        print("❌ Extension failed")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run interactive mode
        success = interactive_extend()
    else:
        # Arguments provided, run command line mode
        success = main()
    
    if not success:
        sys.exit(1)
    print("\n🚀 Ready to use your extended vector store!")
