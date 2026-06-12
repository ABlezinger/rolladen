#!/usr/bin/env python3
"""
Script to fix the current vector store issue by properly copying the unified vector store
to the main directory so the chatbot can access it.

This script will:
1. Check if unified vector store exists
2. Copy it to the main directory
3. Test that the chatbot can now access the documents
"""

import os
import shutil
import sys
from vector_store_management import load_unified_vector_store, test_embeddings_search, OpenAIEmbeddingsWrapper
from openai import OpenAI
import streamlit as st

def fix_vector_store():
    print("🔧 Fixing Vector Store Access")
    print("=" * 50)
    
    persist_directory = "kisski_db_v2"
    unified_path = os.path.join(persist_directory, "unified")
    
    # Check if unified vector store exists
    if not os.path.exists(unified_path):
        print(f"❌ Unified vector store not found at: {unified_path}")
        print("Please run setup_vector_store.py first to create the unified vector store.")
        return False
    
    print(f"✅ Found unified vector store at: {unified_path}")
    
    # Check if main directory exists
    if not os.path.exists(persist_directory):
        print(f"❌ Main directory not found: {persist_directory}")
        return False
    
    print(f"✅ Found main directory: {persist_directory}")
    
    # Backup current main directory content
    backup_dir = f"{persist_directory}_backup"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    
    print(f"📦 Creating backup of current main directory...")
    shutil.copytree(persist_directory, backup_dir)
    print(f"✅ Backup created at: {backup_dir}")
    
    # Clean main directory (keep unified folder)
    print(f"🧹 Cleaning main directory...")
    for item in os.listdir(persist_directory):
        item_path = os.path.join(persist_directory, item)
        if item != "unified":
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)
    
    # Copy unified vector store to main directory
    print(f"📋 Copying unified vector store to main directory...")
    try:
        # Copy all files from unified to main directory
        for item in os.listdir(unified_path):
            src = os.path.join(unified_path, item)
            dst = os.path.join(persist_directory, item)
            
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        print("✅ Unified vector store copied to main directory!")
        
    except Exception as e:
        print(f"❌ Error copying unified vector store: {str(e)}")
        return False
    
    # Test the fixed vector store
    print(f"\n🧪 Testing fixed vector store...")
    try:
        client = OpenAI(
            base_url="https://chat-ai.academiccloud.de/v1",
            api_key=st.secrets["KISSKI_API_KEY"]
        )
        embeddings = OpenAIEmbeddingsWrapper(client, "qwen3-embedding-4b")
        
        # Test loading
        vector_store = load_unified_vector_store(persist_directory)
        if vector_store is None:
            print("❌ Still cannot load vector store after fix")
            return False
        
        print("✅ Vector store loads successfully!")
        
        # Test search
        test_queries = [
            "Kalthämmern Metalltechnik",
            "Elektrotechnik Grundlagen",
            "Fahrzeugtechnik Motor"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            results = vector_store.similarity_search(query, k=2)
            print(f"   Found {len(results)} results")
            
            if results:
                for i, doc in enumerate(results, 1):
                    folder = doc.metadata.get('folder', 'Unknown')
                    source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    content_preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
                    print(f"   Result {i}: {folder}/{source} - {content_preview}")
            else:
                print("   ⚠️ No results found")
        
        print(f"\n🎉 Vector store is now working correctly!")
        print(f"📁 Main directory: {persist_directory}")
        print(f"📦 Backup directory: {backup_dir}")
        print(f"🔄 Unified directory: {unified_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing fixed vector store: {str(e)}")
        return False

def main():
    success = fix_vector_store()
    
    if success:
        print("\n✅ Fix completed successfully!")
        print("Your chatbot should now be able to access all documents.")
        print("\nTo test the chatbot:")
        print("  streamlit run bbs_streaming.py")
    else:
        print("\n❌ Fix failed!")
        print("Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
