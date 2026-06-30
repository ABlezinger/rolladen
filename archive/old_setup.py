#!/usr/bin/env python3
"""
Simple script to set up a unified vector store for the BBS chatbot.

Usage:
    python old_setup.py

This script will:
1. Create a unified vector store from all documents in the data folder
2. Make it available for the chatbot to use
3. Test the search functionality

For extending with new documents later, use:
    from vector_store_management import extend_existing_vector_store
    extend_existing_vector_store("/path/to/new/documents", "kisski_db_v2")
"""

import os
import sys
from rag.vector_store_management import create_fresh_unified_vector_store, load_unified_vector_store

def main():
    print("🚀 BBS Vector Store Setup")
    print("=" * 50)
    
    # Default paths (adjust these for your setup)
    data_folder = "../../BBS/data/Projekt KI Demonstrator - Unterlagen/Unterrichtsmaterialien"
    persist_directory = "kisski_db_v3"
    
    # Check if data folder exists
    if not os.path.exists(data_folder):
        print(f"❌ Data folder not found: {data_folder}")
        print("Please update the 'data_folder' variable in this script with the correct path.")
        return False
    
    print(f"📁 Data folder: {data_folder}")
    print(f"💾 Vector store will be saved to: {persist_directory}")
    
    # Create the unified vector store
    print("\n🔄 Creating unified vector store...")
    vector_store = create_fresh_unified_vector_store(data_folder, persist_directory)
    
    if vector_store:
        print("\n✅ Vector store created successfully!")
        
        # Test loading the vector store
        print("\n🧪 Testing vector store loading...")
        loaded_store = load_unified_vector_store(persist_directory)
        
        if loaded_store:
            print("✅ Vector store can be loaded successfully!")
            print("\n🎉 Setup complete! Your chatbot is ready to use.")
            print(f"\nTo use in your chatbot, the vector store is available at: {persist_directory}")
            return True
        else:
            print("❌ Failed to load the created vector store")
            return False
    else:
        print("❌ Failed to create vector store")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    print("\n🚀 Ready to run your BBS chatbot!")