#!/usr/bin/env python3
"""
Test script to verify the vector store is working correctly.
Run this to check if your vector store can be loaded and searched.
"""

import os
import sys
from vector_store_management import load_unified_vector_store, test_embeddings_search, OpenAIEmbeddingsWrapper
from openai import OpenAI
import streamlit as st

def test_vector_store():
    print("🧪 Testing Vector Store Functionality")
    print("=" * 50)
    
    # Test 1: Load vector store
    print("\n1️⃣ Testing vector store loading...")
    vector_store = load_unified_vector_store("kisski_db_v3")
    
    if vector_store is None:
        print("❌ Failed to load vector store")
        return False
    
    print("✅ Vector store loaded successfully!")
    
    # Test 2: Test search functionality
    print("\n2️⃣ Testing search functionality...")
    
    test_queries = [
        "Was ist Kalthämmern?",
        "Elektrotechnik Grundlagen",
        "Fahrzeugtechnik Motor",
        "Mathematik Formeln"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        try:
            results = vector_store.similarity_search(query, k=3)
            print(f"   Found {len(results)} results")
            
            if results:
                for i, doc in enumerate(results, 1):
                    folder = doc.metadata.get('folder', 'Unknown')
                    source = doc.metadata.get('source', 'Unknown')
                    content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    print(f"   Result {i}: {folder} - {content_preview}")
            else:
                print("   ⚠️ No results found")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Test 3: Test with embeddings function directly
    print("\n3️⃣ Testing with embeddings function...")
    try:
        client = OpenAI(
            base_url="https://chat-ai.academiccloud.de/v1",
            api_key=st.secrets["KISSKI_API_KEY"]
        )
        embeddings = OpenAIEmbeddingsWrapper(client, "qwen3-embedding-4b")
        
        # Test the search function
        test_embeddings_search("kisski_db_v3", embeddings, "Was ist Kalthämmern?", k=3)
        
    except Exception as e:
        print(f"❌ Error testing embeddings: {str(e)}")
        return False
    
    print("\n✅ All tests completed!")
    return True

if __name__ == "__main__":
    success = test_vector_store()
    if not success:
        sys.exit(1)
    print("\n🎉 Vector store is working correctly!")
