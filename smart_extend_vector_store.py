#!/usr/bin/env python3
"""
Smart Vector Store Extension - Enhanced version for main directory with subfolders.

This script intelligently extends an existing vector store by:
1. Scanning a main directory and all its subfolders
2. Identifying only NEW documents that aren't already indexed
3. Processing only the new documents for embedding generation
4. Adding them to the existing vector store

Usage:
    python smart_extend_vector_store.py [main_directory] [vector_store_path]
"""

import os
import sys
import argparse
from vector_store_management import (
    extend_existing_vector_store, 
    load_unified_vector_store, 
    check_vector_store_status,
    OpenAIEmbeddingsWrapper,
    load_document,
    clean_text
)
from openai import OpenAI
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import streamlit as st

def smart_extend_vector_store(main_directory: str, persist_directory: str = "kisski_db_v3", model: str = "qwen3-embedding-4b", dry_run: bool = False):
    """
    Smart extension that scans main directory + subfolders for new documents only.
    
    Args:
        main_directory: Main directory containing subfolders with documents
        persist_directory: Path to existing vector store directory
        model: Embedding model to use
        dry_run: If True, only scan and report what would be added without actually doing it
    
    Returns:
        Updated Chroma vector store instance or None if failed (or dry run results if dry_run=True)
    """
    if dry_run:
        print(f"\n=== Smart Vector Store Extension (DRY RUN) ===")
        print("🔍 This is a preview mode - no changes will be made!")
    else:
        print(f"\n=== Smart Vector Store Extension ===")
    
    print(f"Main directory: {main_directory}")
    print(f"Vector store: {persist_directory}")
    print(f"Embedding model: {model}")
    
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
        
        # Load existing vector store (only if not dry run)
        if not dry_run:
            try:
                vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            except Exception as e:
                print(f"❌ Error loading vector store: {str(e)}")
                print("💡 The vector store may be corrupted or inaccessible")
                return None
        
        # Get existing sources to avoid duplicates
        try:
            existing_sources, metadata_count = check_vector_store_status(persist_directory, embeddings)
            print(f"📊 Current vector store: {metadata_count} documents from {len(existing_sources)} sources")
        except Exception as e:
            print(f"⚠️ Warning: Could not access vector store status: {str(e)}")
            print("🔍 Proceeding with empty existing sources list (will process all files)")
            existing_sources = set()
            metadata_count = 0
        
        # Scan main directory and all subfolders for new documents
        print(f"\n🔍 Scanning main directory and subfolders for new documents...")
        
        new_docs = []
        scanned_files = 0
        new_files = 0
        skipped_files = 0
        encrypted_files = 0
        
        # Walk through main directory and all subfolders
        try:
            for root, dirs, files in tqdm(os.walk(main_directory), desc="Scanning directories"):
                for file in files:
                    if file.lower().endswith((".pdf", ".docx", ".xlsx")):
                        scanned_files += 1
                        try:
                            file_path = os.path.join(root, file)
                        except UnicodeDecodeError as e:
                            print(f"🔒 Skipping file with encoding issues: {file}")
                            print(f"   Reason: Unicode decode error in file path")
                            encrypted_files += 1
                            continue
                    
                        if file_path not in existing_sources:
                            new_files += 1
                            print(f"📄 New document found: {file_path}")
                            
                            if dry_run:
                                # In dry run mode, just estimate document chunks without loading
                                print(f"   🔍 Would load and process this document")
                                # Estimate chunks (rough approximation)
                                estimated_chunks = 1  # Conservative estimate
                                new_docs.extend([None] * estimated_chunks)  # Placeholder
                            else:
                                try:
                                    docs = load_document(file_path)
                                    if docs:
                                        # Clean the text content
                                        for doc in docs:
                                            if hasattr(doc, 'page_content'):
                                                doc.page_content = clean_text(doc.page_content)
                                        new_docs.extend(docs)
                                        print(f"   ✅ Loaded {len(docs)} document chunks")
                                    else:
                                        print(f"   ⚠️ No content extracted from {file}")
                                except UnicodeDecodeError as e:
                                    print(f"   🔒 Skipping encrypted/corrupted file: {file_path}")
                                    print(f"      Reason: Unicode decode error - file may be encrypted or corrupted")
                                    encrypted_files += 1
                                except Exception as e:
                                    error_msg = str(e).lower()
                                    if any(keyword in error_msg for keyword in ['encrypted', 'password', 'protected', 'corrupted', 'decode', 'utf-8']):
                                        print(f"   🔒 Skipping encrypted/corrupted file: {file_path}")
                                        print(f"      Reason: {str(e)}")
                                        encrypted_files += 1
                                    else:
                                        print(f"   ❌ Error loading {file_path}: {str(e)}")
                        else:
                            skipped_files += 1
                            print(f"⏭️  Skipping existing document: {file_path}")
        except Exception as e:
            print(f"❌ Error during directory scanning: {str(e)}")
            print("💡 Some files may have encoding issues in their paths")
            # Continue with what we have so far
        
        # Summary of scan results
        print(f"\n📋 Scan Summary:")
        print(f"   📁 Total files scanned: {scanned_files}")
        print(f"   🆕 New files found: {new_files}")
        print(f"   ⏭️  Existing files skipped: {skipped_files}")
        if encrypted_files > 0:
            print(f"   🔒 Encrypted/corrupted files skipped: {encrypted_files}")
        if dry_run:
            print(f"   📄 Estimated document chunks to add: {len(new_docs)}")
        else:
            print(f"   📄 Total document chunks to add: {len(new_docs)}")
        
        if not new_docs:
            print("✅ No new documents found to add!")
            if dry_run:
                return {"new_files": new_files, "skipped_files": skipped_files, "estimated_chunks": 0}
            return vector_store
        
        if dry_run:
            # Dry run mode - just return summary
            print(f"\n🔍 DRY RUN SUMMARY:")
            print(f"   📁 Files that would be processed: {new_files}")
            print(f"   📄 Estimated chunks to add: {len(new_docs)}")
            print(f"   ⏭️ Files that would be skipped: {skipped_files}")
            if encrypted_files > 0:
                print(f"   🔒 Encrypted/corrupted files that would be skipped: {encrypted_files}")
            print(f"\n💡 To actually perform the extension, run without --dry-run flag")
            return {"new_files": new_files, "skipped_files": skipped_files, "encrypted_files": encrypted_files, "estimated_chunks": len(new_docs)}
        
        # Process new documents (only in non-dry-run mode)
        print(f"\n🔄 Processing {len(new_docs)} new document chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        
        split_docs = text_splitter.split_documents(new_docs)
        print(f"📄 Split into {len(split_docs)} chunks for embedding")
        
        if split_docs:
            batch_size = 100
            successful_batches = 0
            
            print(f"\n🚀 Adding new documents to vector store...")
            for i in tqdm(range(0, len(split_docs), batch_size), desc="Processing batches"):
                batch = split_docs[i:i + batch_size]
                try:
                    vector_store.add_documents(batch)
                    successful_batches += 1
                except Exception as e:
                    print(f"❌ Error adding batch {i // batch_size + 1}: {str(e)}")
                    return None
            
            print(f"✅ Successfully added {len(split_docs)} new document chunks in {successful_batches} batches")
            
            # Final status check
            final_sources, final_metadata_count = check_vector_store_status(persist_directory, embeddings)
            print(f"\n📊 Updated vector store: {final_metadata_count} documents from {len(final_sources)} sources")
            print(f"📈 Added: {final_metadata_count - metadata_count} new documents")
            
            return vector_store
        else:
            print("❌ No documents were split for addition")
            return None
            
    except Exception as e:
        print(f"❌ Error in smart extension: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Smart extend vector store with new documents from main directory')
    parser.add_argument('main_directory', nargs='?', 
                       default='../../BBS/data/Projekt KI Demonstrator - Unterlagen/Unterrichtsmaterialien',
                       help='Main directory containing subfolders with documents')
    parser.add_argument('vector_store_path', nargs='?', default='kisski_db_v2',
                       help='Path to existing vector store directory')
    parser.add_argument('--test', action='store_true',
                       help='Test the vector store after extension')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be added without actually doing it')
    
    args = parser.parse_args()
    
    print("🚀 Smart Vector Store Extension")
    print("=" * 50)
    
    # Check if main directory exists
    if not os.path.exists(args.main_directory):
        print(f"❌ Main directory not found: {args.main_directory}")
        print("Please provide a valid path to the main directory containing subfolders.")
        return False
    
    # Check if vector store exists
    if not os.path.exists(args.vector_store_path):
        print(f"❌ Vector store not found: {args.vector_store_path}")
        print("Please create a vector store first using setup_vector_store.py")
        return False
    
    print(f"📁 Main directory: {args.main_directory}")
    print(f"💾 Vector store: {args.vector_store_path}")
    
    # Show subfolders that will be scanned
    subfolders = [f for f in os.listdir(args.main_directory) 
                  if os.path.isdir(os.path.join(args.main_directory, f))]
    print(f"📂 Subfolders to scan: {subfolders}")
    
    # Run smart extension
    updated_store = smart_extend_vector_store(args.main_directory, args.vector_store_path, dry_run=args.dry_run)
    
    if updated_store:
        if args.dry_run:
            print("\n✅ Dry run completed successfully!")
            print("💡 No changes were made to your vector store.")
        else:
            print("\n✅ Smart extension completed successfully!")
        
        # Test the updated vector store (only if not dry run)
        if args.test and not args.dry_run:
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
                            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                            content_preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
                            print(f"   Result {i}: {folder}/{source} - {content_preview}")
                    else:
                        print("   ⚠️ No results found")
                        
            except Exception as e:
                print(f"❌ Error testing updated vector store: {str(e)}")
        
        if not args.dry_run:
            print(f"\n🎉 Smart extension complete!")
        return True
    else:
        print("❌ Smart extension failed")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    print("\n🚀 Your vector store is now up to date!")
