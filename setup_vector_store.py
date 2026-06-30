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
"""

import os
import sys
import argparse
from pathlib import Path
from rag.vector_store_management import (
    create_fresh_unified_vector_store, 
    load_unified_vector_store,
    extend_existing_vector_store,
    smart_update_vector_store
)

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
