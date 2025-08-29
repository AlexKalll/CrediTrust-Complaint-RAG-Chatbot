#!/usr/bin/env python3
"""
Test script to verify the project setup and identify issues.
"""

import sys
from pathlib import Path
import traceback


def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")

    try:
        import pandas as pd

        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pandas: {e}")
        return False

    try:
        import numpy as np

        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        return False

    try:
        import faiss

        print("✓ faiss imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import faiss: {e}")
        return False

    try:
        from sentence_transformers import SentenceTransformer

        print("✓ sentence_transformers imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import sentence_transformers: {e}")
        return False

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        print("✓ langchain imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import langchain: {e}")
        return False

    try:
        from langchain_community.llms import LlamaCpp

        print("✓ langchain_community imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import langchain_community: {e}")
        return False

    try:
        from transformers import pipeline

        print("✓ transformers imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import transformers: {e}")
        return False

    try:
        import gradio as gr

        print("✓ gradio imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import gradio: {e}")
        return False

    try:
        import streamlit as st

        print("✓ streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import streamlit: {e}")
        return False

    return True


def test_data_files():
    """Test if required data files exist."""
    print("\nTesting data files...")

    root_dir = Path(__file__).parent

    # Check if data directory exists
    data_dir = root_dir / "data"
    if data_dir.exists():
        print("✓ data directory exists")

        # Check for filtered complaints
        complaints_file = data_dir / "filtered_complaints.csv"
        if complaints_file.exists():
            print(
                f"✓ filtered_complaints.csv exists ({complaints_file.stat().st_size / (1024*1024):.1f} MB)"
            )
        else:
            print("✗ filtered_complaints.csv not found")
            return False
    else:
        print("✗ data directory not found")
        return False

    # Check if vectorstore directory exists
    vectorstore_dir = root_dir / "vectorstore"
    if vectorstore_dir.exists():
        print("✓ vectorstore directory exists")

        # Check for FAISS index
        faiss_index = vectorstore_dir / "faiss_index.bin"
        if faiss_index.exists():
            print(f"✓ FAISS index exists ({faiss_index.stat().st_size / 1024:.1f} KB)")
        else:
            print("✗ FAISS index not found")
            return False

        # Check for metadata
        metadata_file = vectorstore_dir / "metadata.parquet"
        if metadata_file.exists():
            print(
                f"✓ metadata.parquet exists ({metadata_file.stat().st_size / 1024:.1f} KB)"
            )
        else:
            print("✗ metadata.parquet not found")
            return False
    else:
        print("✗ vectorstore directory not found")
        return False

    return True


def test_source_code():
    """Test if source code can be imported."""
    print("\nTesting source code imports...")

    try:
        sys.path.append(str(Path(__file__).parent / "src"))

        from src.data_preprocessing import load_data, preprocess_data

        print("✓ data_preprocessing imported successfully")
    except Exception as e:
        print(f"✗ Failed to import data_preprocessing: {e}")
        traceback.print_exc()
        return False

    try:
        from src.text_chunking_embedding import TextChunkingEmbedding

        print("✓ text_chunking_embedding imported successfully")
    except Exception as e:
        print(f"✗ Failed to import text_chunking_embedding: {e}")
        traceback.print_exc()
        return False

    try:
        from src.rag_pipeline import RAGPipeline

        print("✓ rag_pipeline imported successfully")
    except Exception as e:
        print(f"✗ Failed to import rag_pipeline: {e}")
        traceback.print_exc()
        return False

    return True


def test_rag_pipeline():
    """Test if RAG pipeline can be initialized."""
    print("\nTesting RAG pipeline initialization...")

    try:
        from src.rag_pipeline import RAGPipeline

        root_dir = Path(__file__).parent
        index_path = root_dir / "vectorstore" / "faiss_index.bin"
        metadata_path = root_dir / "vectorstore" / "metadata.parquet"

        rag = RAGPipeline(index_path, metadata_path)
        print("✓ RAG pipeline initialized successfully")

        # Test a simple query
        try:
            result = rag.query("What are common credit card issues?")
            print("✓ RAG pipeline query successful")
            print(f"  Response length: {len(result['response'])} characters")
            print(f"  Sources found: {len(result['sources'])}")
        except Exception as e:
            print(f"✗ RAG pipeline query failed: {e}")
            return False

    except Exception as e:
        print(f"✗ Failed to initialize RAG pipeline: {e}")
        traceback.print_exc()
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("CreditTrust RAG Chatbot - Setup Test")
    print("=" * 50)

    all_tests_passed = True

    # Test imports
    if not test_imports():
        all_tests_passed = False

    # Test data files
    if not test_data_files():
        all_tests_passed = False

    # Test source code
    if not test_source_code():
        all_tests_passed = False

    # Test RAG pipeline
    if not test_rag_pipeline():
        all_tests_passed = False

    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! The project is ready to run.")
        print("\nTo run the app:")
        print("  Streamlit: streamlit run dashboards/streamlit_app.py")
        print("  Gradio:    python dashboards/app.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
