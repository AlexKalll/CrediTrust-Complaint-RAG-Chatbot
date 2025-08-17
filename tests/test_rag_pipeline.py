import pytest
from pathlib import Path
import pandas as pd
import faiss
from src.rag_pipeline import RAGPipeline

# Mock data and paths for testing
@pytest.fixture
def mock_vectorstore_paths(tmp_path):
    # Create dummy files for testing
    index_path = tmp_path / "faiss_index.bin"
    metadata_path = tmp_path / "metadata.parquet"

    # Create a dummy FAISS index
    d = 128  # dimension
    nb = 100 # database size
    xb = faiss.rand((nb, d)).astype('float32')
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    faiss.write_index(index, str(index_path))

    # Create dummy metadata
    metadata_df = pd.DataFrame({
        'complaint_id': range(nb),
        'product': ['Product A'] * (nb // 2) + ['Product B'] * (nb - nb // 2),
        'issue': [f'Issue {i}' for i in range(nb)],
        'chunk_text': [f'This is a test chunk {i}' for i in range(nb)]
    })
    metadata_df.to_parquet(metadata_path)

    return index_path, metadata_path

def test_rag_pipeline_initialization(mock_vectorstore_paths):
    index_path, metadata_path = mock_vectorstore_paths
    rag_pipeline = RAGPipeline(index_path=index_path, metadata_path=metadata_path)

    assert rag_pipeline.index is not None
    assert rag_pipeline.metadata is not None
    assert not rag_pipeline.metadata.empty

def test_rag_pipeline_query(mock_vectorstore_paths):
    index_path, metadata_path = mock_vectorstore_paths
    rag_pipeline = RAGPipeline(index_path=index_path, metadata_path=metadata_path)

    question = "What are the common issues with Product A?"
    result = rag_pipeline.query(question, product_filter="Product A")

    assert "response_streamer" in result
    assert "sources" in result
    assert "generation_thread" in result
    assert isinstance(result['sources'], list)
    assert len(result['sources']) > 0

    # Check if sources are filtered by product
    for source in result['sources']:
        assert source['product'] == 'Product A'

    # Test with 'All Products' filter
    result_all = rag_pipeline.query(question, product_filter="All Products")
    assert len(result_all['sources']) > 0

    # Test with no matching product
    result_no_match = rag_pipeline.query(question, product_filter="NonExistentProduct")
    assert len(result_no_match['sources']) == 0