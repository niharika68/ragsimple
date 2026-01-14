"""
Simple tests for the Fashion RAG system.
Note: These tests require a valid OpenAI API key.
"""

import os
import pytest
from fashion_rag import FashionRAG
from fashion_data import FASHION_DOCUMENTS


def test_fashion_documents_loaded():
    """Test that fashion documents are available."""
    assert len(FASHION_DOCUMENTS) > 0
    assert all(isinstance(doc, str) for doc in FASHION_DOCUMENTS)
    assert all(len(doc) > 0 for doc in FASHION_DOCUMENTS)


def test_fashion_rag_initialization_without_api_key():
    """Test that FashionRAG raises error without API key."""
    # Save current env var
    old_key = os.environ.get("OPENAI_API_KEY")
    
    # Remove API key from environment
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="OpenAI API key"):
        FashionRAG()
    
    # Restore env var
    if old_key:
        os.environ["OPENAI_API_KEY"] = old_key


def test_fashion_rag_initialization_with_api_key():
    """Test that FashionRAG initializes with API key."""
    # This test requires a valid API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    
    rag = FashionRAG(openai_api_key=api_key)
    assert rag.api_key == api_key
    assert rag.embeddings is not None


def test_document_chunking():
    """Test that documents are properly chunked."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    
    # Create sample document
    sample_doc = "This is a test. " * 100
    doc_objects = [Document(page_content=sample_doc)]
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    split_docs = text_splitter.split_documents(doc_objects)
    
    # Check that splitting occurred
    assert len(split_docs) > 0
    
    # Check chunk sizes
    for chunk in split_docs:
        assert len(chunk.page_content) <= 500 + 50  # Allow for overlap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
