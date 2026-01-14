#!/usr/bin/env python3
"""
Demo script for the Fashion RAG system.
This script demonstrates the key features without requiring an OpenAI API key.
It shows the structure and data loading capabilities.
"""

from fashion_data import FASHION_DOCUMENTS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def main():
    """Demonstrate the Fashion RAG system structure."""
    
    print("=" * 60)
    print("Fashion RAG System - Demo")
    print("=" * 60)
    
    # Show knowledge base size
    print(f"\n1. Knowledge Base Statistics:")
    print(f"   - Total documents: {len(FASHION_DOCUMENTS)}")
    total_chars = sum(len(doc) for doc in FASHION_DOCUMENTS)
    print(f"   - Total characters: {total_chars:,}")
    print(f"   - Average document size: {total_chars // len(FASHION_DOCUMENTS):,} characters")
    
    # Show document topics
    print(f"\n2. Fashion Topics Covered:")
    topics = [
        "Clothing Types (Dresses, Tops, Bottoms)",
        "Color Coordination and Seasonal Palettes",
        "Fabrics and Materials",
        "Styling Tips for Different Body Types",
        "Accessorizing Guidelines",
        "Business Casual, Casual, and Formal Wear",
        "Current Fashion Trends",
        "Clothing Care and Maintenance"
    ]
    for topic in topics:
        print(f"   ✓ {topic}")
    
    # Show document chunking
    print(f"\n3. Document Processing:")
    doc_objects = [Document(page_content=doc) for doc in FASHION_DOCUMENTS]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    split_docs = text_splitter.split_documents(doc_objects)
    
    print(f"   - Original documents: {len(FASHION_DOCUMENTS)}")
    print(f"   - Split into chunks: {len(split_docs)}")
    print(f"   - Chunk size: 500 characters (with 50 overlap)")
    
    # Show a sample chunk
    print(f"\n4. Sample Knowledge Chunk:")
    print(f"   {'-' * 56}")
    sample = split_docs[0].page_content[:200]
    print(f"   {sample}...")
    print(f"   {'-' * 56}")
    
    # Show example queries
    print(f"\n5. Example Queries You Can Ask:")
    example_queries = [
        "What should I wear for a business casual meeting?",
        "How do I style an A-line dress?",
        "What colors are best for fall?",
        "What fabrics are good for summer?",
        "How do I care for silk clothing?",
        "What are current fashion trends?",
    ]
    for query in example_queries:
        print(f"   • {query}")
    
    print(f"\n6. To Use the Full RAG System:")
    print(f"   - Set your OpenAI API key in .env file")
    print(f"   - Run: python main.py")
    print(f"   - Or: python main.py --question \"your question\"")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
