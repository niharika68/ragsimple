"""
Simple RAG (Retrieval-Augmented Generation) system for fashion questions.

This module implements a basic RAG system using ChromaDB for vector storage
and retrieval. It can answer fashion-related questions by retrieving relevant
information from a knowledge base and generating contextual answers.
"""

import os
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from fashion_data import FASHION_DOCUMENTS


class FashionRAG:
    """
    A simple RAG system for answering fashion-related questions.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, persist_directory: str = "./chroma_db"):
        """
        Initialize the Fashion RAG system.
        
        Args:
            openai_api_key: OpenAI API key for embeddings and LLM. 
                          If None, will look for OPENAI_API_KEY environment variable.
            persist_directory: Directory to persist the vector database.
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.vectorstore = None
        self.qa_chain = None
        
    def setup_knowledge_base(self, documents: List[str] = None):
        """
        Set up the knowledge base by creating embeddings and storing them.
        
        Args:
            documents: List of text documents to use as knowledge base.
                      If None, uses the default FASHION_DOCUMENTS.
        """
        # Use provided documents or default fashion documents
        docs = documents if documents is not None else FASHION_DOCUMENTS
        
        # Create Document objects
        doc_objects = [Document(page_content=doc) for doc in docs]
        
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        split_docs = text_splitter.split_documents(doc_objects)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Persist the database
        self.vectorstore.persist()
        
        print(f"Knowledge base created with {len(split_docs)} document chunks.")
        
    def load_knowledge_base(self):
        """
        Load an existing knowledge base from disk.
        """
        if not os.path.exists(self.persist_directory):
            raise ValueError(f"No persisted database found at {self.persist_directory}. "
                           "Please run setup_knowledge_base() first.")
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        print("Knowledge base loaded from disk.")
        
    def create_qa_chain(self, temperature: float = 0.0, model: str = "gpt-3.5-turbo"):
        """
        Create the question-answering chain.
        
        Args:
            temperature: Temperature for LLM generation (0.0 = deterministic).
            model: OpenAI model to use.
        """
        if self.vectorstore is None:
            raise ValueError("Knowledge base not set up. Run setup_knowledge_base() or load_knowledge_base() first.")
        
        # Create LLM
        llm = ChatOpenAI(
            temperature=temperature,
            model_name=model,
            openai_api_key=self.api_key
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
            ),
            return_source_documents=True
        )
        
        print("QA chain created successfully.")
        
    def ask_question(self, question: str, include_sources: bool = False) -> str:
        """
        Ask a fashion-related question and get an answer.
        
        Args:
            question: The fashion question to ask.
            include_sources: Whether to include source documents in the response.
            
        Returns:
            The answer to the question.
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not created. Run create_qa_chain() first.")
        
        result = self.qa_chain({"query": question})
        answer = result["result"]
        
        if include_sources:
            sources = result.get("source_documents", [])
            if sources:
                answer += "\n\nSources:\n"
                for i, doc in enumerate(sources, 1):
                    preview = doc.page_content[:100].replace("\n", " ")
                    answer += f"{i}. {preview}...\n"
        
        return answer
    
    def retrieve_similar_docs(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve documents similar to the query.
        
        Args:
            query: The search query.
            k: Number of documents to retrieve.
            
        Returns:
            List of similar documents.
        """
        if self.vectorstore is None:
            raise ValueError("Knowledge base not set up.")
        
        return self.vectorstore.similarity_search(query, k=k)
