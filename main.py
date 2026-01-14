#!/usr/bin/env python3
"""
Main script to run the Fashion RAG system.
Provides a command-line interface for asking fashion questions.
"""

import os
import sys
import argparse
from typing import Optional
from dotenv import load_dotenv
from fashion_rag import FashionRAG


def initialize_rag(api_key: Optional[str] = None, force_rebuild: bool = False) -> FashionRAG:
    """
    Initialize the RAG system, creating or loading the knowledge base.
    
    Args:
        api_key: OpenAI API key.
        force_rebuild: If True, rebuild the knowledge base even if it exists.
        
    Returns:
        Initialized FashionRAG instance.
    """
    rag = FashionRAG(openai_api_key=api_key)
    
    persist_dir = "./chroma_db"
    
    # Check if knowledge base exists and whether to rebuild
    if force_rebuild or not os.path.exists(persist_dir):
        print("Setting up knowledge base...")
        rag.setup_knowledge_base()
    else:
        print("Loading existing knowledge base...")
        rag.load_knowledge_base()
    
    # Create QA chain
    rag.create_qa_chain()
    
    return rag


def interactive_mode(rag: FashionRAG):
    """
    Run the RAG system in interactive mode.
    
    Args:
        rag: Initialized FashionRAG instance.
    """
    print("\n" + "="*60)
    print("Fashion RAG System - Interactive Mode")
    print("="*60)
    print("Ask fashion-related questions. Type 'quit' or 'exit' to stop.")
    print("Type 'sources' to toggle source document display.")
    print("="*60 + "\n")
    
    show_sources = False
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using Fashion RAG! Goodbye!")
                break
                
            if question.lower() == 'sources':
                show_sources = not show_sources
                status = "enabled" if show_sources else "disabled"
                print(f"Source document display {status}.")
                continue
            
            print("\nThinking...")
            answer = rag.ask_question(question, include_sources=show_sources)
            print(f"\nAnswer: {answer}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


def single_question_mode(rag: FashionRAG, question: str, show_sources: bool = False):
    """
    Answer a single question and exit.
    
    Args:
        rag: Initialized FashionRAG instance.
        question: The question to answer.
        show_sources: Whether to show source documents.
    """
    try:
        answer = rag.ask_question(question, include_sources=show_sources)
        print(f"\nQuestion: {question}")
        print(f"\nAnswer: {answer}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Fashion RAG - Answer fashion-related questions using RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python main.py
  
  # Ask a single question
  python main.py --question "What should I wear for a business casual meeting?"
  
  # Rebuild the knowledge base
  python main.py --rebuild
  
  # Show source documents
  python main.py --question "What are A-line dresses?" --sources
        """
    )
    
    parser.add_argument(
        '-q', '--question',
        type=str,
        help='Ask a single question and exit'
    )
    
    parser.add_argument(
        '-s', '--sources',
        action='store_true',
        help='Show source documents with answers'
    )
    
    parser.add_argument(
        '-r', '--rebuild',
        action='store_true',
        help='Rebuild the knowledge base from scratch'
    )
    
    parser.add_argument(
        '-k', '--api-key',
        type=str,
        help='OpenAI API key (can also be set via OPENAI_API_KEY env variable)'
    )
    
    args = parser.parse_args()
    
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OpenAI API key is required.", file=sys.stderr)
        print("Set it via --api-key argument or OPENAI_API_KEY environment variable.", file=sys.stderr)
        print("\nYou can create a .env file with:", file=sys.stderr)
        print("OPENAI_API_KEY=your-api-key-here", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Initialize RAG system
        rag = initialize_rag(api_key=api_key, force_rebuild=args.rebuild)
        
        # Run in appropriate mode
        if args.question:
            single_question_mode(rag, args.question, show_sources=args.sources)
        else:
            interactive_mode(rag)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
