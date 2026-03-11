import argparse
import asyncio
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from src.engine import SupportEngine

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_argparser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Support Ticket RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Streamlit UI
  python main.py --ui
  
  # Initialize and load data
  python main.py --init
  
  # Query from CLI
  python main.py --query "How do I reset my password?" --type technical
  
  # Force reload data
  python main.py --init --reload
        """
    )
    
    parser.add_argument(
        '--ui',
        action='store_true',
        help='Launch Streamlit UI'
    )
    
    parser.add_argument(
        '--init',
        action='store_true',
        help='Initialize the engine and load documents'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Force reload all documents (use with --init)'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Query string to search for similar tickets'
    )
    
    parser.add_argument(
        '--type',
        type=str,
        choices=['technical', 'product', 'customer'],
        help='Support type filter (technical, product, or customer)'
    )
    
    parser.add_argument(
        '--results',
        type=int,
        default=3,
        help='Number of similar tickets to retrieve (default: 3)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data',
        help='Path to data directory (default: data)'
    )
    
    parser.add_argument(
        '--vector-store',
        type=str,
        default='vector_store',
        help='Path to vector store directory (default: vector_store)'
    )
    
    return parser

async def run_cli_query(engine: SupportEngine, query: str, support_type: str = None, k: int = 3):
    """Run a query from CLI and display results."""
    try:
        rag_chain = engine.get_rag_chain()
        
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Support Type: {support_type or 'All'}")
        print(f"{'='*80}\n")
        
        # Get relevant documents
        print("🔍 Searching for relevant tickets...\n")
        docs = rag_chain.get_relevant_documents(query=query, support_type=support_type, k=k)
        
        if not docs:
            print("⚠️  No relevant tickets found.\n")
            return
        
        # Display tickets
        print(f"📋 Found {len(docs)} Relevant Ticket(s):\n")
        for idx, doc in enumerate(docs, 1):
            metadata = doc.get('metadata', {})
            content = doc.get('content', '')
            
            print(f"{'─'*80}")
            print(f"Ticket {idx}: {metadata.get('ticket_id', 'Unknown')}")
            print(f"{'─'*80}")
            print(f"Support Type: {metadata.get('support_type', 'N/A').title()}")
            print(f"Priority: {metadata.get('priority', 'N/A')}")
            print(f"Type: {metadata.get('type', 'N/A')}")
            print(f"Queue: {metadata.get('queue', 'N/A')}")
            
            tags = metadata.get('tags', [])
            if tags:
                print(f"Tags: {', '.join(tags)}")
            
            print(f"\nContent:\n{content}\n")
        
        # Generate AI response
        print(f"{'='*80}")
        print("🤖 Generating AI Response...")
        print(f"{'='*80}\n")
        
        response = await rag_chain.query(query=query, support_type=support_type)
        print(f"💡 AI Assistant Response:\n\n{response}\n")
        print(f"{'='*80}\n")
        
    except ValueError as ve:
        print(f"Validation Error: {str(ve)}\n")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing query: {str(e)}\n")
        logger.error(f"Query processing error: {e}", exc_info=True)
        sys.exit(1)

def main():
    """Main entry point."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n💡 Tip: Use --ui to launch the Streamlit interface")
        sys.exit(0)
    
    # Launch Streamlit UI
    if args.ui:
        import subprocess
        print("🚀 Launching Streamlit UI...")
        subprocess.run(["streamlit", "run", "app.py"])
        return
    
    # Initialize engine
    try:
        print("🔧 Initializing Support Engine...")
        engine = SupportEngine(
            data_path=args.data_path,
            persist_directory=args.vector_store
        )
        
        if args.init or args.reload:
            print("📚 Loading documents...")
            engine.initialize(force_reload=args.reload)
            print("✅ Engine initialized successfully!")
            
            if not args.query:
                print("\n💡 Engine is ready. Use --query to search for tickets.")
                return
        else:
            # Just initialize without reloading
            engine.initialize(force_reload=False)
        
        # Run query if provided
        if args.query:
            asyncio.run(run_cli_query(
                engine=engine,
                query=args.query,
                support_type=args.type,
                k=args.results
            ))
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        logger.error(f"Main execution error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
