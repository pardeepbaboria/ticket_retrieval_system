# RAG Ticket Support System

A comprehensive support ticket retrieval and analysis system powered by RAG (Retrieval-Augmented Generation) technology. This system enables intelligent search and AI-powered responses across technical, product, and customer support tickets.

## 🎯 Features

- **Semantic Search**: Find relevant support tickets using natural language queries
- **Multi-Format Support**: Process both JSON and XML ticket formats
- **AI-Powered Responses**: Get intelligent answers using GPT-4
- **Support Type Filtering**: Filter by technical, product, or customer support
- **Interactive UI**: Beautiful Streamlit interface for easy interaction
- **CLI Support**: Command-line interface for automation and scripting
- **Vector Store**: Efficient similarity search using ChromaDB
- **Persistent Storage**: Vector embeddings are cached for fast retrieval

## 🏗️ Architecture

```
ticket_retrieval_system/
├── src/
│   ├── document_loader.py    # Load and parse ticket documents
│   ├── vector_store.py        # ChromaDB vector store management
│   ├── rag_chain.py           # RAG chain with LangChain
│   └── engine/
│       └── __init__.py        # Main engine orchestration
├── data/                      # Ticket data files (JSON/XML)
├── vector_store/              # Persistent vector embeddings
├── app.py                     # Streamlit UI application
├── main.py                    # CLI entry point
└── pyproject.toml            # Project dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- OpenAI API Key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ticket_retrieval_system
```

2. Install dependencies using uv:
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

3. Create a `.env` file with your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Data Setup

Place your support ticket files in the `data/` directory:
- `Technical Support_tickets.json` / `Technical Support_tickets.xml`
- `Product Support_tickets.json` / `Product Support_tickets.xml`
- `Customer Service_tickets.json` / `Customer Service_tickets.xml`

## 💻 Usage

### Streamlit UI (Recommended)

Launch the interactive web interface:

```bash
python main.py --ui
```

Or directly:

```bash
streamlit run app.py
```

The UI provides:
- 🔍 Semantic search interface
- 📊 Query statistics and history
- 🎫 Formatted ticket display
- 🤖 AI-powered responses
- ⚙️ Configurable search parameters

### Command Line Interface

#### Initialize the system:
```bash
python main.py --init
```

#### Force reload all documents:
```bash
python main.py --init --reload
```

#### Query from CLI:
```bash
python main.py --query "Queries are running 10x slower than usual" --type technical --results 5
```

### CLI Options

```
--ui                    Launch Streamlit UI
--init                  Initialize engine and load documents
--reload                Force reload all documents
--query TEXT            Query string to search
--type {technical,product,customer}  Support type filter
--results N             Number of results (default: 3)
--data-path PATH        Data directory path (default: data)
--vector-store PATH     Vector store path (default: vector_store)
```

## 📚 API Usage

### Python API

```python
from src.engine import SupportEngine
import asyncio

# Initialize engine
engine = SupportEngine(
    data_path="data",
    persist_directory="vector_store"
)
engine.initialize()

# Get RAG chain
rag_chain = engine.get_rag_chain()

# Query tickets
async def query_tickets():
    # Get relevant documents
    docs = rag_chain.get_relevant_documents(
        query="How do I reset my password?",
        support_type="technical",
        k=3
    )
    
    # Get AI response
    response = await rag_chain.query(
        query="How do I reset my password?",
        support_type="technical"
    )
    print(response)

asyncio.run(query_tickets())
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Vector Store

The system uses ChromaDB for vector storage with OpenAI embeddings:
- Model: `text-embedding-ada-002`
- Persistent storage in `vector_store/` directory
- Separate collections for each support type

### RAG Chain

- LLM: GPT-4o
- Temperature: 0 (deterministic responses)
- Configurable number of retrieved documents

## 📊 Data Format

### JSON Format
```json
[
  {
    "Ticket ID": "12345",
    "subject": "Password reset issue",
    "body": "User cannot reset password",
    "answer": "Follow the password reset link...",
    "type": "incident",
    "queue": "support",
    "priority": "high",
    "language": "en",
    "tag_1": "password",
    "tag_2": "authentication"
  }
]
```

### XML Format
```xml
<Tickets>
  <Ticket>
    <TicketID>12345</TicketID>
    <subject>Password reset issue</subject>
    <body>User cannot reset password</body>
    <answer>Follow the password reset link...</answer>
    <type>incident</type>
    <queue>support</queue>
    <priority>high</priority>
    <language>en</language>
    <tag_1>password</tag_1>
    <tag_2>authentication</tag_2>
  </Ticket>
</Tickets>
```

## 🎨 Streamlit UI Features

### Main Interface
- **Search Bar**: Enter natural language queries
- **Support Type Filter**: Filter by technical, product, or customer support
- **Results Slider**: Control number of retrieved tickets
- **Query History**: Track recent searches

### Ticket Display
- Formatted ticket cards with metadata
- Expandable content sections
- Tag visualization
- Priority and type indicators

### AI Response
- Contextual answers based on retrieved tickets
- Formatted response display
- Source ticket references

## 🛠️ Development

### Project Structure

- **document_loader.py**: Handles loading and parsing of JSON/XML ticket files
- **vector_store.py**: Manages ChromaDB vector store operations
- **rag_chain.py**: Implements RAG chain with LangChain and OpenAI
- **engine/__init__.py**: Orchestrates initialization and component integration
- **app.py**: Streamlit UI implementation
- **main.py**: CLI interface and entry point

### Coding Style

The project follows these conventions:
- Type hints for all function parameters and returns
- Comprehensive error handling with logging
- Docstrings for all classes and methods
- Consistent naming conventions
- Modular, reusable components

## 📝 Logging

The system uses Python's logging module:
- INFO level for normal operations
- ERROR level for exceptions
- Detailed error messages with stack traces

## 🔒 Security

- API keys stored in `.env` file (not committed to git)
- Input validation for queries
- Sanitized metadata handling
- No sensitive data in logs

## 🤝 Contributing

1. Follow the existing coding style
2. Add type hints to all functions
3. Include docstrings for new classes/methods
4. Test both UI and CLI interfaces
5. Update README for new features

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- LangChain for RAG framework
- ChromaDB for vector storage
- OpenAI for embeddings and LLM
- Streamlit for UI framework
