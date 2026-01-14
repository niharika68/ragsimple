# Fashion RAG System

A simple Retrieval-Augmented Generation (RAG) system for answering fashion-related questions. This system uses OpenAI embeddings and language models along with ChromaDB for vector storage to provide intelligent answers to fashion queries.

## Features

- 🎨 **Fashion Knowledge Base**: Pre-loaded with comprehensive fashion information covering:
  - Clothing types (dresses, tops, bottoms)
  - Color coordination and seasonal palettes
  - Fabrics and materials
  - Styling tips for different body types
  - Occasion-based fashion advice
  - Current fashion trends
  - Clothing care and maintenance

- 🔍 **Semantic Search**: Uses vector embeddings to find relevant information
- 💬 **Interactive Mode**: Chat-like interface for multiple questions
- 🎯 **Single Question Mode**: Quick answers for one-off queries
- 📚 **Source Citations**: Optional display of source documents

## Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/niharika68/ragsimple.git
cd ragsimple
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your-api-key-here
```

Or export it as an environment variable:
```bash
export OPENAI_API_KEY=your-api-key-here
```

## Usage

### Demo Mode (No API Key Required)

To see the system structure and capabilities without an API key:

```bash
python demo.py
```

This will show you:
- Knowledge base statistics
- Topics covered
- Document processing details
- Example queries you can ask

### Interactive Mode (Default)

Run the script without arguments to enter interactive mode:

```bash
python main.py
```

You can then ask multiple questions:
```
Your question: What should I wear for a business casual meeting?
Your question: How do I style an A-line dress?
Your question: What colors go well together?
```

Type `quit` or `exit` to stop.

### Single Question Mode

Ask a single question and get an immediate answer:

```bash
python main.py --question "What are the best fabrics for summer?"
```

### Show Source Documents

Include source documents with your answer:

```bash
python main.py --question "How do I care for silk clothing?" --sources
```

### Rebuild Knowledge Base

Force rebuild the vector database:

```bash
python main.py --rebuild
```

### Command-line Options

```
-q, --question    Ask a single question and exit
-s, --sources     Show source documents with answers
-r, --rebuild     Rebuild the knowledge base from scratch
-k, --api-key     OpenAI API key (alternative to .env file)
-h, --help        Show help message
```

## Example Questions

Here are some example questions you can ask:

- "What should I wear to a formal event?"
- "How do I dress for my body type?"
- "What colors are best for fall?"
- "What's the difference between a blouse and a t-shirt?"
- "How do I care for wool clothing?"
- "What are current fashion trends?"
- "How should I accessorize a simple outfit?"
- "What fabrics are best for winter?"

## Project Structure

```
ragsimple/
├── main.py              # Main application script
├── fashion_rag.py       # RAG system implementation
├── fashion_data.py      # Fashion knowledge base
├── demo.py              # Demo script (no API key needed)
├── test_fashion_rag.py  # Unit tests
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment file
├── .env                 # OpenAI API key (create this)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## How It Works

1. **Document Processing**: Fashion documents are split into smaller chunks for better retrieval
2. **Embedding Creation**: OpenAI embeddings convert text chunks into vectors
3. **Vector Storage**: ChromaDB stores the vectors for efficient similarity search
4. **Query Processing**: User questions are embedded and matched against the knowledge base
5. **Answer Generation**: Retrieved context is sent to GPT-3.5-turbo to generate natural answers

## Customization

### Adding Your Own Fashion Documents

Edit `fashion_data.py` and add your documents to the `FASHION_DOCUMENTS` list:

```python
FASHION_DOCUMENTS = [
    "Your fashion knowledge here...",
    # Add more documents
]
```

Then rebuild the knowledge base:
```bash
python main.py --rebuild
```

### Changing the LLM Model

You can modify the model in `fashion_rag.py`:

```python
# In create_qa_chain method
llm = ChatOpenAI(
    temperature=0.0,
    model_name="gpt-4",  # Change to gpt-4 or other models
    openai_api_key=self.api_key
)
```

## Limitations

- Requires an active internet connection for OpenAI API calls
- Knowledge is limited to the pre-loaded fashion documents
- API calls incur costs based on OpenAI's pricing

## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the fashion knowledge base or add new features.