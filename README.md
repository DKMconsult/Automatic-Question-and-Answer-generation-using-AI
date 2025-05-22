# Automatic-Question-and-Answer-generation-using-AI(This is in the master branch with a small change in this README version of it)
This project implements a robust pipeline to process NCERT textbook PDFs across classes and subjects, convert them into semantically meaningful vector embeddings using OpenAI's text-embedding-ada-002, and store them in a Qdrant vector database for efficient retrieval. Built with the LangChain framework, this system is designed to power intelligent question-answering, tutoring, or curriculum automation tools.

Key Features
1. Dynamic Question Generation for the Same PDF
Unlike static question systems, this implementation ensures that each query generates a different set of questions from the same input PDF, enabling:

Richer learning outcomes through non-repetitive assessments

Improved student engagement

A generative approach to knowledge understanding

This is achieved through a combination of:

Contextually chunked embeddings

Probabilistic or randomized prompt structuring

Non-deterministic LLM behavior in downstream tasks

2. Token-Aware PDF Processing with LangChain
To tackle OpenAI’s API token limitations (especially for large PDFs), this project integrates token-level chunking using LangChain’s TokenTextSplitter. This ensures:

Each chunk remains within the token limit (e.g., 200 tokens per chunk)

No data is lost due to token overflows

More efficient and scalable embedding generation

The system dynamically splits large documents into overlapping token-based segments, preserving semantic continuity and enabling effective vector search.

Tech Stack
Python 3.10+

LangChain

OpenAI Embeddings

Qdrant Vector DB

tiktoken for token counting

backoff for resilient API retries

Functional Overview
Folder Traversal & Unzipping: Traverses a class-wise directory of zipped subject PDFs and extracts them.

Filtering: Only relevant subjects like Science, Maths, History, etc., are processed.

PDF Loading: Uses PyPDFLoader to load textual content.

Tokenized Chunking: Documents are split into manageable chunks (200 tokens with 20-token overlap).

Embedding: Text chunks are embedded using OpenAI’s embedding API.

Vector Storage: Embedded vectors are stored in Qdrant for fast semantic search.

Future Enhancements
Integrated a Streamlit based-frontend to generate and display dynamic questions

Enable multilingual PDF handling

Extend support to more educational content (worksheets, reference books, etc.)



