# Tests Directory

This directory contains all test scripts for the Smart ATS RAG FAQ Assistant.

## Files

- `test_pinecone.py` - Tests Pinecone configuration and connection
- `test_rag.py` - Comprehensive RAG system functionality tests
- `run_tests.py` - Test runner script that executes all tests

## Usage

### Run All Tests
```bash
python tests/run_tests.py
```

### Run Individual Tests
```bash
# Test Pinecone configuration
python tests/test_pinecone.py

# Test RAG functionality
python tests/test_rag.py
```

## Test Coverage

- ✅ Environment variable validation
- ✅ API key verification
- ✅ Pinecone connection and configuration
- ✅ Vector store operations
- ✅ Embedding generation
- ✅ LLM service integration
- ✅ RAG chain functionality
- ✅ End-to-end query processing

## Prerequisites

Before running tests, ensure:
1. All dependencies are installed (`pipenv install`)
2. Environment variables are set in `.env` file
3. Vector database is initialized (if testing RAG functionality)
