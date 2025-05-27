# Scripts Directory

This directory contains utility and setup scripts for the Smart ATS RAG FAQ Assistant.

## Files

- `index.py` - Interactive index showing all available setup scripts
- `complete_setup.py` - Complete automated setup script
- `setup.py` - Original setup script for vector database initialization  
- `validate.py` - System validation and health check script
- `init_vector_db.py` - Vector database initialization script
- `download_dataset.py` - Kaggle dataset download script

## Usage

### Interactive Script Index (NEW)
```bash
python scripts/index.py
```
This will show you all available scripts and their purposes.

### Complete Setup (Recommended)
```bash
python scripts/complete_setup.py
```
This script will:
1. ✅ Check environment variables
2. ✅ Verify dependencies
3. ✅ Download dataset (if needed)
4. ✅ Initialize vector database
5. ✅ Run system tests

### Individual Scripts
```bash
# Initialize vector database only
python scripts/setup.py

# Validate system health
python scripts/validate.py

# Download dataset from Kaggle
python scripts/download_dataset.py

# Initialize vector database with job descriptions
python scripts/init_vector_db.py
```

## Setup Process

The complete setup process includes:

1. **Environment Check** - Validates all required API keys
2. **Dependency Check** - Ensures all packages are installed
3. **Dataset Download** - Downloads Kaggle job descriptions dataset
4. **Vector Database** - Creates and populates Pinecone index
5. **System Tests** - Runs comprehensive tests to verify functionality

## Prerequisites

1. Python 3.8+ installed
2. Pipenv for dependency management
3. Valid API keys for:
   - OpenAI (embeddings and LLM)
   - Groq (alternative LLM)
   - Pinecone (vector database)
   - Kaggle (dataset access)

## Environment Variables

Ensure your `.env` file contains:
```bash
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_INDEX_NAME=smart-ats-faq
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```
