# Smart ATS Resume Analyzer with RAG-based FAQ Assistant

A comprehensive AI-powered career guidance platform that combines ATS resume analysis with an intelligent FAQ assistant powered by Retrieval-Augmented Generation (RAG) technology.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![RAG](https://img.shields.io/badge/RAG-Powered-orange.svg)
![AI](https://img.shields.io/badge/AI-Multi--Modal-purple.svg)

## 🚀 Features

### Resume Analyzer
- **ATS-Optimized Analysis**: Evaluate resume compatibility with job descriptions
- **Keyword Matching**: Identify missing keywords and optimization opportunities
- **Personalized Feedback**: Get specific recommendations for improvement
- **Cover Letter Generation**: Create tailored cover letters for job applications
- **Resume Updates**: Generate improved versions of your resume

### RAG-powered FAQ Assistant
- **Intelligent Career Guidance**: Get expert advice on resumes, interviews, and job searching
- **Job Market Insights**: Access information from 60,000+ job descriptions
- **Multi-modal AI Support**: Choose between OpenAI GPT and Groq models
- **Contextual Responses**: Smart retrieval system adapts to different query types
- **Conversation Memory**: Maintains context across multi-turn conversations
- **Source References**: See exactly where information comes from

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Multi-page application)
- **Backend**: Python with Langchain framework
- **Vector Database**: Pinecone for semantic search
- **LLM Providers**: OpenAI GPT, Groq (Llama, Mixtral)
- **Embeddings**: OpenAI text-embedding-ada-002
- **Data Source**: Kaggle job descriptions dataset (60,000+ entries)
- **Document Processing**: Advanced chunking and metadata extraction

## 📁 Project Structure

```
smart-ats-rag-assistant/
├── app.py                    # Main Streamlit application
├── pages/
│   └── faq_assistant.py     # RAG-powered FAQ assistant
├── rag/                     # RAG system components
│   ├── embeddings.py        # Embedding functions
│   ├── vector_store.py      # Pinecone operations
│   ├── llm_service.py       # LLM providers
│   ├── document_processor.py # Text processing
│   ├── retriever.py         # Retrieval strategies
│   └── rag_qa_chain.py      # QA chain creation
├── scripts/                 # Setup and utility scripts
│   ├── complete_setup.py    # Automated setup
│   ├── download_dataset.py  # Dataset download
│   ├── init_vector_db.py    # Vector DB initialization
│   ├── validate.py          # System validation
│   └── index.py             # Scripts overview
├── data/                    # Datasets and documents
│   ├── job_title_des.csv    # Kaggle job descriptions
│   └── resume/              # Sample resume files
├── tests/                   # Test suite
│   ├── run_tests.py         # Test runner
│   ├── test_rag.py          # RAG system tests
│   └── test_pinecone.py     # Vector DB tests
├── utils.py                 # Utility functions
├── Pipfile                  # Dependencies
├── .env.template            # Environment template
└── README.md               # Documentation
```

## 📋 Prerequisites

- Python 3.8+
- Pipenv for dependency management
- API keys for:
  - OpenAI (for embeddings and LLM)
  - Groq (alternative LLM provider)
  - Pinecone (vector database)
  - Kaggle (for dataset access)

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Smart_ATS_With_RAG
```

### 2. Install Dependencies
```bash
pipenv install
pipenv shell
```

### 3. Environment Configuration
Copy the example environment file and add your API keys:
```bash
cp .env.example .env
```

Edit `.env` with your actual API keys:
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Groq Configuration  
GROQ_API_KEY=your_groq_api_key

# Pinecone Vector Database Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=smart-ats-faq

# Kaggle Configuration for Dataset Download
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

### 4. Download Dataset
```bash
cd scripts
python download_dataset.py
```

### 5. View All Setup Scripts
```bash
python scripts/index.py
```

This will show you all available setup scripts and their purposes.

### 6. Complete Setup (Automated)
```bash
python scripts/complete_setup.py
```

This automated setup script will:
- ✅ Verify environment variables and dependencies
- ✅ Download dataset if needed
- ✅ Initialize vector database with job descriptions
- ✅ Run comprehensive tests to verify functionality

### 7. Manual Setup (Alternative)
If you prefer manual setup:
```bash
# Initialize vector database only
python scripts/setup.py

# Validate system
python scripts/validate.py

# Run tests
python tests/run_tests.py
```

### 7. Run the Application
```bash
streamlit run app.py
```

## 🎯 Usage Guide

### Resume Analyzer
1. Navigate to the main page (Resume Analyzer)
2. Select your preferred AI model from the sidebar
3. Paste a job description in the text area
4. Upload your resume (PDF format)
5. Click "Analyze Resume" to get detailed feedback
6. Use additional features like cover letter generation

### FAQ Assistant
1. Navigate to the "FAQ Assistant" page
2. Configure your AI model and retrieval settings in the sidebar
3. Ask questions about:
   - Resume optimization and ATS systems
   - Interview preparation and techniques
   - Salary negotiation strategies
   - Job search best practices
   - Career development advice
4. Use suggested questions for quick starts
5. Enable conversation memory for context-aware responses

## 🔍 RAG System Architecture

### Document Processing Pipeline
1. **Data Ingestion**: Load 60,000+ job descriptions from Kaggle
2. **Content Enhancement**: Add career guidance documents
3. **Text Chunking**: Split documents using recursive character splitting
4. **Metadata Extraction**: Capture job titles, types, and sources
5. **Vector Embedding**: Generate semantic embeddings using OpenAI

### Retrieval Strategies
- **Similarity Search**: Standard cosine similarity
- **Maximum Marginal Relevance (MMR)**: Diverse, relevant results
- **Multi-Query Retrieval**: Generate multiple query perspectives
- **Contextual Retrieval**: Adapt strategy based on query type

### Response Generation
- **Prompt Engineering**: Expert career advisor persona
- **Source Integration**: Combine multiple relevant documents
- **Citation Support**: Track and display information sources
- **Conversation Context**: Maintain dialogue history

## 📊 Dataset Information

- **Source**: Kaggle Jobs and Job Description dataset
- **Size**: 60,000+ job descriptions
- **Coverage**: Multiple industries and job roles
- **Enhancement**: Added expert career guidance content
- **Processing**: Chunked and vectorized for optimal retrieval

## 🚀 Advanced Features

### Multi-Provider LLM Support
- **OpenAI**: GPT-3.5-turbo, GPT-4, GPT-4o
- **Groq**: Llama-3.1, Mixtral-8x7B (high-speed inference)

### Intelligent Retrieval
- **Query Classification**: Automatically detect question types
- **Adaptive Search**: Adjust retrieval strategy per query
- **Source Diversity**: Balance relevance with information variety

### User Experience
- **Responsive Design**: Works on desktop and mobile
- **Real-time Processing**: Stream responses as they generate
- **Export Options**: Download generated content
- **Session Persistence**: Remember conversation history

## 🛡️ Error Handling & Monitoring

- **API Rate Limiting**: Automatic retry with exponential backoff
- **Fallback Mechanisms**: Switch between providers if needed
- **Input Validation**: Comprehensive error checking
- **Usage Analytics**: Track system performance and usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Quick Diagnostics
```bash
# Run all tests to identify issues
python tests/run_tests.py

# Validate system health
python scripts/validate.py

# Complete setup if needed
python scripts/complete_setup.py
```

### Common Issues

**Vector Database Initialization Fails**
- Check Pinecone API credentials
- Verify index name doesn't already exist
- Ensure sufficient Pinecone quota

**LLM API Errors**
- Verify API keys are correct
- Check rate limits and quotas
- Try alternative provider (OpenAI ↔ Groq)

**Dataset Download Issues**
- Confirm Kaggle credentials
- Check internet connectivity
- Verify dataset availability

**Console Warnings**
- Most LangChain deprecation warnings are suppressed
- "Index already exists" messages are informational only
- Restart Streamlit app if needed

**Streamlit Performance**
- Reduce retrieval `k` value for faster responses
- Use Groq for faster inference
- Clear conversation history periodically

### Support
For issues and questions, please create an issue in the repository or contact the development team.

---