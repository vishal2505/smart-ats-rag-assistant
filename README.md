# Smart ATS Resume Analyzer with RAG-based FAQ Assistant

A comprehensive AI-powered career guidance platform that combines ATS resume analysis, psychometric testing, resume ranking, and an intelligent FAQ assistant powered by Retrieval-Augmented Generation (RAG) technology.

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

### MCQ Generator - Psychometric Test
- **AI-Powered Question Generation**: Create customized multiple-choice questions for job roles
- **Role-Specific Assessment**: Generate questions tailored to specific job positions and skills
- **Configurable Difficulty**: Adjust question complexity and focus areas
- **Automated Scoring**: Real-time evaluation with detailed explanations
- **Export Functionality**: Download test results and analysis reports
- **Multiple Choice Formats**: Support for various question types and formats

### Resume Ranking System
- **Multi-Algorithm Ranking**: Compare resumes using TF-IDF, Sentence Transformers, and LLM-based scoring
- **Bulk Resume Processing**: Upload and analyze multiple resumes simultaneously
- **Semantic Similarity**: Advanced embedding-based resume-job description matching
- **AI-Powered Evaluation**: LLM-driven scoring with detailed explanations
- **Comparative Analysis**: Side-by-side resume comparison and ranking
- **Export Results**: Download ranking reports for hiring decisions

### RAG-powered FAQ Assistant
- **Intelligent Career Guidance**: Get expert advice on resumes, interviews, and job searching
- **Job Market Insights**: Access information from 60,000+ job descriptions
- **Multi-modal AI Support**: Choose between OpenAI GPT and Groq models
- **Contextual Responses**: Smart retrieval system adapts to different query types
- **Conversation Memory**: Maintains context across multi-turn conversations
- **Source References**: See exactly where information comes from
- **Rigorous Evaluation**: Comprehensive testing with RAGAS and DeepEval frameworks ensuring production-ready quality

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Multi-page application)
- **Backend**: Python with Langchain framework
- **Vector Database**: Pinecone for semantic search
- **LLM Providers**: OpenAI GPT, Groq (Llama, Mixtral, DeepSeek)
- **Embeddings**: OpenAI text-embedding-ada-002, Sentence Transformers
- **ML Libraries**: Scikit-learn (TF-IDF, Cosine Similarity), Sentence Transformers
- **Document Processing**: PyPDF2, Advanced chunking and metadata extraction
- **Data Source**: Kaggle job descriptions dataset (60,000+ entries)
- **Testing Framework**: Custom psychometric assessment engine
- **Evaluation Frameworks**: RAGAS (Multi-turn conversation evaluation), DeepEval (Single-turn Q&A evaluation)
- **Quality Assurance**: Comprehensive model benchmarking and performance monitoring

## 📁 Project Structure

```
smart-ats-rag-assistant/
├── app.py                    # Main Streamlit application
├── mcq_utils.py             # MCQ generation utilities
├── pages/
│   ├── faq_assistant.py     # RAG-powered FAQ assistant
│   ├── mcq_app.py           # Psychometric test generator
│   └── resume_ranking_app.py # Resume ranking system
├── rag/                     # RAG system components
│   ├── embeddings.py        # Embedding functions
│   ├── vector_store.py      # Pinecone operations
│   ├── llm_service.py       # LLM providers
│   ├── document_processor.py # Text processing
│   ├── retriever.py         # Retrieval strategies
│   └── rag_qa_chain.py      # QA chain creation
├── evaluation/              # Model evaluation framework
│   ├── deepeval_evaluation_runner.py # DeepEval single-turn evaluation
│   ├── ragas_evaluation_runner.py    # RAGAS multi-turn evaluation wrapper
│   ├── ragas_evaluator.py            # RAGAS core evaluation logic
│   ├── ragas_results_processor.py    # Results analysis & visualization
│   ├── README.md                     # Detailed evaluation methodology
│   └── results/                      # Evaluation results & reports
│       ├── RAGAS_Evaluation_Report.md     # RAGAS detailed results
│       ├── DeepEval_Evaluation_Report.md  # DeepEval detailed results
│       ├── Final_Evaluation_Summary.md    # Combined analysis & recommendations
│       ├── comprehensive_evaluation_summary.json
│       ├── deepeval/                      # DeepEval raw results
│       └── ragas/                         # RAGAS raw results & CSVs
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
The project includes all necessary dependencies for:
- Resume analysis and ranking
- MCQ generation and testing
- RAG-powered FAQ system
- Machine learning algorithms

```bash
pipenv install
pipenv shell
```

Key dependencies include:
- `streamlit` - Web application framework
- `langchain` - LLM application framework
- `pinecone-client` - Vector database
- `openai` - OpenAI API integration
- `groq` - Groq API integration
- `scikit-learn` - Machine learning algorithms
- `sentence-transformers` - Semantic embeddings
- `PyPDF2` - PDF processing
- `pandas` - Data manipulation

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

### MCQ Generator - Psychometric Test (For Recruiters)
1. Navigate to the "MCQ Generator" page
2. Select your preferred AI model from the sidebar
3. Enter the job title and job description
4. Configure test parameters:
   - Number of questions (5-20)
   - Question types (Technical, Behavioral, Situational)
   - Difficulty level (Easy, Medium, Hard)
5. Click "Generate MCQ Test" to create customized questions
6. Review questions and modify if needed
7. Take the test or export for candidate assessment
8. View detailed results with explanations and scoring

### Resume Ranking System (For Hiring Companies)
1. Navigate to the "Resume Ranking" page
2. Enter the job description in the text area
3. Upload multiple resume files (PDF format)
4. Choose ranking method from three tabs:
   - **TF-IDF**: Traditional keyword-based ranking
   - **Sentence Transformers**: Semantic similarity ranking
   - **LLM-Based**: AI-powered evaluation with explanations
5. View ranked results with scores and detailed analysis
6. Export rankings for hiring decisions

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
- **Groq**: Llama-3.1, Mixtral-8x7B, DeepSeek R1 (high-speed inference)

### Intelligent Retrieval
- **Query Classification**: Automatically detect question types
- **Adaptive Search**: Adjust retrieval strategy per query
- **Source Diversity**: Balance relevance with information variety

### Advanced Ranking Algorithms
- **TF-IDF Vectorization**: Traditional keyword-based similarity
- **Sentence Transformers**: Semantic embedding-based matching
- **LLM-Powered Evaluation**: AI-driven scoring with explanations
- **Multi-Algorithm Comparison**: Side-by-side ranking results

### Psychometric Testing Engine
- **Dynamic Question Generation**: AI-powered MCQ creation
- **Adaptive Difficulty**: Intelligent question complexity adjustment
- **Multi-Domain Assessment**: Technical, behavioral, and situational testing
- **Real-time Scoring**: Instant evaluation with detailed feedback

### User Experience
- **Responsive Design**: Works on desktop and mobile
- **Real-time Processing**: Stream responses as they generate
- **Export Options**: Download generated content and test results
- **Session Persistence**: Remember conversation history
- **Bulk Processing**: Handle multiple resumes simultaneously

## 🧪 **Model Evaluation Framework**

We've implemented a comprehensive evaluation system using industry-standard frameworks to ensure the highest quality AI responses.

### **Evaluation Overview**
- **Frameworks**: RAGAS (Multi-turn) + DeepEval (Single-turn)
- **Models Tested**: 4 leading AI models
- **Total Evaluations**: 40+ test cases across different scenarios
- **Status**: ✅ **COMPLETE** - All models evaluated and benchmarked

### **Models Evaluated**
| Model | Provider | RAGAS Score | DeepEval Score | Status |
|-------|----------|-------------|----------------|--------|
| 🏆 **GPT-3.5-Turbo** | OpenAI | **0.498** | **1.000** | ✅ Recommended |
| 🥈 **Llama-3.1-8B-Instant** | Groq | **0.418** | **1.000** | ✅ Speed Option |
| 🥉 **GPT-4o-Mini** | OpenAI | **0.386** | **1.000** | ✅ Accuracy Option |
| **Llama3-8B-8192** | Groq | **0.383** | **0.946** | ✅ Baseline |

### **Evaluation Methodologies**

#### **RAGAS Multi-Turn Evaluation**
- **Purpose**: Evaluate conversational AI capabilities
- **Scenarios**: 5 conversation types (Resume Optimization, Interview Prep, Career Transition, Salary Negotiation, Skills Development)
- **Metrics**: 
  - Faithfulness (0.866 best - GPT-4o-Mini)
  - Answer Relevancy (0.941 best - Llama-3.1-8B)
  - Context Precision & Recall
  - Answer Correctness & Similarity
- **Method**: Multi-turn conversations with job-specific contexts

#### **DeepEval Single-Turn Evaluation**
- **Purpose**: Evaluate single Q&A accuracy
- **Scenarios**: Job-specific skill questions
- **Metrics**:
  - Answer Relevancy (threshold: 0.5)
  - Faithfulness (threshold: 0.5)
- **Method**: Dataset-driven test cases with ground truth validation

### **Key Findings**
1. **Production Ready**: All models achieve excellent single-turn performance (≥94.6%)
2. **Conversational Excellence**: OpenAI GPT-3.5-Turbo leads in multi-turn scenarios
3. **Speed vs. Quality**: Groq models offer faster inference with competitive accuracy
4. **Context Handling**: Critical area for improvement across all models

### **Business Impact**
- **Reliability**: Rigorous testing ensures consistent user experience
- **Transparency**: Complete evaluation methodology available
- **Optimization**: Data-driven model selection for different use cases
- **Quality Assurance**: Continuous monitoring and improvement framework

### **Detailed Reports**
Complete evaluation documentation available in `/evaluation/`:
- **[RAGAS Evaluation Report](./evaluation/results/RAGAS_Evaluation_Report.md)** - Multi-turn conversation analysis
- **[DeepEval Evaluation Report](./evaluation/results/DeepEval_Evaluation_Report.md)** - Single-turn Q&A analysis  
- **[Final Evaluation Summary](./evaluation/results/Final_Evaluation_Summary.md)** - Combined analysis & recommendations

### **Evaluation Commands**
```bash
# Run complete evaluation suite
cd evaluation
python ragas_evaluation_runner.py      # ~30-45 minutes
python deepeval_evaluation_runner.py   # ~10-15 minutes
python ragas_results_processor.py      # ~2-3 minutes

# View comprehensive results
open results/Final_Evaluation_Summary.md
```

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