#!/usr/bin/env python3
"""
Component validation for Smart ATS RAG system
Tests individual components to ensure they work correctly
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_environment():
    """Validate environment setup"""
    print("🔍 Validating Environment Setup")
    print("-" * 35)
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API access",
        "PINECONE_API_KEY": "Pinecone vector database",
        "GROQ_API_KEY": "Groq LLM provider (optional)"
    }
    
    all_good = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            masked_value = value[:8] + "..." if len(value) > 8 else "***"
            print(f"✅ {var}: {masked_value} ({description})")
        else:
            if var == "GROQ_API_KEY":
                print(f"⚠️  {var}: Not set ({description})")
            else:
                print(f"❌ {var}: Missing ({description})")
                all_good = False
    
    return all_good

def validate_data_files():
    """Validate required data files"""
    print("\n📁 Validating Data Files")
    print("-" * 25)
    
    # Resolve absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_dir, "..", "data", "job_title_des.csv")
    data_file_path = os.path.abspath(data_file_path)
    download_script_path = os.path.join(script_dir, "download_dataset.py")
    download_script_path = os.path.abspath(download_script_path)
    
    files_to_check = [
        (data_file_path, "Job descriptions dataset"),
        (download_script_path, "Dataset download script")
    ]
    
    all_good = True
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path}: {size:,} bytes ({description})")
        else:
            print(f"❌ {file_path}: Missing ({description})")
            all_good = False
    
    return all_good

def validate_rag_components():
    """Validate RAG system components"""
    print("\n🧩 Validating RAG Components")
    print("-" * 30)
    
    components = [
        ("rag.embeddings", "Embedding functions"),
        ("rag.vector_store", "Vector store operations"),
        ("rag.llm_service", "LLM service providers"),
        ("rag.document_processor", "Document processing"),
        ("rag.retriever", "Retrieval strategies"),
        ("rag.rag_qa_chain", "QA chain creation")
    ]
    
    all_good = True
    for module_name, description in components:
        try:
            __import__(module_name)
            print(f"✅ {module_name}: Available ({description})")
        except ImportError as e:
            print(f"❌ {module_name}: Import failed - {str(e)}")
            all_good = False
        except Exception as e:
            print(f"⚠️  {module_name}: Available but has issues - {str(e)}")
    
    return all_good

def validate_streamlit_pages():
    """Validate Streamlit application pages"""
    print("\n📄 Validating Streamlit Pages")
    print("-" * 30)
    
    pages = [
        ("app.py", "Main application"),
        ("pages/faq_assistant.py", "FAQ Assistant page")
    ]
    
    all_good = True
    for page_path, description in pages:
        if os.path.exists(page_path):
            print(f"✅ {page_path}: Available ({description})")
        else:
            print(f"❌ {page_path}: Missing ({description})")
            all_good = False
    
    return all_good

def generate_setup_instructions():
    """Generate setup instructions based on validation results"""
    print("\n📋 Setup Instructions")
    print("-" * 20)
    
    print("1. Environment Variables:")
    print("   Copy .env.example to .env and add your API keys")
    
    print("\n2. Download Dataset:")
    print("   cd data && python download_dataset.py")
    
    print("\n3. Install Dependencies:")
    print("   pipenv install")
    
    print("\n4. Initialize Vector Database:")
    print("   python setup.py")
    
    print("\n5. Run Application:")
    print("   streamlit run app.py")

def main():
    """Main validation function"""
    print("🔬 Smart ATS RAG System Validation")
    print("=" * 40)
    
    validations = [
        validate_environment(),
        validate_data_files(),
        validate_rag_components(),
        validate_streamlit_pages()
    ]
    
    all_passed = all(validations)
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All validations passed!")
        print("✨ Your RAG system is ready to use!")
        print("\n🚀 Start the application:")
        print("   streamlit run app.py")
    else:
        print("❌ Some validations failed.")
        generate_setup_instructions()

if __name__ == "__main__":
    main()
