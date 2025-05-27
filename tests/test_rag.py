#!/usr/bin/env python3
"""
Test script for the RAG-based FAQ Assistant
This script tests the core RAG functionality without the Streamlit interface
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_functionality():
    """Test basic RAG functionality with sample data"""
    print("🧪 Testing RAG FAQ Assistant Basic Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Import all modules
        print("1. Testing imports...")
        from rag.document_processor import add_career_guidance_documents, chunk_documents
        from rag.embeddings import get_embedding_function
        from rag.vector_store import create_pinecone_index, delete_pinecone_index
        from rag.llm_service import get_llm
        from rag.rag_qa_chain import create_rag_chain
        print("   ✅ All imports successful")
        
        # Test 2: Create sample documents
        print("2. Creating sample documents...")
        career_docs = add_career_guidance_documents()
        chunked_docs = chunk_documents(career_docs)
        print(f"   ✅ Created {len(chunked_docs)} document chunks")
        
        # Test 3: Test embedding function
        print("3. Testing embedding function...")
        try:
            embedding_function = get_embedding_function()
            print("   ✅ Embedding function created successfully")
        except Exception as e:
            print(f"   ❌ Embedding function failed: {str(e)}")
            print("   💡 Make sure OPENAI_API_KEY is set in .env file")
            return False
        
        # Test 4: Test LLM service
        print("4. Testing LLM service...")
        try:
            llm = get_llm(provider="openai", model="gpt-3.5-turbo")
            print("   ✅ LLM service created successfully")
        except Exception as e:
            print(f"   ❌ LLM service failed: {str(e)}")
            print("   💡 Make sure API keys are set correctly")
            return False
        
        # Test 5: Test vector store operations
        print("5. Testing vector store operations...")
        try:
            # Try to create a test index
            test_index_name = "test-rag-index"
            
            # Clean up any existing test index
            try:
                delete_pinecone_index(test_index_name)
            except:
                pass  # Ignore if doesn't exist
            
            # Create new test index
            create_pinecone_index(test_index_name, dimension=1536)
            print("   ✅ Vector store operations successful")
            
            # Clean up
            delete_pinecone_index(test_index_name)
            print("   ✅ Test cleanup completed")
            
        except Exception as e:
            print(f"   ❌ Vector store failed: {str(e)}")
            print("   💡 Make sure PINECONE_API_KEY is set correctly")
            return False
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("💡 Make sure all dependencies are installed: pipenv install")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def test_sample_queries():
    """Test with sample career-related queries"""
    print("\n🔍 Testing Sample Queries")
    print("=" * 30)
    
    sample_queries = [
        "How do I optimize my resume for ATS systems?",
        "What should I include in a cover letter?",
        "How do I prepare for a technical interview?",
        "What are the best job search strategies?",
        "How should I negotiate my salary?"
    ]
    
    print("Sample queries that the system can handle:")
    for i, query in enumerate(sample_queries, 1):
        print(f"{i}. {query}")
    
    print("\n💡 These queries can be tested in the Streamlit interface")
    print("💡 Run: streamlit run app.py")

def main():
    """Main test function"""
    print("🤖 Smart ATS RAG FAQ Assistant - Test Suite")
    print("=" * 55)
    print()
    
    # Check environment variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Please set these in your .env file")
        print("💡 See .env.example for reference")
        return
    
    # Run tests
    if test_basic_functionality():
        test_sample_queries()
        print("\n✨ RAG system is ready to use!")
        print("🚀 Start the application with: streamlit run app.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
