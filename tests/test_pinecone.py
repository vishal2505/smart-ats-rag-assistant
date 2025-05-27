#!/usr/bin/env python3
"""
Test Pinecone configuration and connection
"""
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

def test_pinecone_connection():
    """Test Pinecone connection and configuration"""
    print("Testing Pinecone Configuration...")
    print("=" * 50)
    
    # Check environment variables
    api_key = os.getenv("PINECONE_API_KEY")
    cloud = os.getenv("PINECONE_CLOUD")
    region = os.getenv("PINECONE_REGION")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    print(f"PINECONE_API_KEY: {'✓ Set' if api_key else '✗ Missing'}")
    print(f"PINECONE_CLOUD: {cloud}")
    print(f"PINECONE_REGION: {region}")
    print(f"PINECONE_INDEX_NAME: {index_name}")
    print()
    
    if not api_key:
        print("❌ PINECONE_API_KEY is not set!")
        return False
    
    try:
        from pinecone import Pinecone, ServerlessSpec
        
        # Initialize Pinecone
        print("Initializing Pinecone client...")
        pc = Pinecone(api_key=api_key)
        print("✓ Pinecone client initialized successfully")
        
        # List existing indexes
        print("\nListing existing indexes...")
        existing_indexes = [index.name for index in pc.list_indexes()]
        print(f"Existing indexes: {existing_indexes}")
        
        # Test ServerlessSpec configuration
        print(f"\nTesting ServerlessSpec with cloud='{cloud}', region='{region}'...")
        spec = ServerlessSpec(cloud=cloud, region=region)
        print("✓ ServerlessSpec created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Pinecone: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_embeddings():
    """Test embeddings functionality"""
    print("\nTesting Embeddings...")
    print("=" * 50)
    
    try:
        from rag.embeddings import get_embedding_function
        
        print("Getting embedding function...")
        embedding_func = get_embedding_function()
        print("✓ Embedding function loaded successfully")
        
        # Test embedding a simple text
        test_text = "This is a test document for embeddings"
        print(f"Testing embedding for: '{test_text}'")
        
        embedding = embedding_func.embed_query(test_text)
        print(f"✓ Embedding generated successfully (dimension: {len(embedding)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Starting Pinecone and Embeddings Test")
    print("=" * 60)
    
    success = True
    
    # Test Pinecone connection
    if not test_pinecone_connection():
        success = False
    
    # Test embeddings
    if not test_embeddings():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Configuration is working correctly.")
    else:
        print("❌ Some tests failed. Please check the configuration.")
    
    print("=" * 60)
