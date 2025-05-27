#!/usr/bin/env python3
"""
Test runner for Smart ATS RAG FAQ Assistant
Run all tests to verify system functionality
"""

import os
import sys
import subprocess
from pathlib import Path

def run_test(test_script, description):
    """Run a test script and return the result"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, test_script
        ], capture_output=True, text=True, check=False)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n{status}: {description}")
        return success
        
    except Exception as e:
        print(f"❌ ERROR running {test_script}: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Smart ATS RAG FAQ Assistant - Test Suite")
    print(f"{'='*60}")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"
    
    # Define tests to run
    tests = [
        (tests_dir / "test_pinecone.py", "Pinecone Configuration & Connection Test"),
        (tests_dir / "test_rag.py", "RAG System Functionality Test"),
    ]
    
    # Run tests
    results = []
    for test_script, description in tests:
        if test_script.exists():
            success = run_test(str(test_script), description)
            results.append((description, success))
        else:
            print(f"❌ Test script not found: {test_script}")
            results.append((description, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {description}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAG system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
