#!/usr/bin/env python3
"""
Setup Scripts Index - Smart ATS RAG FAQ Assistant
Shows all available setup and utility scripts
"""

import os
import sys
from pathlib import Path

def print_header():
    """Print a nice header"""
    print("=" * 60)
    print("🚀 Smart ATS RAG FAQ Assistant - Setup Scripts")
    print("=" * 60)

def list_setup_scripts():
    """List all available setup scripts with descriptions"""
    
    scripts_dir = Path(__file__).parent
    
    scripts = [
        {
            "file": "complete_setup.py",
            "title": "🎯 Complete Automated Setup",
            "description": "Runs all setup steps automatically (RECOMMENDED)",
            "usage": "python scripts/complete_setup.py"
        },
        {
            "file": "download_dataset.py", 
            "title": "📊 Download Dataset",
            "description": "Downloads job descriptions dataset from Kaggle",
            "usage": "python scripts/download_dataset.py"
        },
        {
            "file": "init_vector_db.py",
            "title": "🗄️ Initialize Vector Database", 
            "description": "Populates Pinecone vector database with job descriptions",
            "usage": "python scripts/init_vector_db.py"
        },
        {
            "file": "setup.py",
            "title": "⚙️ Basic Setup",
            "description": "Original setup script for vector database",
            "usage": "python scripts/setup.py"
        },
        {
            "file": "validate.py",
            "title": "✅ System Validation",
            "description": "Validates system health and configuration",
            "usage": "python scripts/validate.py"
        }
    ]
    
    print("\\n📋 Available Setup Scripts:")
    print("-" * 40)
    
    for script in scripts:
        script_path = scripts_dir / script["file"]
        exists = "✅" if script_path.exists() else "❌"
        
        print(f"\\n{exists} {script['title']}")
        print(f"   📝 {script['description']}")
        print(f"   🔧 Usage: {script['usage']}")
        
        if not script_path.exists():
            print(f"   ⚠️ File not found: {script_path}")

def show_quick_start():
    """Show quick start instructions"""
    print("\\n" + "=" * 60)
    print("🚀 QUICK START (New Users)")
    print("=" * 60)
    print("""
1. First time setup:
   python scripts/complete_setup.py

2. Run the application:
   streamlit run app.py

3. Validate system health:
   python scripts/validate.py
""")

def show_individual_setup():
    """Show individual setup steps"""
    print("\\n" + "=" * 60)
    print("🔧 INDIVIDUAL SETUP STEPS")
    print("=" * 60)
    print("""
For manual step-by-step setup:

1. Download dataset:
   python scripts/download_dataset.py

2. Initialize vector database:
   python scripts/init_vector_db.py

3. Validate setup:
   python scripts/validate.py

4. Run application:
   streamlit run app.py
""")

def show_directory_structure():
    """Show the scripts directory structure"""
    print("\\n" + "=" * 60)
    print("📁 SCRIPTS DIRECTORY STRUCTURE")
    print("=" * 60)
    
    scripts_dir = Path(__file__).parent
    
    print(f"\\n{scripts_dir}/")
    for item in sorted(scripts_dir.iterdir()):
        if item.is_file() and item.suffix == '.py':
            size = item.stat().st_size
            print(f"  ├── {item.name} ({size:,} bytes)")
    
    # Show related directories
    parent_dir = scripts_dir.parent
    for dir_name in ['data', 'tests', 'rag']:
        dir_path = parent_dir / dir_name
        if dir_path.exists():
            print(f"\\n../{dir_name}/")
            for item in sorted(dir_path.iterdir()):
                if item.is_file():
                    print(f"  ├── {item.name}")

def main():
    """Main function"""
    print_header()
    list_setup_scripts()
    show_quick_start()
    show_individual_setup()
    show_directory_structure()
    
    print("\\n" + "=" * 60)
    print("💡 Need help? Check the README files in each directory!")
    print("=" * 60)

if __name__ == "__main__":
    main()
