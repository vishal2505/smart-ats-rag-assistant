# /Users/vishalmishra/MyDocuments/SMU_MITB/Term-3/Gen_AI_with_LLM/Project/Smart_ATS_With_RAG/pages/faq_assistant.py
import os
import streamlit as st
import sys
import warnings
from datetime import datetime

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embeddings import get_embedding_function
from rag.vector_store import get_or_create_vector_store
from rag.retriever import get_retriever, get_multi_query_retriever, get_contextual_retriever
from rag.llm_service import get_llm
from rag.rag_qa_chain import create_rag_chain, create_conversation_chain

st.set_page_config(
    page_title="Career FAQ Assistant", 
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

def classify_query_type(query):
    """Classify the query type for contextual retrieval"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['resume', 'cv', 'application', 'portfolio']):
        return "resume"
    elif any(word in query_lower for word in ['interview', 'preparation', 'questions', 'behavioral']):
        return "interview"
    elif any(word in query_lower for word in ['salary', 'compensation', 'negotiate', 'pay', 'benefits']):
        return "salary"
    else:
        return "general"

def get_suggested_questions():
    """Return a list of suggested questions for users"""
    return [
        "How do I optimize my resume for ATS systems?",
        "What are the most important skills for a software developer?",
        "How should I prepare for a technical interview?",
        "What questions should I ask during an interview?",
        "How do I negotiate my salary effectively?",
        "What are the current trends in data science jobs?",
        "How do I transition to a career in AI/ML?",
        "What are red flags to look for in job postings?",
        "How do I write an effective cover letter?",
        "What networking strategies work best for job searching?"
    ]

def process_question(prompt, qa_chain, retrieval_strategy, vector_store, search_kwargs, enable_memory, llm):
    """Process a question and return the response"""
    try:
        # Use contextual retriever if selected
        if retrieval_strategy == "contextual":
            query_type = classify_query_type(prompt)
            contextual_retriever = get_contextual_retriever(vector_store, query_type, search_kwargs)
            # Create a new chain with the contextual retriever
            if enable_memory:
                qa_chain = create_conversation_chain(llm, contextual_retriever)
            else:
                qa_chain = create_rag_chain(llm, contextual_retriever)
        
        # Prepare input based on chain type
        if enable_memory and hasattr(qa_chain, 'memory'):
            # For conversation chain with memory - check if it's ConversationalRetrievalChain
            if hasattr(qa_chain, 'combine_docs_chain'):
                response = qa_chain.invoke({"question": prompt, "chat_history": []})
            else:
                response = qa_chain.invoke({"question": prompt})
        else:
            # For regular RAG chain - use invoke instead of __call__
            response = qa_chain.invoke({"query": prompt})
        
        # Extract answer from response - handle different response formats
        answer = None
        if isinstance(response, dict):
            answer = response.get("answer") or response.get("result", "I apologize, but I couldn't generate a response.")
        else:
            answer = str(response)
            
        sources = response.get("source_documents", []) if isinstance(response, dict) else []
        
        return answer, sources, None
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        return None, [], error_msg

def display_sources(sources):
    """Display source documents in an organized way"""
    if not sources:
        return
        
    with st.expander("📚 Sources & References", expanded=False):
        for i, doc in enumerate(sources):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                title = doc.metadata.get('title', f'Document {i+1}')
                doc_type = doc.metadata.get('doc_type', 'Unknown')
                
                st.markdown(f"**{title}**")
                st.caption(f"Type: {doc_type}")
                
                # Show preview of content
                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                st.text(preview)
                
            with col2:
                # Show relevance score if available
                if hasattr(doc, '_distance'):
                    relevance = 1 - doc._distance
                    st.metric("Relevance", f"{relevance:.2%}")
                    
            st.divider()

def main():
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .suggestion-button {
        margin: 5px;
        padding: 10px;
        background-color: #f0f2f6;
        border: 1px solid #d3d3d3;
        border-radius: 5px;
        cursor: pointer;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        align-self: flex-end;
    }
    .assistant-message {
        background-color: #f5f5f5;
        align-self: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🤖 Career FAQ Assistant</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
    🎯 <strong>Your AI-powered career guidance companion</strong><br>
    Get expert advice on resumes, interviews, job searching, and career development based on 
    thousands of job descriptions and industry best practices.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Settings
        st.subheader("AI Model Settings")
        llm_provider = st.selectbox(
            "Select AI Provider",
            ["groq", "openai"],
            index=0,
            help="Choose your preferred AI provider"
        )
        
        if llm_provider == "openai":
            model = st.selectbox(
                "Select OpenAI Model",
                ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
                index=0
            )
        else:
            model = st.selectbox(
                "Select Groq Model",
                ["llama3-8b-8192", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
                index=0
            )
        
        # Retrieval Settings
        st.subheader("Search Settings")
        retrieval_strategy = st.selectbox(
            "Retrieval Strategy",
            ["contextual", "similarity", "multi_query", "mmr"],
            index=0,
            help="Choose how the system searches for relevant information"
        )
        
        num_sources = st.slider(
            "Number of Sources",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of source documents to retrieve"
        )
        
        # Chat Settings
        st.subheader("Chat Settings")
        enable_memory = st.checkbox(
            "Enable Conversation Memory", 
            value=True,
            help="Remember previous messages in the conversation"
        )
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.conversation_memory = []
            st.rerun()
        
        # Statistics
        if "messages" in st.session_state:
            st.subheader("📊 Session Stats")
            st.metric("Messages", len(st.session_state.messages))
            st.metric("Queries Answered", len([m for m in st.session_state.messages if m["role"] == "assistant"]))

    # Initialize RAG components
    try:
        with st.spinner("🔄 Loading AI Assistant..."):
            embedding_function = get_embedding_function()
            vector_store = get_or_create_vector_store(embedding_function)
            
            # Select retriever based on user choice
            search_kwargs = {"k": num_sources}
            
            if retrieval_strategy == "contextual":
                # We'll determine context per query
                retriever = get_retriever(vector_store, search_kwargs)
            elif retrieval_strategy == "multi_query":
                llm = get_llm(provider=llm_provider, model=model)
                retriever = get_multi_query_retriever(vector_store, llm, search_kwargs)
            else:
                retriever = get_retriever(vector_store, search_kwargs, retrieval_strategy)
            
            llm = get_llm(provider=llm_provider, model=model)
            
            if enable_memory:
                qa_chain = create_conversation_chain(llm, retriever)
            else:
                qa_chain = create_rag_chain(llm, retriever)
            
        st.success("✅ AI Assistant loaded successfully!")
        
    except Exception as e:
        st.error(f"❌ Error loading AI Assistant: {str(e)}")
        st.info("💡 Make sure you have set up your API keys in the .env file")
        return

    # Initialize chat history and memory
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []    # Suggested questions section
    if not st.session_state.messages:
        st.subheader("💡 Suggested Questions")
        st.markdown("Click on any question below to get started:")
        
        suggestions = get_suggested_questions()
        cols = st.columns(2)
        
        for i, suggestion in enumerate(suggestions):
            col = cols[i % 2]
            with col:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    # Add the suggested question as user input to chat history
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    
                    # Generate and add response for suggested question
                    with st.spinner("🤔 Thinking..."):
                        try:
                            answer, sources, error_msg = process_question(
                                suggestion, qa_chain, retrieval_strategy, vector_store, 
                                search_kwargs, enable_memory, llm
                            )
                            
                            if error_msg:
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": error_msg,
                                    "timestamp": datetime.now().isoformat()
                                })
                            else:
                                # Add assistant response to chat history with sources
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": answer,
                                    "sources": sources,
                                    "timestamp": datetime.now().isoformat()
                                })
                            
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_msg,
                                "timestamp": datetime.now().isoformat()
                            })
                    
                    # Rerun to display the new messages
                    st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                display_sources(message["sources"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about careers, resumes, interviews, or job searching..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    answer, sources, error_msg = process_question(prompt, qa_chain, retrieval_strategy, vector_store, search_kwargs, enable_memory, llm)
                    
                    if error_msg:
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        # Display the answer
                        st.markdown(answer)
                        
                        # Display sources
                        if sources:
                            display_sources(sources)
                        
                        # Add assistant response to chat history with sources
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
    💼 Powered by Advanced RAG Technology | 
    🎯 Smart ATS Resume Analyzer | 
    🤖 Built with Langchain & Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()