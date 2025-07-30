import json
import os
import sys
import boto3
import streamlit as st

## Amazon Titan Embeddings model
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

## Vector Store
from langchain.vectorstores import FAISS

## LLM models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# Create LLM models
def get_claude_llm():
    llm = Bedrock(model_id="anthropic.claude-sonnet-4-20250514-v1:0", client=bedrock, model_kwargs={"max_tokens": 512})
    return llm

def get_llama3_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={"max_gen_len": 512})
    return llm

def get_deepseek_llm():
    llm = Bedrock(model_id="deepseek.r1-v1:0", client=bedrock, model_kwargs={"max_tokens": 512})
    return llm

def get_mistral_llm():
    llm = Bedrock(model_id="mistral.mistral-7b-instruct-v0:2", client=bedrock, model_kwargs={"max_tokens": 512})
    return llm

# Prompt Template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    answer = qa({"query": query})
    return answer["result"]

def main():
    st.set_page_config(
        page_title="RAG PROJECT with AWS BEDROCK",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Remove default Streamlit padding */
    .main > div {
        padding-top: 1rem;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #3949ab 25%, #1e88e5 75%, #42a5f5 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 15px 35px rgba(26, 35, 126, 0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    
    .main-title {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.4);
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    
    .subtitle {
        color: rgba(255,255,255,0.95) !important;
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        margin-top: 0.8rem;
        margin-bottom: 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    
    .aws-logo-container {
        display: flex;
        justify-content: center;
        margin: 1.5rem 0;
        padding: 0 1rem;
    }
    
    .aws-logo-large {
        width: 100%;
        max-width: 1477px;
        height: auto;
        aspect-ratio: 1477/256;
        object-fit: cover;
        border-radius: 15px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
        transition: all 0.4s ease;
        border: 3px solid rgba(255,255,255,0.1);
    }
    
    .aws-logo-large:hover {
        transform: translateY(-5px) scale(1.01);
        box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    }
    
    .aws-logo-fallback {
        background: linear-gradient(135deg, #FF9500 0%, #FF6B35 50%, #E65100 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        font-weight: 700;
        font-size: 2.2rem;
        text-align: center;
        width: 100%;
        max-width: 1477px;
        height: 256px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 12px 40px rgba(255, 149, 0, 0.3);
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        border: 3px solid rgba(255,255,255,0.2);
    }
    
    .question-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #dee2e6 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.8);
    }
    
    .model-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
        margin: 1.5rem 0;
    }
    
    .model-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.8rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 2px solid rgba(102, 126, 234, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .model-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .model-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.15);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .model-name {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.2rem;
        color: #1a237e;
        margin-bottom: 0.8rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .model-description {
        font-family: 'Inter', sans-serif;
        color: #37474f;
        font-size: 0.9rem;
        margin-bottom: 1rem;
        line-height: 1.6;
        font-weight: 400;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 50%, #66bb6a 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(46, 125, 50, 0.3);
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    .sidebar-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.3rem;
        margin: 0;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    
    .vector-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #e74c3c;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(231, 76, 60, 0.1);
    }
    
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #FF8C42 0%, #FF6B35 50%, #FF5722 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 1.5rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        border: 2px solid rgba(255,255,255,0.2) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(255, 107, 53, 0.5) !important;
        background: linear-gradient(135deg, #FF7A28 0%, #FF5722 50%, #D84315 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, #FF8C42 0%, #FF6B35 50%, #FF5722 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 4px 15px rgba(255, 140, 66, 0.3);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 3px solid #e1e8ed !important;
        padding: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        background: rgba(255,255,255,0.9) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FF8C42 !important;
        box-shadow: 0 0 20px rgba(255, 140, 66, 0.4) !important;
        background: white !important;
    }
    
    .bedrock-footer {
        margin-top: 3rem;
        text-align: center;
        padding: 0 1rem;
    }
    
    .bedrock-image {
        width: 100%;
        max-width: 1477px;
        height: auto;
        aspect-ratio: 1477/256;
        object-fit: cover;
        border-radius: 20px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.2);
        transition: all 0.4s ease;
        border: 3px solid rgba(255,255,255,0.1);
    }
    
    .bedrock-image:hover {
        transform: translateY(-5px) scale(1.01);
        box-shadow: 0 25px 70px rgba(0,0,0,0.25);
    }
    
    .bedrock-fallback {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #5a67d8 100%);
        color: white;
        padding: 50px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 2.8rem;
        text-align: center;
        width: 100%;
        max-width: 1477px;
        height: 256px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.3);
        text-shadow: 2px 2px 6px rgba(0,0,0,0.4);
        border: 3px solid rgba(255,255,255,0.2);
    }
    
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a237e;
        margin: 2rem 0 1.2rem 0;
        text-align: center;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .section-title::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #FF8C42 0%, #FF6B35 100%);
        border-radius: 2px;
    }
    
    /* Enhanced Responsive Design */
    @media (max-width: 768px) {
        .model-grid {
            grid-template-columns: 1fr;
            gap: 15px;
        }
        
        .main-title {
            font-size: 2.2rem;
        }
        
        .subtitle {
            font-size: 1rem;
        }
        
        .aws-logo-fallback {
            font-size: 1.8rem;
            padding: 20px;
            height: 200px;
        }
        
        .bedrock-fallback {
            font-size: 2rem;
            padding: 30px;
            height: 200px;
        }
        
        .question-container {
            padding: 1.5rem;
        }
        
        .model-card {
            padding: 1.2rem;
        }
        
        .section-title {
            font-size: 1.3rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-title {
            font-size: 1.8rem;
        }
        
        .aws-logo-fallback {
            font-size: 1.4rem;
        }
        
        .bedrock-fallback {
            font-size: 1.6rem;
        }
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🚀 RAG PROJECT with AWS BEDROCK</h1>
        <p class="subtitle">Advanced Retrieval-Augmented Generation using Amazon Bedrock AI Services</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AWS Logo Section - Large and centered
    st.markdown('<div class="aws-logo-container">', unsafe_allow_html=True)
    try:
        st.image("aws.jpeg", use_container_width=True, caption="")
    except:
        st.markdown('<div class="aws-logo-fallback">⚡ Amazon Web Services Platform</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Question Input Section
    st.markdown('<div class="question-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">💭 What\'s on Your Mind?</h3>', unsafe_allow_html=True)
    user_question = st.text_input(
        "",
        placeholder="Enter your question about the uploaded documents and let AI models analyze them...",
        help="Ask any question about your PDF documents and get intelligent responses from multiple advanced AI models"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Selection Grid
    st.markdown('<h3 class="section-title">🚀 Choose Your AI Model</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <div class="model-name">🧠 Claude Sonnet 4</div>
            <div class="model-description">Anthropic's most advanced reasoning model with superior comprehension, analytical capabilities, and nuanced understanding</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 Get Claude Response", key="claude"):
            if user_question:
                with st.spinner("🔄 Claude is analyzing your documents..."):
                    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                    llm = get_claude_llm()
                    response = get_response_llm(llm, faiss_index, user_question)
                    st.markdown("#### 🧠 Claude Sonnet 4 Response:")
                    st.markdown(f"**{response}**")
                    st.success("✅ Analysis Complete!")
            else:
                st.warning("⚠️ Please enter a question first!")
    
    with col2:
        st.markdown("""
        <div class="model-card">
            <div class="model-name">🦙 Llama 3 70B</div>
            <div class="model-description">Meta's powerful open-source model with exceptional performance, multilingual support, and robust reasoning capabilities</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 Get Llama Response", key="llama"):
            if user_question:
                with st.spinner("🔄 Llama is processing your query..."):
                    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                    llm = get_llama3_llm()
                    response = get_response_llm(llm, faiss_index, user_question)
                    st.markdown("#### 🦙 Llama 3 Response:")
                    st.markdown(f"**{response}**")
                    st.success("✅ Analysis Complete!")
            else:
                st.warning("⚠️ Please enter a question first!")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class="model-card">
            <div class="model-name">🔮 Mistral 7B</div>
            <div class="model-description">Efficient and fast model optimized for quick responses, resource efficiency, and streamlined processing</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 Get Mistral Response", key="mistral"):
            if user_question:
                with st.spinner("🔄 Mistral is generating your answer..."):
                    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                    llm = get_mistral_llm()
                    response = get_response_llm(llm, faiss_index, user_question)
                    st.markdown("#### 🔮 Mistral Response:")
                    st.markdown(f"**{response}**")
                    st.success("✅ Analysis Complete!")
            else:
                st.warning("⚠️ Please enter a question first!")
    
    with col4:
        st.markdown("""
        <div class="model-card">
            <div class="model-name">🧬 DeepSeek R1</div>
            <div class="model-description">Advanced reasoning model with deep analytical capabilities, complex problem solving, and sophisticated inference</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 Get DeepSeek Response", key="deepseek"):
            if user_question:
                with st.spinner("🔄 DeepSeek is thinking deeply..."):
                    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                    llm = get_deepseek_llm()
                    response = get_response_llm(llm, faiss_index, user_question)
                    st.markdown("#### 🧬 DeepSeek Response:")
                    st.markdown(f"**{response}**")
                    st.success("✅ Analysis Complete!")
            else:
                st.warning("⚠️ Please enter a question first!")

    # Bedrock Footer Image
    st.markdown('<div class="bedrock-footer">', unsafe_allow_html=True)
    try:
        st.image("bedrock.jpeg", use_container_width=True, caption="")
    except:
        st.markdown('<div class="bedrock-fallback">🏗️ Amazon Bedrock AI Platform</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2 class="sidebar-title">🚀 Developed by Enes Aydin</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="vector-card">', unsafe_allow_html=True)
        st.markdown("#### 🔄 Update Document Index")
        st.markdown("Process your PDF documents and create searchable vector embeddings using Amazon Titan for enhanced retrieval accuracy.")
        
        if st.button("🚀 Process Documents", key="vector_update"):
            with st.spinner("⚙️ Processing documents and creating embeddings..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("🎉 Vector store updated successfully!")
                st.balloons()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown("""
        <div class="feature-highlight">
            💡 <strong>Pro Tip:</strong> Upload PDFs to the 'data' folder before processing for optimal results!
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🌟 Key Features")
        st.markdown("""
        - 🤖 **4 Advanced AI Models** - Claude, Llama, Mistral, DeepSeek
        - 🔍 **Semantic Search Engine** - FAISS-powered vector similarity
        - 📚 **Multi-Document Support** - Process multiple PDFs simultaneously
        - ⚡ **Real-time Processing** - Instant document analysis and responses
        - 🎯 **Accurate Information Retrieval** - RAG-enhanced precision
        - 🔒 **Enterprise Security** - AWS Bedrock infrastructure
        - 🌐 **Scalable Architecture** - Cloud-native design
        """)
        
        st.markdown("---")
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        - **AWS Bedrock** - Enterprise AI Platform & Model Access
        - **FAISS** - High-performance Vector Search Engine
        - **LangChain** - Advanced AI Framework & Orchestration
        - **Streamlit** - Interactive Web Interface
        - **Amazon Titan** - Professional Embeddings Model
        - **RAG Architecture** - Retrieval-Augmented Generation
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Comparison")
        st.markdown("""
        **Claude Sonnet 4**: Best for complex reasoning and analysis  
        **Llama 3 70B**: Excellent for general tasks and multilingual content  
        **Mistral 7B**: Fastest responses with good efficiency  
        **DeepSeek R1**: Superior for mathematical and logical problems
        """)

if __name__ == "__main__":
    main()