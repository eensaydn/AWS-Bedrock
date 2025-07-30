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
## Vectore Store
from langchain.vectorstores import FAISS
## LLM models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

#Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

#Data Ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    #Text splitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs=text_splitter.split_documents(documents)
    return docs

#Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(docs,bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

#Create LLM models
    
def get_claude_llm():
    llm = Bedrock(model_id="anthropic.claude-sonnet-4-20250514-v1:0", client=bedrock, model_kwargs={"max_tokens": 512})
    return llm

def get_llama3_llm():
    llm=Bedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock,model_kwargs={"max_gen_len": 512})
    return llm

def get_deepseek_llm():
    llm=Bedrock(model_id="deepseek.r1-v1:0",client=bedrock,model_kwargs={"max_tokens": 512})
    return llm
    
def get_mistral_llm():
    llm=Bedrock(model_id="mistral.mistral-7b-instruct-v0:2",client=bedrock,model_kwargs={"max_tokens": 512})
    return llm

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k":3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":prompt}
    )
    answer=qa({"query":query})
    return answer["result"]

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_claude_llm()
    
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_llama3_llm()

    if st.button("Mistral Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_mistral_llm()

    if st.button("DeepSeek Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_deepseek_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()