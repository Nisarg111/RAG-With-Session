import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from huggingface_hub import login
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

# Embedding and ChatGroq Initialization
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="Gemma2-9b-it",api_key=GROQ_API_KEY)

# Streamlit 
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload Pdf's and chat with their content")

session_id = st.text_input("Session ID",value="default_session")
# statefull management
if session_id:

    if 'store' not in st.session_state:
        st.session_state.store={}
    
    uploaded_files = st.file_uploader("Choose A PDF file",type="pdf",accept_multiple_files=True)
    # Process upload files
    if uploaded_files:
        document = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            document.extend(docs)
        # Creation of retriever
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents=document)
        vectorstore = FAISS.from_documents(documents=splits,embedding= embedding)
        
        retriever = vectorstore.as_retriever()
        
        # Question Interpretaion By System
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer Design
        system_prompt = (
            "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversional_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        st.write(conversional_rag_chain)
        user_input = st.text_input("Your Question: ")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversional_rag_chain.invoke(
                {
                    "input":user_input
                },
                config={"configurable":{"session_id":session_id}}
            )
            st.write("Assistant:", response['answer'])

    
