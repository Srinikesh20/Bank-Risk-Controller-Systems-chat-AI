from dotenv import load_dotenv  # type: ignore
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # Facebook AI similarity search
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import docx  # type: ignore
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import StdOutCallbackHandler
from streamlit_chat import message  # type: ignore
import requests

st.set_page_config(page_title="BANK RISK MANAGEMENT")

def run_chatbot():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    st.header("BANKBOT")

    # Display an image for the header
    st.image("C:/Users/91979/Downloads/bank/—Pngtree—futuristic 3d chatbot design showcasing_19220247.png", caption="Welcome!")

    # Process the inbuilt PDF
    inbuilt_pdf_path = "C:/Users/91979/Downloads/Final project-20250110T061811Z-001/Final project/BANKING Q&A.pdf"  # Path to your inbuilt PDF
    if os.path.exists(inbuilt_pdf_path):
        files_text = get_pdf_text(inbuilt_pdf_path)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vetorestore)
        st.session_state.processComplete = True
    else:
        st.error("Please check the file path.")

    if st.session_state.processComplete:
        user_question = st.chat_input("How can I help you!")
        if user_question:
            handel_userinput(user_question)


def get_pdf_text(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=500,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base


def get_conversation_chain(vetorestore):
    handler = StdOutCallbackHandler()
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.5, "max_length": 150000}  # Adjusted for longer and more coherent answers
    )
    retriever = vetorestore.as_retriever()
    retriever.search_kwargs = {"k": 5}  # Fetch more documents for context
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        callbacks=[handler]
    )
    return conversation_chain


def handel_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Custom fallback message
    custom_fallback_message = "I'm not sure about that. I'll make a note to learn more about this topic."

    # Display the chat history
    response_container = st.container()
    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:  # User message
                message(messages.content, is_user=True, key=str(i))
            else:  # Chatbot response
                # Check if the response contains typical "I don't know" phrases or is empty
                if "i don't know" in messages.content.lower() or "not sure" in messages.content.lower() or len(messages.content.strip()) == 0:
                    message(custom_fallback_message, key=str(i))
                else:
                    message(messages.content, key=str(i))


if __name__ == "__main__":
    run_chatbot()