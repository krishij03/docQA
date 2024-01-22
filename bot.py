import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import gdown

def get_pdf_from_drive(url):
    output = 'temp.pdf'
    gdown.download(url, output, quiet=False)
    with open(output, 'rb') as f:
        return BytesIO(f.read())

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        if isinstance(pdf, BytesIO):
            pdf_reader = PdfReader(pdf)
        else:
            pdf_reader = PdfReader(BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
def convert_to_direct_link(shareable_link):
    file_id = shareable_link.split('/')[-2]
    direct_link = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    return direct_link

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header("Chat with multiple PDFs")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here:", accept_multiple_files=True)
        webpage_urls = st.text_input("Enter URLs to fetch text (separate multiple URLs with a comma):")
        drive_urls = st.text_input("Enter a Google Drive URL of a PDF:")
        if st.button("Process"):
            with st.spinner("Processing"):
                all_text = ""
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    all_text += raw_text + " "
                if webpage_urls:
                    for webpage_url in webpage_urls.split(','):
                        webpage_url = webpage_url.strip()
                        webpage_text = get_text_from_url(webpage_url)
                        all_text += webpage_text + " "
                if drive_urls:
                    for drive_url in drive_urls.split(','):
                        drive_url = drive_url.strip()
                        direct_link = convert_to_direct_link(drive_url)
                        pdf_data = get_pdf_from_drive(direct_link)
                        raw_text = get_pdf_text([pdf_data])
                        all_text += raw_text + " "
                text_chunks = get_text_chunks(all_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()