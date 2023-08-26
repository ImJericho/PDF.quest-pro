import streamlit as st
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS         #facebook AI similarty search
# from langchain.vectorstore import FAISS, Annoy, HNSW, NGT, AnnoyNGT, AnnoyHNSW, HNSWNGT, AnnoyHNSWNGT
from  langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

#models::::::::::
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub




def extract_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

def get_text_chunks(text):
    # i can play with this text_splitter 
    text_splitter = CharacterTextSplitter(
        # seprators=['\n', '.', '?', '!', ';', ':', '।', '॥', '\u0964', '\u0965'],
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,      #start the next chunk from 200 characters before the end of the previous chunk
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, model_name='bge'):    
    embeddings = None
    if(model_name=="openAI"):
        embeddings = OpenAIEmbeddings()      # you can't download the model and use it without an internet connection.
    if(model_name=="Bge"):
        embeddings = HuggingFaceInstructEmbeddings(model_name='BAAI/bge-large-en')
    elif(model_name=="Instructor"):
        embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')       ##takes about 2 min for LLM_test pdf

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    # st.write(response)
    

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 1:
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("Vivek"):
                st.write(message.content)


def main():
    load_dotenv()
    st.set_page_config(page_title='PDFQuestPro', page_icon=':books:', layout='wide')
    

    if "conversation" not in st.session_state:          # intialize the conversation
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('PDF.Quest-Pro :books:')
    st.markdown('created by Vivek Patidar')
    # user_question = st.text_input('Ask the question about the PDFs')
    
    user_question = st.chat_input("Ask question about the PDFs")
    
    # with st.chat_message("user"):
    #     st.write("this is user")
    
    # with st.chat_message("ME"):
    #     st.write("this is me")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader('Upload your PDFs here :envelope_with_arrow:', type=['pdf'], accept_multiple_files=True)

        embedding_model = st.selectbox('choose a embedding model',('openAI', 'Bge', 'Instructor'))
        # st.write('You selected:', option)
        raw_text = None
        
        if st.button('Process :bow_and_arrow:'):
            with st.spinner("Processing"):
                #get pdf text...............
                raw_text = extract_pdf_text(pdf_docs)                
                #i want to print this raw_text in my main page of streamlit app
                # st.write(raw_text)
                with st.chat_message("user"):
                    st.write(f"length of combined text:{len(raw_text)}")

                #get the text chunks......................
                text_chunks = get_text_chunks(raw_text)
                #i want to print the chunks of data in main page of streamlit app
                # st.write(text_chunks)


                #create vector store...................
                curr = time.time()
                vectorstore = get_vectorstore(text_chunks, model_name=embedding_model)
                now = time.time()
                vectorstore_time = now-curr
                with st.chat_message("user"):
                    st.write(f"Time taken to create vectorstore :{vectorstore_time: .0f} seconds")
                #write the time taken to create vectorstore in second without any decimal point

                #create conversation chain.....................
                curr = time.time()
                st.session_state.conversation = get_conversation_chain(vectorstore)     #st.session_state.conversation is a global variable in streamlit
                now = time.time()
                conversation_chain_time = now-curr

                with st.chat_message("user"):
                    st.write(f"Time taken to create conversation chain :{conversation_chain_time: .0f} seconds")

        # with st.chat_message("user"):
        #     st.write(f"length of combined text:{len(raw_text)}")

        # with st.chat_message("user"):
        #     st.write(f"Time taken to create vectorstore :{vectorstore_time: .0f} seconds")
            
        # with st.chat_message("user"):
        #     st.write(f"Time taken to create conversation chain :{conversation_chain_time: .0f} seconds")


if __name__ == '__main__':
    main()