import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.document_loaders import DirectoryLoader
import openai

# Load environment variables
load_dotenv(find_dotenv())
#openai.api_key = os.getenv("OPENAI_API_KEY")

PERSIST_DIRECTORY = "./vector_db"
DOCUMENT_FOLDER = "./test_data"

# Initialize Chroma with persistence settings
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=OpenAIEmbeddings()
)

# Load and vectorize documents from a folder
def load_and_vectorize_documents(folder_path):
    loader = DirectoryLoader(folder_path, glob="*.txt")
    documents = loader.load()
    vectorstore.add_documents(documents)
    vectorstore.persist()

# Check if vector store exists and has documents
def check_vector_store():
    collection_count = len(vectorstore._collection.get()['ids'])
    if collection_count == 0:
        st.error("No documents found in the collection. Please check the folder path and document format.")
    else:
        st.success(f"Found {collection_count} documents in the collection.")

# Set up the conversational chain

def create_conversational_chain():
    # Define your prompt template
    template = """
    You are an AI assistant for answering questions about my documents.
    You are given the following extracted parts of a document and a question. Provide a conversational answer.
    If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer. Be very precise and answer to the point.
    Lastly, answer the question in a easy to understand with bit of fun and humor.

    Question: {question}
    =========
    {context}
    =========
    Answer in Markdown:
    """
    promptTemplate = PromptTemplate(
        template=template, 
        input_variables=["question", "context"]
    )


    llm = ChatOpenAI(
        temperature=0, 
        model="gpt-4o-mini",
        streaming=True)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    return ConversationalRetrievalChain.from_llm(
        llm, 
        retriever, 
        combine_docs_chain_kwargs={'prompt': promptTemplate})


# Create the conversational chain
chain = create_conversational_chain()


if __name__ == "__main__":
    # Set up Streamlit app
    st.set_page_config(page_title="Personal AI Assistant", page_icon=":robot:")
    st.sidebar.title("Personal AI Assistant")
    st.sidebar.markdown("## Settings")
    st.sidebar.markdown("### Document Settings")
    st.sidebar.markdown("#### Document Folder")
    st.sidebar.text_input("Folder Path", value=DOCUMENT_FOLDER)
    st.sidebar.markdown("#### Vector Store")
    st.sidebar.button("Check Vector Store", on_click=check_vector_store)
    st.sidebar.markdown("### Chat Settings")
    st.sidebar.markdown("#### Model")
    model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"])
    st.sidebar.markdown("#### Temperature")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
    st.sidebar.markdown("### About")
    st.sidebar.markdown("This is a personal AI assistant that can answer questions about your documents.")

    st.title("Welcome to your Personal AI Assistant")
    st.write("This application allows you to interact with your documents using AI.")

    if st.button("Load and Vectorize Documents"):
        load_and_vectorize_documents(DOCUMENT_FOLDER)
        st.write("Documents loaded and vectorized.")

    # Update the LLM in the chain with the selected parameters like model and temperature
    updated_llm = ChatOpenAI(model=model, temperature=temperature)
    chain.combine_docs_chain.llm_chain.llm = updated_llm

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    # Display assistant's initial message if there is no historical messages in chat session
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.write("Hello! I am your personal AI assistant. How can I help you today?")

    # User input
    if prompt := st.chat_input("Ask me something..."):

        # Add user message to chat history
        st.session_state.messages.append(("user", prompt))

        #Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = chain({"question": prompt, "chat_history": st.session_state.messages})
            st.write(response['answer'])

        st.session_state.messages.append(("assistant", response['answer']))