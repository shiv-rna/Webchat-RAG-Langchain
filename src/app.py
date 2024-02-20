
import os

import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


def set_openai_key(api_key):
    """Set the OpenAI API key in the environment variable.
    Args:
    key (str): The OpenAI API key to set.
    """
    os.environ["OPENAI_API_KEY"] = f"{api_key}"


def get_vectorstore_from_url(url):
    """Retrieve vector store from the given URL.

    This function loads a document from the provided URL,
    splits it into chunks & creates a vector store.

    Args:
        url (str): The URL of the document.

    Returns:
        Chroma: The vector store created from the document chunks.
    """
    # Get text-in documents
    loader = WebBaseLoader(url)
    document = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Create Vector-store from the chunks:
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(document_chunks, embeddings)
    return vector_store


def get_context_retriever_chain(vector_store):
    """Create a context retriever chain using the given vector store.

    This function creates a history-aware retriever chain
    using a vector store and a chat prompt template.

    Args:
        vector_store (Chroma): The vector store to use.

    Returns:
        RetrievalChain: The context retriever chain.
    """
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query "
                 "to look up in order to get information relevant to "
                 "the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    """Create a conversational RAG chain
    using the given context retriever chain.

    This function creates a conversational RAG chain
    using a context retriever chain and a chat prompt template.

    Args:
        retriever_chain (RetrievalChain): The context retriever chain.

    Returns:
        RetrievalChain: The conversational RAG chain.
    """
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer the user's questions based"
            " on the below context:\n\n{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    """Get the response for the given user query.

    This function retrieves the response for the user query
    using the conversational RAG chain.

    Args:
        user_input (str): The user query.

    Returns:
        str: The response to the user query.
    """
    # Create Conversation Chain
    retriever_chain = get_context_retriever_chain(
        st.session_state.vector_store
        )
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    # Invoke conversational RAG chain
    inv_response = conversation_rag_chain.invoke({
          "chat_history": st.session_state.chat_history,
          "input": user_input
      })
    return inv_response['answer']


# App Config
st.set_page_config(page_title="Chat  with Websites", page_icon="🦜")
st.title("_Chat_ with :blue[Websites] 🤖")

# App Sidebar
with st.sidebar:
    st.header(":red[Settings]", divider='rainbow')
    key = st.text_input("OpenAI API key 🔑", type="password")
    set_openai_key(key)
    website_url = st.text_input("Website URL 🌐")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    # For persistent variables : Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Yellow👋 I am a bot, Need help right ?")
            ]

    # Persistent Vector Store
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # User Input
    user_query = st.chat_input("Type your message ✍")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Conversation
    for message in st.session_state.chat_history:
        MESSAGE_TYPE = "AI" if isinstance(message, AIMessage) else "Human"
        with st.chat_message(MESSAGE_TYPE):
            st.write(message.content)
