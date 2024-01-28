from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent, create_csv_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_community.document_loaders import DataFrameLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st
import pandas as pd
import os
import re


# Opening Session
st.set_page_config(page_title="LangChain: Chat with pandas DataFrame", page_icon="🦜")
st.title("Chat with Google Sheet data")

# Sidebar check openai key
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()

url_googlesheet = st.sidebar.text_input("Enter Google Sheet URL")

url = 'https://docs.google.com/spreadsheets/d/1OReP9PTU8p6VNrGRX1ePl0D6h9mhDMy1bO8B8pRjhzM/edit?usp=sharing'

if not url_googlesheet:
    url_googlesheet = url

# define gg sheet url and prompt
PREFIX="""You are a product analyst with the information given in the sheet below. 
You will give the answer based on the information in the sheet only. 
If you cannot find the information, say I don't know.
Please reply with the user's language. If user speaks Dutch, reply with Dutch"""

PREFIX="""You are a product analyst that will give the answer based on the provided context only. 
Please reply with the user's language. If user speaks Dutch, reply with Dutch
If you cannot find the information, must reply I don't know in the user's language."""

user_prefix = st.sidebar.text_area("Instruction for Chatbot (Optional)", height=200, placeholder=PREFIX)

if not user_prefix:
    user_prefix = PREFIX

template = user_prefix + """
The context provided:
{context}

Question: {question}
Answer:
"""

# Define function to convert gg sheet url to csv url
def convert_google_sheet_url(url):
    # Regular expression to match and capture the necessary part of the URL
    pattern = r'https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)(/edit#gid=(\d+)|/edit.*)?'

    # Replace function to construct the new URL for CSV export
    # If gid is present in the URL, it includes it in the export URL, otherwise, it's omitted
    replacement = lambda m: f'https://docs.google.com/spreadsheets/d/{m.group(1)}/export?' + (f'gid={m.group(3)}&' if m.group(3) else '') + 'format=csv'

    # Replace using regex
    new_url = re.sub(pattern, replacement, url)

    return new_url


# Define function to clear submit button
def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False



# Define function to create models
@st.cache_resource(show_spinner="Load models...")
def create_models():
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    llm = ChatOpenAI(
            temperature=0.5, model="gpt-4-0125-preview", openai_api_key=openai_api_key, verbose=True
        )
    return llm, embedding_function

# Define function to load data
@st.cache_data(ttl=20, show_spinner="Fetching data from Google Sheet...")
def load_data(url):
    new_url = convert_google_sheet_url(url)
    df = pd.read_csv(new_url )
    df['text'] = df['Question'] + ': ' + df['Answer']

    loader = DataFrameLoader(df, page_content_column='text')
    documents = loader.load()

    return documents

# Define function to qa_chain
def create_agent(url):
    
    # Load data
    documents = load_data(url)

    # Create models
    llm, embedding_function = create_models()

    # Create vectorstore
    persist_directory = 'docs/chroma/'
    db = Chroma.from_documents(documents, 
                            embedding_function, 
                            persist_directory=persist_directory)

    # Create prompt
    QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=["context" ,"question"])


    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type='mmr', k=3, fetch_k=5),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, "verbose": True},
        verbose=True,
    )

    return qa_chain


# Display the Clear Conversation History button
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


# Create qa_chain
agent = create_agent(url_googlesheet)

# Display the conversation
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])





# Check the prompt is not empty
if prompt := st.chat_input(placeholder="What is this data about?"):

    # append to the conversation
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Run the agent
    with st.chat_message("assistant"):
        
        print(st.session_state.messages)
        print("prompt: ", prompt)

        response = agent({"query": prompt})['result']
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)