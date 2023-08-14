# LangChain Dependencies
from langchain import ConversationChain, PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS

#For Streamlit to build UI
from streamlit_chat import message
import streamlit as st

#Load Configuration
import os
from dotenv import load_dotenv

# Bootstrap
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


# Data loader. In this tutorial, we use PDF loader to load PDF
loader = PyMuPDFLoader("./pdf_library/JOODB.pdf")
docs = loader.load()

# Embedding does convert text to vector
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# We create mem(Chroma) for the application.

#chroma_mem = Chroma.from_documents(documents=docs, embedding_function=embeddings)
#Change from Chroma to FAISS
db = FAISS.from_documents(docs, embeddings)

#We create Index, which indexes data on Chroma
retriever = db.as_retriever(search_kwargs=dict(k=5))
#We create a VectorStore (Chroma Wrapper)
memory = VectorStoreRetrieverMemory(retriever=retriever)

# Template
_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)

llm = OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)  # Can be any valid LLM

conversation_with_summary = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    # We set a very low max_token_limit for the purposes of testing.
    memory=memory,
    verbose=True
)

st.header("PDF-chatGPT")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = conversation_with_summary.predict(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
