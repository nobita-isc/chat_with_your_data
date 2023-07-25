import os
import openai
import sys
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv

# Do not understand what does this mean
sys.path.append('../..')

# read local .env file
_ = load_dotenv(find_dotenv()) 
openai.api_key  = os.environ['OPENAI_API_KEY']

# document loading
loaders = [
    PyPDFLoader("docs/Northwind_Health_Plus_Benefits_Details.pdf"),
    PyPDFLoader("docs/Northwind_Standard_Benefits_Details.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# document splitting
# we also have CharacterTextSplitter and TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# for Japanese, we can consider separators liked "。" character
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
splits = text_splitter.split_documents(docs)

# embedding api
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

# vector storage (Chroma)
from langchain.vectorstores import Chroma
persist_directory = 'docs/chroma/'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
print(vectordb._collection.count())

vectordb.persist()