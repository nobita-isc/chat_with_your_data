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

import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

#question = "What is included in health plus plan?"
#docs = vectordb.similarity_search(question,k=3)
#len(docs)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# Prompt
from langchain.prompts import PromptTemplate
# Build prompt
template = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum. Keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

question = "What is included in health plus plan?"
result1 = qa_chain({"query": question})
print("Q: " + question)
print("A: " + str(result1["result"]))

# Memory
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

def get_chat_history(chat_turns):
    return "\n".join(chat_turns)

# Converstational Retrieval Chain
from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever(search_type="mmr")
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    return_source_documents=True,
    get_chat_history=get_chat_history,
)

def print_reference_source(source_documents):
    print("Reference Sources:")
    for documents in source_documents:
        print(documents.page_content)
        print(documents.metadata)
    print("xxxxxxxx End of Reference Sources xxxxxxxxx")

chat_history = []
question = "What is included in health standard plan?"
result = qa({"question": question, "chat_history": chat_history})
print("")
print("Q: " + question)
print(result['answer'])
print_reference_source(result["source_documents"])

question = "How about preventive care service? Does it cover dental service?"
result = qa({"question": question, "chat_history": chat_history})
print("")
print("Q: " + question)
print(result['answer'])
print_reference_source(result["source_documents"])

question = "What is the price?"
print("")
result = qa({"question": question, "chat_history": chat_history})
print("Q: " + question)
print(result['answer'])
print_reference_source(result["source_documents"])

question = "What is not included?"
print("")
result = qa({"question": question, "chat_history": chat_history})
print("Q: " + question)
print(result['answer'])
print_reference_source(result["source_documents"])

#question = "What is not included in health standard plan?"

# result = qa_chain({"query": question})

# print(result["result"])
# print(result["source_documents"][0])

# Map reduce
# qa_chain_mr = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectordb.as_retriever(),
#     chain_type="map_reduce"
# )
# result = qa_chain_mr({"query": question})

# print(result["result"])

# Refine
# print("")
# print("Refine------------------------------")
# qa_chain_mr = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectordb.as_retriever(),
#     chain_type="refine"
# )
# result = qa_chain_mr({"query": question})
# print(result["result"])
