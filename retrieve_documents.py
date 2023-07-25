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

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(vectordb._collection.count())

question = "Tell me about the cost of preventive care services?"

docs_ss = vectordb.similarity_search(question,k=3)

print(str(docs_ss[0].page_content) + "¥n" + str(docs_ss[0].metadata))
print("---------------")
print(str(docs_ss[1].page_content) + "¥n" + str(docs_ss[1].metadata))
print("---------------")
print(str(docs_ss[2].page_content) + "¥n" + str(docs_ss[2].metadata))

docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)

print("Document MMR---------------")

print(str(docs_mmr[0].page_content) + "¥n" + str(docs_mmr[0].metadata))
print("---------------")
print(str(docs_mmr[1].page_content) + "¥n" + str(docs_mmr[1].metadata))
print("---------------")
print(str(docs_mmr[2].page_content) + "¥n" + str(docs_mmr[2].metadata))
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]

document_content_description = "Lecture notes"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)

question = "what did they say about regression in the third lecture?"
docs = retriever.get_relevant_documents(question)

for d in docs:
    print(d.metadata)


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)

question = "Tell me about the cost of preventive care services?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)