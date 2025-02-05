import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import openai
from openai import api_key
from os import environ

os.environ["OPENAI_API_KEY"] = "Your key"

loader = PyPDFLoader('resumes.pdf')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_documents(docs, embeddings)

query = input("Ask me anything: ")
relevant_docs = vectordb.similarity_search(query, k=3)
document_texts = "\n\n".join([doc.page_content for doc in relevant_docs])

prompt = f"Based on the following documents, answer the query: {query}\n\n{document_texts}"

# Initialize LLM
llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])

# Generate response
response = llm.invoke(prompt)

print(response)