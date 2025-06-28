# 1. Install required packages if not already installed:
# pip install langchain openai chromadb tiktoken

import os
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

from dotenv import load_dotenv

# Load environment variables (especially your OpenAI API key)
load_dotenv()

# 2. Load the document (data dictionary)
loader = TextLoader("docs/tableau_superstore_data_dictionary.txt")
documents = loader.load()

# 3. Split the document into manageable chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 4. Create embeddings from the text
embedding = OpenAIEmbeddings()

# 5. Store in a vector database (Chroma)
vectorstore = Chroma.from_documents(docs, embedding, persist_directory="rag_vectorstore")
vectorstore.persist()

# 6. Set up a retriever
db_retriever = vectorstore.as_retriever()

# 7. Initialize the chat model
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

# 8. Build Retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db_retriever,
    return_source_documents=True
)

# 9. Example query
# query = "What does the Discount column represent in the Superstore dataset?"
query = "What are the various columns I have access to in the data set?"
response = qa_chain({"query": query})

print("\nAnswer:", response["result"])
print("\nRelevant Sources:")
for doc in response["source_documents"]:
    print("-", doc.metadata["source"])
