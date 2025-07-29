from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
from dotenv import load_dotenv

import pprint
load_dotenv()
file_path =Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
split_docs=text_splitter.split_documents(documents)
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",

)
docs_embeddings = embeddings_model.embed_documents([doc.page_content for doc in split_docs])
# chunking the documents

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="learning_vector",
    embedding=embeddings_model,
)

print("Vector store created and documents embedded successfully.")
