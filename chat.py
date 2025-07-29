# Take a query from user side
# Make a connection to the vector database
# Convert the query into an embedding
# Search the vector database for similar embeddings
# Return the results to the user

from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
vector_store = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_vector",
    embedding=embeddings_model,
)

query =input("/ >  Enter your query: ")

results=vector_store.similarity_search(
    query=query,
)

context= "\n\n\n".join([f"page_content : {result.page_content} \n page_number: {result.metadata["page_label"]} \n File Location: {result.metadata["source"]}" for result in results ])

SYSTEM_PROMPT = f"""
You are a helpful assistant. Answer the question based on the provided context.
retrieved from pdf file along with page_content and page_number

You should only answer the user based on the following context and navigate to the user 
and open the right page number to know more 

If anyone asks you related python , can ask him I can provide you 
only the information related to the nodejs 

Exmaples:
UserQuery: what is python?
Assistant: I can only provide information related to Node.js. Please ask about Node.js instead.

Context:
{context}

"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ],
)
print("Response from the model:")
print(response.choices[0].message.content)