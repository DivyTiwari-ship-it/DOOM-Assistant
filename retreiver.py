from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

print("1. Waking up DOOM's offline memory...")

embeddings = OllamaEmbeddings(model="llama3.2")
persist_dir = "./doom_memory"


vector_db = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings
)


retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}  # Top 1 best matching chunk nikal kar layega
)

print("2. Scanning memory for answers...\n")


query = "Who is the creator of Mark I and what is he preparing for?"
results = retriever.invoke(query)

print("==== DOOM'S MEMORY MATCH ====")
if results:
    print(results[0].page_content)
else:
    print("No matching memory found.")
print("=============================\n")
