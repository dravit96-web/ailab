import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # <-- Use OpenAI embeddings
from langchain_openai import ChatOpenAI
import httpx

# Source - https://stackoverflow.com/a
# Posted by VarBird, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-20, License - CC BY-SA 4.0

tiktoken_cache_dir = r"Y:\Hackathon\Chennai\Siruseri\team11\Aniruth\tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# validate
assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))


print("Testing Langchain with GenAI Lab Models")
client = httpx.Client(verify=False)
# Make sure your OpenAI API key is set in the environment:
# Windows: setx OPENAI_API_KEY "your_api_key_here"
# Linux/Mac: export OPENAI_API_KEY="your_api_key_here"

def load_the_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print("PDF loaded successfully.")
    print("--------------------------------------------------------------------------------------")
    return documents

def split_the_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documents split into {len(chunks)} chunks.")
    print("--------------------------------------------------------------------------------------")
    return chunks

def create_embeddings():
    embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    #api_key="sk-aHHp5CYA-dVpH3M71ek1dw",  
    api_key='sk-V4pmNP__HX36T0eUIpnPdA', # Replace with your actual API key
    http_client=client
)

    print("Embedding Model instance created successfully")
    print("OpenAI embedding model created.")
    print("--------------------------------------------------------------------------------------")
    return embedding_model

def create_vector_database(chunks, embedding_model, db_path):
    # Chroma expects a directory, not a file ending with .db
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )
    vector_db.persist()
    print(f"Vector database created and persisted at: {db_path}")
    print("--------------------------------------------------------------------------------------")

if __name__ == "__main__":
    pdf_path = r"Y:\Hackathon\Chennai\Siruseri\team11\Aniruth\powershell_scripting.pdf"
    db_path = r"Y:\Hackathon\Chennai\Siruseri\team11\Aniruth\Vector_DB"

    documents = load_the_pdf(pdf_path)
    chunks = split_the_documents(documents)

    embedding_model = create_embeddings()
    create_vector_database(chunks, embedding_model, db_path)
