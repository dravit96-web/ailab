from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma   
from langchain.embeddings import HuggingFaceBgeEmbeddings

#stage 1: Load the PDF 
def load_the_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print("PDF loaded successfully.")
    return documents

#stage 2: Split the text into chunks
def split_the_documents(documents, chunk_size=400, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documents split into {len(chunks)} chunks.")
    return chunks

#stage 3: Create embeddings
def create_embeddings():
    embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model created.")
    return embedding_model

#stage 4: create vector database  
def create_vector_database(chunks, embedding_model, db_path):
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path )
    vector_db.persist()
    print("Vector database created and persisted.")

if __name__ == "__main__":
    pdf_path = "sample.pdf"  # Path to your PDF file
    db_path = "vector_db"    # Directory to store the vector database

    # Load and process the PDF
    documents = load_the_pdf(pdf_path)
    chunks = split_the_documents(documents)

    # Create embeddings and vector database
    embedding_model = create_embeddings()
    create_vector_database(chunks, embedding_model, db_path)  