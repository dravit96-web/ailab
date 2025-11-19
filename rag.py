from langchain.chains import RetrievalQA
from langchain.llms import ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate

#load the vector database

db_path = "./vector_db/guidness.db" # Path to your Chroma vector database
embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
retriever = vector_db.as_retriever()

#define the prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""you are a helpful AI assistant. Use the following context to answer the question.
     If you don't know the answer, just say that you don't know, don't try to make up an answer.
     Context: {context}
     question: {question}
     Answer:''""",
)

# Initialize the LLM
llm = ollama.Ollama(model="gpt-oss:120b-cloud", temperature=0)

#create retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,    
    chain_type_kwargs={"prompt": prompt}
)
print("RAG chatbot is ready to answer your questions!")
while True:
    user_question = input("Please enter your question (or type 'exit' to quit): ")
    if user_question.lower() == 'exit':
        print("Exiting the RAG chatbot. Goodbye!")
        break
    response = qa_chain.run(user_question)
    print(response)