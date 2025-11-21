from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import httpx
import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import uvicorn

# --- Configuration & Setup (Preserving your original paths and logic) ---

tiktoken_cache_dir = r"C:\Users\GenAICHNSIRUSR111\Desktop\Aniruth\tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# Validate cache directory exists
if not os.path.exists(os.path.join(tiktoken_cache_dir, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4")):
    print(f"Warning: Cache file not found in {tiktoken_cache_dir}")

print("Initializing Langchain with GenAI Lab Models...")
client = httpx.Client(verify=False)

# Initialize Embedding Model (Done globally to avoid reloading on every request)
embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key='sk-V4pmNP__HX36T0eUIpnPdA', 
    http_client=client
)

db_path = r"C:\Users\GenAICHNSIRUSR111\Desktop\Aniruth\Vector_DB"

# Initialize Vector DB connection
vector_db = Chroma(
    persist_directory=db_path,
    embedding_function=embedding_model
)

# Initialize LLM
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-gpt-4o-mini",
    api_key="sk-V4pmNP__HX36T0eUIpnPdA",
    http_client=client
)

# Define Prompt
prompt_template = ChatPromptTemplate.from_template("""
You are an expert in PowerShell scripting. Answer the question with minimal explanation but crisp with accurate commands.
Do not answer if the question is not related to powershell.
Context:
{context}

Question:
{question}

Answer:
""")

# --- FastAPI App Structure ---

app = FastAPI(title="PowerShell RAG API")

origins = [
    # Replace the * with your frontend URL (e.g., "http://localhost:5173") 
    # for production, but using "*" is quickest for development testing.
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # Allows the origins listed above
    allow_credentials=True,         # Allows cookies/auth headers
    allow_methods=["*"],            # IMPORTANT: Allows all methods, including OPTIONS, POST, GET
    allow_headers=["*"],            # Allows all headers
)


# Request Model
class QueryRequest(BaseModel):
    question: str

# Response Model
class QueryResponse(BaseModel):
    question: str
    answer: str

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        # 1. Retrieve Documents
        retrieved_docs = vector_db.similarity_search(request.question, k=2)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 2. Format Prompt
        final_prompt = prompt_template.format(context=context, question=request.question)

        # 3. Generate Response
        response = llm.invoke(final_prompt)

        return QueryResponse(
            question=request.question,
            answer=response.content
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "active", "model": "genailab-maas-gpt-4o-mini"}

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=8000)