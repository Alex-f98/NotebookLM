import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
 
from app.ingester import load_document, split_documents
from app.vector_store import VectorStoreManager
from app.rag_chain import RAGService
from fastapi import Depends



UPLOAD_DIR= "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app = FastAPI(title="Notebook LM local")

vector_store = VectorStoreManager(persist_directory="data/db")
rag_service  = RAGService(vector_store=vector_store)


def get_rag_service() -> RAGService:
    return rag_service

class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".txt", ".pdf"):
        raise HTTPException(status_code=400, detail="Formato no soportado")
    
    tmp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    try:
        docs   = load_document(tmp_path)
        chunks = split_documents(docs)
        ids    = vector_store.add_documents(chunks)
    finally:
        os.remove(tmp_path)

    return {"filename": file.filename, "status": "processed", "chunks_created": len(ids)}


@app.post("/query")
async def query(req: QueryRequest, rag: RAGService = Depends(get_rag_service)):
    return rag.generate_answer(req.question)



