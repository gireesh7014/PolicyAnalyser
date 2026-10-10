import os
import base64
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional

from config import settings
from ingestion.parser import PDFParser
from agent.graph import build_agent_graph

app = FastAPI(title="LexReviewer 2.0 API")

class DocumentUploadRequest(BaseModel):
    file: str  # Base64 encoded PDF

class AskQuestionRequest(BaseModel):
    question: str

@app.post("/upload-documents")
async def upload_documents(
    request: DocumentUploadRequest,
    document_id: str = Header(..., alias="document-id")
):
    try:
        parser = PDFParser()
        chunks = parser.parse_base64_pdf(request.file)
        
        # Here we would initialize indexing to VectorDB and DocStore
        # e.g., indexer.index(chunks, document_id)
        
        return {"status": "success", "chunks_processed": len(chunks), "message": f"Document {document_id} parsed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(
    request: AskQuestionRequest,
    document_id: str = Header(..., alias="document-id")
):
    try:
        agent_app = build_agent_graph()
        # Initialize graph state
        inputs = {"question": request.question, "retries": 0}
        
        # In a real app we'd pass document_id to the retriever via configurable thread/state
        result = agent_app.invoke(inputs)
        
        # We simulate the NDJSON streaming response pattern
        return {
            "answer": result["generation"],
            "reference_positions": [doc["bboxes"] for doc in result["documents"]] if result.get("documents") else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "llm_model": settings.LLM_MODEL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
