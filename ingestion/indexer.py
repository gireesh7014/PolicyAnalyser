from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from pymongo import MongoClient
import uuid
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

class Indexer:
    def __init__(self):
        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        
        # Initialize Qdrant
        if settings.QDRANT_API_KEY:
            self.qdrant = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        else:
            self.qdrant = QdrantClient(url=settings.QDRANT_URL)
            
        # Ensure Qdrant collection exists (Dimensionality for all-MiniLM-L6-v2 is 384)
        try:
            self.qdrant.get_collection(collection_name=settings.QDRANT_COLLECTION_NAME)
        except Exception:
            self.qdrant.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            
        # Initialize MongoDB
        self.mongo_client = MongoClient(settings.MONGODB_URL)
        self.db = self.mongo_client[settings.MONGODB_DATABASE]
        self.doc_collection = self.db["doc_store"]

    def index(self, chunks: List[Dict[str, Any]], document_id: str):
        """
        Indexes chunks into Qdrant for vector search and MongoDB for full text + bboxes.
        """
        if not chunks:
            return
            
        texts = [chunk["text"] for chunk in chunks]
        
        # 1. Embed texts
        vectors = self.embeddings.embed_documents(texts)
        
        # 2. Store in Qdrant & MongoDB
        qdrant_points = []
        mongo_docs = []
        
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_id = str(uuid.uuid4())
            
            # MongoDB Document
            mongo_docs.append({
                "_id": chunk_id,
                "document_id": document_id,
                "chunk_index": i,
                "text": chunk["text"],
                "bboxes": chunk["bboxes"]
            })
            
            # Qdrant Point (storing only metadata needed for retrieval)
            qdrant_points.append(
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={"document_id": document_id, "chunk_index": i}
                )
            )
            
        # Insert into DBs
        self.doc_collection.insert_many(mongo_docs)
        self.qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=qdrant_points
        )
