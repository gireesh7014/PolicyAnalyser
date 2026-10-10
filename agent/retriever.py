from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

class Retriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        
        if settings.QDRANT_API_KEY:
            self.qdrant = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        else:
            self.qdrant = QdrantClient(url=settings.QDRANT_URL)
            
        self.mongo_client = MongoClient(settings.MONGODB_URL)
        self.db = self.mongo_client[settings.MONGODB_DATABASE]
        self.doc_collection = self.db["doc_store"]

    def retrieve(self, query: str, document_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embeds the query, searches Qdrant for matching chunk IDs for this document,
        then fetches the full text and bboxes from MongoDB.
        """
        query_vector = self.embeddings.embed_query(query)
        
        # Filter by document_id
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )
        
        search_result = self.qdrant.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k
        )
        
        chunk_ids = [hit.id for hit in search_result]
        
        if not chunk_ids:
            return []
            
        # Fetch full data from MongoDB
        mongo_docs = list(self.doc_collection.find({"_id": {"$in": chunk_ids}}))
        
        # Format the output
        retrieved_documents = []
        for doc in mongo_docs:
            retrieved_documents.append({
                "text": doc.get("text", ""),
                "bboxes": doc.get("bboxes", [])
            })
            
        return retrieved_documents
