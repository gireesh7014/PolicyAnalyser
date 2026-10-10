from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import settings

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: Optional[str]
    documents: List[Dict[str, Any]]
    retries: int

def get_llm():
    if settings.GROQ_API_KEY:
        return ChatGroq(temperature=0, model_name=settings.LLM_MODEL, groq_api_key=settings.GROQ_API_KEY)
    elif settings.GEMINI_API_KEY:
        return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=settings.GEMINI_API_KEY)
    raise ValueError("Neither GROQ_API_KEY nor GEMINI_API_KEY is set in environment.")

# --- Nodes ---
def retrieve(state: GraphState):
    """
    Retrieve documents from Vector Store & BM25
    """
    question = state["question"]
    # In a real setup, we would call our Hybrid Retriever here
    # For now, this is a placeholder for the actual retriever call
    documents = [{"text": "Sample retrieved contract text...", "bboxes": []}]
    return {"documents": documents, "question": question, "retries": state.get("retries", 0)}

def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    question = state["question"]
    documents = state["documents"]
    
    llm = get_llm()
    # Structured output for grading
    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
        
    structured_llm_grader = llm.with_structured_output(Grade)
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")])
    
    retrieval_grader = grade_prompt | structured_llm_grader
    
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d["text"]})
        if score.binary_score == "yes":
            filtered_docs.append(d)
            
    return {"documents": filtered_docs, "question": question, "retries": state.get("retries", 0)}

def generate(state: GraphState):
    """
    Generate answer
    """
    question = state["question"]
    documents = state["documents"]
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "You are a legal assistant. Answer the question based only on the provided context.\n\n"
        "Context: {context}\n\nQuestion: {question}"
    )
    
    context = "\n\n".join([doc["text"] for doc in documents])
    chain = prompt | llm
    generation = chain.invoke({"context": context, "question": question})
    
    return {"documents": documents, "question": question, "generation": generation.content}

def rewrite_query(state: GraphState):
    """
    Transform the query to produce a better question.
    """
    question = state["question"]
    llm = get_llm()
    msg = [
        ("system", "You are generating a better search query for a legal RAG system. Look at the input and try to reason about the underlying intent."),
        ("human", f"Initial question: {question}")
    ]
    response = llm.invoke(msg)
    return {"documents": state["documents"], "question": response.content, "retries": state.get("retries", 0) + 1}

# --- Edges ---
def decide_to_generate(state: GraphState):
    """
    Determines whether to generate an answer, or re-generate a question.
    """
    filtered_documents = state["documents"]
    retries = state.get("retries", 0)
    
    if not filtered_documents:
        if retries < 2:
            return "rewrite_query"
        else:
            return "generate" # Just generate anyway or fail gracefully
    else:
        return "generate"

# --- Build Graph ---
def build_agent_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite_query", rewrite_query)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        }
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)
    
    app = workflow.compile()
    return app
