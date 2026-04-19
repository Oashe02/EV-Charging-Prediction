import os
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class AgentState(TypedDict):
    demand_summary: Dict
    guidelines_context: str
    final_report: str
    api_key: str

def init_vector_store():
    file_path = "data/guidelines.txt"
    persist_dir = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    return vector_store

def retrieve_guidelines(state: AgentState):
    print("Fetching local guidelines...")
    vector_store = init_vector_store()
    if not vector_store:
        return {"guidelines_context": "Basic guidelines say that fast chargers should be in high demand areas."}
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke("recommendations for charging station placement and scheduling")
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"guidelines_context": context}

def generate_report(state: AgentState):
    print("Generating the final report...")
    api_key = state.get("api_key", "").strip()
    
    if not api_key:
        return {"final_report": "Error: No API Key found. Please add your Gemini API key in the sidebar. \n\nExpected Output Summary:\n1. Low demand zones need basic chargers.\n2. High demand zones need fast chargers.\n3. Scheduling is needed for peak hours."}
    
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)
        
        prompt = PromptTemplate(
            input_variables=["demand_summary", "guidelines_context"],
            template="""You are an assistant for an EV Infrastructure Planning project. 
            Based on the data and guidelines below, write a report for our final submission.
            
            Data Summary:
            {demand_summary}
            
            Guidelines:
            {guidelines_context}
            
            The report should cover:
            - Overview of demand
            - Areas with high load
            - Suggestions for where to add new chargers
            - How to handle scheduling and load balancing
            - References from the guidelines
            
            Write this in clear Markdown format. Do not use emojis. Keep it professional and simple.
            """
        )
        chain = prompt | llm
        result = chain.invoke({
            "demand_summary": str(state["demand_summary"]),
            "guidelines_context": state["guidelines_context"]
        })
        return {"final_report": result.content}
    except Exception as e:
        return {"final_report": f"Error: {str(e)}"}

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve_guidelines", retrieve_guidelines)
    workflow.add_node("generate_report", generate_report)
    
    workflow.add_edge(START, "retrieve_guidelines")
    workflow.add_edge("retrieve_guidelines", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()

def run_agentic_workflow(demand_summary: Dict, api_key: str):
    app = build_graph()
    result = app.invoke({"demand_summary": demand_summary, "api_key": api_key, "guidelines_context": "", "final_report": ""})
    return result["final_report"]