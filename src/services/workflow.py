from langgraph.graph import StateGraph, END
from src.services.states import AISummaryState
from src.services.nodes import extract_ocr, validate_input, summarize_document


def build_ai_summary_graph():
    """Build and compile the LangGraph workflow for AI summary"""
    # Create a StateGraph with AISummaryState
    workflow = StateGraph(AISummaryState)
    
    # Add nodes (the processing steps)
    workflow.add_node("validate_input", validate_input)
    workflow.add_node("extract_ocr", extract_ocr)
    workflow.add_node("summarize_document", summarize_document)
    
    # Define the flow: validate_input -> extract_ocr -> summarize_document -> END
    workflow.set_entry_point("validate_input")
    workflow.add_edge("validate_input", "extract_ocr")
    workflow.add_edge("extract_ocr", "summarize_document")
    workflow.add_edge("summarize_document", END)
    
    # Compile and return the graph
    return workflow.compile()

# Create the compiled graph instances
ai_summary_graph = build_ai_summary_graph()