# src/agents/graph.py
"""
Enhanced LangGraph implementation with query rewriting and reflection
"""
import logging
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from src.agents.nodes import IntelliFlowNodes
from src.utils.decorators import error_handler, log_execution
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: The user's query.
        original_query: The initial user query, preserved through retries.
        use_rag: Flag to enable/disable RAG.
        similarity_threshold: The threshold for document retrieval.
        loop_count: The number of times the graph has looped.
        response_generated: Flag indicating if a response has been generated.
        needs_improvement: Flag from reflection indicating if a retry is needed.
        reflection_complete: Flag indicating if reflection has been performed.
        context: The retrieved context from documents.
        response: The final generated response.
        retrieved_docs: List of documents retrieved from the vector store.
        weather_info: The retrieved weather information.
        rewritten_query: The query after being rewritten by the LLM.
        query_type: The type of query (e.g., 'weather', 'knowledge', 'general').
        main_intent: The main intent of the query.
        weather_location: The location extracted for a weather query.
        thinking_process: A list of strings detailing the agent's reasoning.
        formatted_thinking: The formatted string of the thinking process.
        execution_metadata: Metadata about the final execution run.
        reflection: The dictionary containing reflection results.
        information_type: The type of information needed (factual, analytical, etc.).
"""
    query: str
    original_query: str
    use_rag: bool
    similarity_threshold: float
    loop_count: int
    response_generated: bool
    needs_improvement: bool
    reflection_complete: bool
    context: str
    response: str
    retrieved_docs: List[Document]
    weather_info: str
    rewritten_query: str

    # Analysis results from analyze_query
    is_weather_query: bool
    is_knowledge_query: bool
    weather_location: str | None
    main_intent: str
    query_type: str
    information_type: str
    
    # Process tracking
    thinking_process: List[str]
    formatted_thinking: str
    reflection: Dict[str, Any]
    execution_metadata: Dict[str, Any]


class IntelliFlowGraph:
    """Enhanced LangGraph implementation with reflection and retry logic"""
    
    def __init__(self):
        """Initialize the graph"""
        self.nodes = IntelliFlowNodes()
        self.graph = self._build_graph()
        logger.info("Enhanced IntelliFlowGraph initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the enhanced LangGraph workflow with reflection and retry
        """
        workflow = StateGraph(GraphState)
        
        # Add all nodes
        workflow.add_node("analyze_query", self.nodes.analyze_query)
        workflow.add_node("rewrite_query", self.nodes.rewrite_query)
        workflow.add_node("retrieve_documents", self.nodes.retrieve_documents)
        workflow.add_node("query_weather", self.nodes.query_weather)
        workflow.add_node("generate_response", self.nodes.generate_response)
        workflow.add_node("reflect_on_response", self.nodes.reflect_on_response)
        workflow.add_node("prepare_retry", self.nodes.prepare_retry)
        workflow.add_node("generate_fallback_response", self.nodes.generate_fallback_response)
        
        # Define the workflow
        workflow.set_entry_point("analyze_query")
        
        # Main flow
        workflow.add_edge("analyze_query", "rewrite_query")
        
        # After rewriting, decide on retrieval strategy
        workflow.add_conditional_edges(
            "rewrite_query",
            self._route_after_rewrite,
            {
                "weather": "query_weather",
                "knowledge": "retrieve_documents",
                "direct": "generate_response"
            }
        )
        
        # Both retrieval paths lead to response generation
        workflow.add_edge("retrieve_documents", "generate_response")
        workflow.add_edge("query_weather", "generate_response")
        
        # After generating response, reflect on quality
        workflow.add_edge("generate_response", "reflect_on_response")
        
        # After reflection, decide whether to retry or finish
        workflow.add_conditional_edges(
            "reflect_on_response",
            self._should_retry_or_fallback,
            {
                "retry": "prepare_retry",
                "fallback": "generate_fallback_response",
                "finish": END
            }
        )
        
        # Retry loop - go back to rewriting
        workflow.add_edge("prepare_retry", "rewrite_query")
        
        return workflow.compile()
    
    def _route_after_rewrite(self, state: Dict[str, Any]) -> str:
        """Route after query rewriting based on query type"""
        query_type = state.get("query_type", "general")
        use_rag = state.get("use_rag", True)
        
        if query_type == "weather":
            return "weather"
        elif query_type == "knowledge" and use_rag:
            return "knowledge"
        else:
            return "direct"
    
    def _should_retry_or_fallback(self, state: Dict[str, Any]) -> str:
        """Decide whether to retry, use fallback, or finish based on reflection"""
        needs_improvement = state.get("needs_improvement", False)
        loop_count = state.get("loop_count", 0)
        query_type = state.get("query_type", "general")
        
        # For weather queries, don't use fallback (weather data is specific)
        if query_type == "weather":
            if needs_improvement and loop_count < 2:
                return "retry"
            else:
                return "finish"
        
        # For knowledge and general queries
        if needs_improvement:
            if loop_count < 2:
                return "retry"
            else:
                # Max retries reached, use fallback for knowledge queries
                return "fallback"
        else:
            return "finish"
    
    @error_handler()
    @log_execution
    def run(
        self, 
        query: str, 
        use_rag: bool = True, 
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Run the enhanced IntelliFlow agent with reflection and retry
        """
        # Initialize state
        initial_state = {
            "query": query,
            "original_query": query,
            "use_rag": use_rag,
            "similarity_threshold": similarity_threshold,
            "loop_count": 0,
            "response_generated": False,
            "needs_improvement": False,
            "reflection_complete": False,
            # Results
            "context": "",
            "response": "",
            "retrieved_docs": [],
            "weather_info": "",
            "rewritten_query": "",
            # Analysis results
            "is_weather_query": False,
            "is_knowledge_query": True,
            "weather_location": None,
            "main_intent": "",
            "query_type": "general"
        }
        
        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Add execution metadata
            final_state["execution_metadata"] = {
                "total_loops": final_state.get("loop_count", 0) + 1,
                "query_rewritten": final_state.get("rewritten_query") != query,
                "reflection_performed": final_state.get("reflection_complete", False),
                "final_query_type": final_state.get("query_type", "general"),
                "fallback_used": final_state.get("fallback_used", False)
            }
            
            logger.info(f"Graph execution completed successfully after {final_state['execution_metadata']['total_loops']} attempts")
            return final_state
            
        except Exception as e:
            logger.error(f"Graph execution failed: {str(e)}")
            return {
                "query": query,
                "original_query": query,
                "response": "I apologize, but I encountered an error while processing your request.",
                "retrieved_docs": [],
                "error": str(e),
                "execution_metadata": {
                    "total_loops": 1,
                    "query_rewritten": False,
                    "reflection_performed": False,
                    "error_occurred": True
                }
            }