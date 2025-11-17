# src/agents/intelliflow_agent_v1.py
"""
IntelliFlow Agent using LangChain 1.0 with create_agent and custom middleware
"""
import logging
import uuid
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import NotRequired
from src.config.settings import settings
from src.database.vector_store import PgVectorStore
from src.tools.weather import WeatherTools

logger = logging.getLogger(__name__)


# ==================== Custom State Schema ====================
class IntelliFlowState(AgentState):
    """Extended state schema for IntelliFlow agent"""
    # Original query tracking
    original_query: NotRequired[str]
    
    # Query analysis results
    is_weather_query: NotRequired[bool]
    is_knowledge_query: NotRequired[bool]
    weather_location: NotRequired[Optional[str]]
    query_type: NotRequired[str]
    main_intent: NotRequired[str]
    
    # Rewriting and retrieval
    rewritten_query: NotRequired[str]
    retrieved_docs: NotRequired[List[Any]]
    context: NotRequired[str]
    weather_info: NotRequired[str]
    
    # Reflection and retry
    loop_count: NotRequired[int]
    needs_improvement: NotRequired[bool]
    reflection_result: NotRequired[Dict[str, Any]]
    refinement_suggestions: NotRequired[List[str]]
    
    # Thinking process
    thinking_process: NotRequired[List[str]]
    formatted_thinking: NotRequired[str]


# ==================== Tools Definition ====================
class IntelliFlowTools:
    """Tools for IntelliFlow agent"""
    
    def __init__(self):
        self.vector_store = PgVectorStore()
        self.weather_tools = WeatherTools(settings.amap_api_key)
    
    def get_tools(self):
        """Get all tools for the agent"""
        
        @tool
        def search_documents(query: str, threshold: float = 0.7) -> str:
            """
            Search relevant documents from the knowledge base.
            
            Args:
                query: The search query
                threshold: Similarity threshold (default: 0.7)
            
            Returns:
                Retrieved document content as context
            """
            try:
                docs = self.vector_store.similarity_search_with_fallback(
                    query=query,
                    threshold=threshold,
                    k=settings.max_retrieved_docs
                )
                if docs:
                    context = self.vector_store.get_context(docs)
                    return f"Found {len(docs)} relevant documents:\n\n{context}"
                return "No relevant documents found in the knowledge base."
            except Exception as e:
                logger.error(f"Document search failed: {str(e)}")
                return f"Error searching documents: {str(e)}"
        
        @tool
        def get_weather(location: str) -> str:
            """
            Get current weather information for a specific location.
            
            Args:
                location: City or location name
            
            Returns:
                Weather information
            """
            try:
                weather_info = self.weather_tools.query_weather(location)
                return weather_info
            except Exception as e:
                logger.error(f"Weather query failed: {str(e)}")
                return f"Unable to retrieve weather for {location}: {str(e)}"
        
        return [search_documents, get_weather]


# ==================== Custom Middleware ====================

class QueryAnalysisMiddleware(AgentMiddleware[IntelliFlowState]):
    """Middleware for analyzing and understanding query intent"""
    
    state_schema = IntelliFlowState
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def before_model(self, state: IntelliFlowState, runtime) -> Dict[str, Any] | None:
        """Analyze query before the first model call"""
        messages = state.get("messages", [])
        thinking = state.get("thinking_process", [])
        
        # Only run on the first message (initial query)
        if len(messages) != 1 or state.get("original_query"):
            return None
        
        query = messages[0].content
        thinking.append("🤔 Step 1: Analyzing query intent")
        thinking.append(f"User query: \"{query}\"")
        
        # Use LLM to analyze query
        analysis_prompt = f"""Analyze this query and return ONLY valid JSON:
{{
    "is_weather_query": true/false,
    "is_knowledge_query": true/false,
    "weather_location": "city name or null",
    "main_intent": "brief description",
    "query_type": "weather/knowledge/general"
}}

Query: {query}"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            import json, re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                # Fallback analysis
                weather_keywords = ["weather", "temperature", "rain", "天气", "温度"]
                is_weather = any(kw in query.lower() for kw in weather_keywords)
                analysis = {
                    "is_weather_query": is_weather,
                    "is_knowledge_query": not is_weather,
                    "weather_location": None,
                    "main_intent": "weather" if is_weather else "knowledge query",
                    "query_type": "weather" if is_weather else "knowledge"
                }
            
            thinking.append(f"Query type: {analysis['query_type']}")
            thinking.append(f"Main intent: {analysis['main_intent']}")
            
            return {
                "original_query": query,
                "is_weather_query": analysis.get("is_weather_query", False),
                "is_knowledge_query": analysis.get("is_knowledge_query", True),
                "weather_location": analysis.get("weather_location"),
                "main_intent": analysis.get("main_intent", ""),
                "query_type": analysis.get("query_type", "general"),
                "loop_count": 0,
                "thinking_process": thinking
            }
            
        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}")
            thinking.append(f"Analysis error: {str(e)}")
            return {"thinking_process": thinking}


class QueryRewritingMiddleware(AgentMiddleware[IntelliFlowState]):
    """Middleware for rewriting queries to improve retrieval"""
    
    state_schema = IntelliFlowState
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def before_model(self, state: IntelliFlowState, runtime) -> Dict[str, Any] | None:
        """Rewrite query before model call if needed"""
        loop_count = state.get("loop_count", 0)
        original_query = state.get("original_query")
        query_type = state.get("query_type", "general")
        thinking = state.get("thinking_process", [])
        
        # Skip if already rewritten in this loop or no original query
        if state.get("rewritten_query") or not original_query:
            return None
        
        thinking.append("🔄 Step 2: Optimizing query for better results")
        if loop_count > 0:
            thinking.append(f"Attempt {loop_count + 1}, refining based on feedback")
        
        # Different strategies based on query type
        if query_type == "weather":
            prompt = f"Extract location and weather type clearly: {original_query}"
        else:
            prompt = f"Rephrase for semantic search (keep meaning): {original_query}"
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            rewritten = response.content.strip()
            if ":" in rewritten:
                rewritten = rewritten.split(":", 1)[-1].strip()
            
            thinking.append(f"Original: \"{original_query}\"")
            thinking.append(f"Optimized: \"{rewritten}\"")
            
            return {
                "rewritten_query": rewritten,
                "thinking_process": thinking
            }
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {str(e)}")
            thinking.append(f"Optimization failed, using original")
            return {
                "rewritten_query": original_query,
                "thinking_process": thinking
            }


class ResponseReflectionMiddleware(AgentMiddleware[IntelliFlowState]):
    """Middleware for reflecting on response quality and deciding retry"""
    
    state_schema = IntelliFlowState
    
    def __init__(self, llm: ChatOpenAI, max_loops: int = 2):
        self.llm = llm
        self.max_loops = max_loops
    
    def after_model(self, state: IntelliFlowState, runtime) -> Dict[str, Any] | None:
        """Reflect on response quality after generation"""
        messages = state.get("messages", [])
        loop_count = state.get("loop_count", 0)
        original_query = state.get("original_query")
        thinking = state.get("thinking_process", [])
        
        # Only reflect if we have a response (AIMessage without tool calls)
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        
        last_msg = messages[-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return None  # Still calling tools
        
        thinking.append(f"🤔 Step 5: Evaluating response (attempt {loop_count + 1}/{self.max_loops + 1})")
        
        response_text = last_msg.content
        
        # Quick quality check
        needs_retry = self._evaluate_quality(original_query, response_text, state)
        
        thinking.append(f"Quality assessment: {'Needs improvement' if needs_retry else 'Acceptable'}")
        
        if needs_retry and loop_count < self.max_loops:
            thinking.append("Preparing for retry with refined approach")
            
            # Clear state for retry
            return {
                "loop_count": loop_count + 1,
                "rewritten_query": "",  # Force rewrite
                "needs_improvement": True,
                "thinking_process": thinking,
                "jump_to": "agent"  # Jump back to agent node for retry
            }
        else:
            if loop_count >= self.max_loops:
                thinking.append("Maximum attempts reached, accepting response")
            
            formatted = self._format_thinking(thinking)
            return {
                "thinking_process": thinking,
                "formatted_thinking": formatted,
                "needs_improvement": False
            }
    
    def _evaluate_quality(self, query: str, response: str, state: IntelliFlowState) -> bool:
        """Evaluate if response needs improvement"""
        # Heuristic checks
        if len(response.split()) < 15:
            return True
        
        error_indicators = ["sorry", "error", "couldn't", "unable"]
        if any(ind in response.lower() for ind in error_indicators):
            return True
        
        # Check if tools were used when they should have been
        query_type = state.get("query_type")
        messages = state.get("messages", [])
        
        tool_used = any(
            hasattr(msg, 'tool_calls') and msg.tool_calls 
            for msg in messages if isinstance(msg, AIMessage)
        )
        
        if query_type == "weather" and not tool_used:
            return True
        
        if query_type == "knowledge" and not tool_used:
            return True
        
        return False
    
    def _format_thinking(self, thinking_list: List[str]) -> str:
        """Format thinking process naturally"""
        formatted = []
        for line in thinking_list:
            if line.startswith(("🤔", "🔄", "📚", "🌤️", "💭")) and "Step" in line:
                if formatted:
                    formatted.append("")
                formatted.append(f"{line} ==> ")
            else:
                formatted.append(line)
        return "\n".join(formatted)


class ThinkingLoggerMiddleware(AgentMiddleware[IntelliFlowState]):
    """Middleware for logging thinking process at each step"""
    
    state_schema = IntelliFlowState
    
    def after_model(self, state: IntelliFlowState, runtime) -> Dict[str, Any] | None:
        """Log after each model call"""
        thinking = state.get("thinking_process", [])
        messages = state.get("messages", [])
        
        if messages and isinstance(messages[-1], AIMessage):
            last_msg = messages[-1]
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                tools = [tc['name'] for tc in last_msg.tool_calls]
                thinking.append(f"💭 Calling tools: {', '.join(tools)}")
                return {"thinking_process": thinking}
        
        return None


# ==================== Main Agent Factory ====================

def create_intelliflow_agent(
    use_rag: bool = True,
    similarity_threshold: float = 0.7,
    max_retry_loops: int = 2
):
    """
    Create IntelliFlow agent using LangChain 1.0 create_agent with middleware
    
    Args:
        use_rag: Enable RAG functionality
        similarity_threshold: Threshold for document similarity
        max_retry_loops: Maximum number of retry attempts
    
    Returns:
        Compiled agent graph
    """
    # Initialize LLM
    llm = ChatOpenAI(
        model=settings.ali_model_name,
        api_key=settings.ali_api_key,
        base_url=settings.ali_base_url,
        temperature=0.1
    )
    
    # Initialize tools
    tool_manager = IntelliFlowTools()
    tools = tool_manager.get_tools()
    
    # System prompt
    system_prompt = """You are IntelliFlow, an intelligent assistant providing accurate and helpful responses.

INSTRUCTIONS:
1. First, ALWAYS try to use your tools to answer the question.
   - For weather queries, use the get_weather tool.
   - For knowledge queries, use the search_documents tool.

2. **IMPORTANT**: If the `search_documents` tool returns "No relevant documents found", then and ONLY then, you should use your own general knowledge to answer the question.

3. When using your general knowledge, clearly state that the information is from your training data and not from the provided documents.

4. Provide comprehensive, well-structured answers.
5. If you use documents, cite your sources.

Be direct, accurate, and helpful."""
    
    # Create agent with middleware
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            QueryAnalysisMiddleware(llm),          # Analyze query intent
            QueryRewritingMiddleware(llm),         # Rewrite for better results
            ThinkingLoggerMiddleware(),            # Log tool usage
            ResponseReflectionMiddleware(          # Reflect and retry if needed
                llm=llm, 
                max_loops=max_retry_loops
            ),
        ],
        checkpointer=MemorySaver(),  # Enable conversation memory
        state_schema=IntelliFlowState
    )
    
    logger.info("IntelliFlow agent created with LangChain 1.0 middleware")
    return agent


# ==================== Usage Example ====================

def run_intelliflow(query: str, thread_id: Optional[str] = None, **kwargs):
    """
    Run IntelliFlow agent with a query for a specific conversation thread.
    
    Args:
        query: User query
        thread_id: A unique identifier for the conversation. If None, a new one is created.
        **kwargs: Additional configuration
    
    Returns:
        Agent response with thinking process
    """
    agent = create_intelliflow_agent(**kwargs)
    
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    logger.info(f"Running query on thread_id: {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}

    # Run agent
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)], "thinking_process": []},
        config=config
    )

    # Extract results
    response = result["messages"][-1].content if result.get("messages") else ""
    thinking = result.get("formatted_thinking", "")
    
    return {
        "query": query,
        "response": response,
        "thinking_process": thinking,
        "loop_count": result.get("loop_count", 0),
        "thread_id": thread_id,
        "metadata": {
            "query_type": result.get("query_type"),
            "tools_used": result.get("needs_improvement") is False
        }
    }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Weather query
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Weather Query")
    print("=" * 60)
    result = run_intelliflow("What's the weather like in Beijing?")
    print(f"\nResponse: {result['response']}")
    print(f"\nThinking:\n{result['thinking_process']}")
    
    # Example 2: Knowledge query
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Knowledge Query")
    print("=" * 60)
    result = run_intelliflow("Tell me about climate change impacts")
    print(f"\nResponse: {result['response']}")
    print(f"\nThinking:\n{result['thinking_process']}")
    print(f"\nLoops: {result['loop_count']}")

