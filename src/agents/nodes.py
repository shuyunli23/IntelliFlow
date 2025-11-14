# src/agents/nodes.py
"""
Enhanced LangGraph nodes for the IntelliFlow agent with query rewriting and reflection
"""
import logging
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.config.settings import settings
from src.database.vector_store import PgVectorStore
from src.tools.weather import WeatherTools
from src.utils.decorators import error_handler, log_execution

logger = logging.getLogger(__name__)


class ThinkingFormatter:
    """Utility class for formatting thinking process with natural styling"""
    
    @staticmethod
    def format_thinking_process(thinking_list: List[str]) -> str:
        """
        Format thinking process list into natural, readable text
        
        Args:
            thinking_list: List of thinking process strings
            
        Returns:
            Formatted text string with natural styling
        """
        if not thinking_list:
            return ""
        
        formatted_lines = []
        
        for line in thinking_list:
            line = line.strip()
            if not line:
                continue
            
            # Detect major section headers (Step X)
            if line.startswith(("🤔", "🔄", "📚", "🌤️", "💭")) and "Step" in line:
                if formatted_lines:  # Add spacing between sections
                    formatted_lines.append("")
                formatted_lines.append(f"{line} ==> ")
                formatted_lines.append("")  # Add space after section header
            
            # Regular content lines
            else:
                formatted_lines.append(line)
        
        return "\n".join(formatted_lines)


class IntelliFlowNodes:
    """Enhanced LangGraph nodes with natural formatted thinking process"""
    
    def __init__(self):
        """Initialize the nodes"""
        self.llm = ChatOpenAI(
            model=settings.ali_model_name,
            api_key=settings.ali_api_key,
            base_url=settings.ali_base_url,
            temperature=0.1
        )
        self.vector_store = PgVectorStore()
        self.weather_tools = WeatherTools(settings.amap_api_key)
        self.thinking_formatter = ThinkingFormatter()
        
        logger.info("Enhanced IntelliFlowNodes with natural thinking format initialized successfully")
    
    @error_handler()
    @log_execution
    def analyze_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced query analysis with simplified thinking process"""
        query = state.get("query", "")
        thinking_process = []
        
        thinking_process.append("🤔 Step 1: Analyzing query intent")
        thinking_process.append(f"User query: \"{query}\"")
        
        # Enhanced query analysis using LLM
        analysis_prompt = """Analyze the following user query and determine:
1. Is this a weather-related query? (yes/no)
2. Is this a document/knowledge-based query that would benefit from RAG? (yes/no)
3. If weather query, extract the city/location name
4. What is the main intent of the query?
5. What type of information does the user need?

Query: {query}

Respond in JSON format:
{{
    "is_weather_query": true/false,
    "is_knowledge_query": true/false,
    "weather_location": "city name or null",
    "main_intent": "brief description of what user wants to know",
    "query_type": "weather/knowledge/general",
    "information_type": "factual/procedural/analytical/other"
}}"""

        try:
            messages = [HumanMessage(content=analysis_prompt.format(query=query))]
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                analysis = self._fallback_analysis(query)
                thinking_process.append("Using keyword-based fallback analysis")
                
        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}")
            analysis = self._fallback_analysis(query)
            thinking_process.append(f"Analysis error, using fallback: {str(e)}")
        
        # Log analysis results
        query_type = analysis.get("query_type", "general")
        main_intent = analysis.get("main_intent", "Unknown")
        
        thinking_process.append(f"Query type: {query_type}")
        thinking_process.append(f"Main intent: {main_intent}")

        if analysis.get("is_weather_query"):
            location = analysis.get("weather_location", "Not specified")
            thinking_process.append(f"Weather location: {location}")
        
        if analysis.get("is_knowledge_query"):
            thinking_process.append("Will search document database")
        
        state.update({
            "is_weather_query": analysis.get("is_weather_query", False),
            "is_knowledge_query": analysis.get("is_knowledge_query", True),
            "weather_location": analysis.get("weather_location"),
            "main_intent": main_intent,
            "query_type": query_type,
            "information_type": analysis.get("information_type", "factual"),
            "loop_count": 0,
            "original_query": query,
            "thinking_process": thinking_process
        })
        
        logger.info(f"Query analysis completed. Type: {query_type}")
        return state
    
    def _fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Enhanced fallback analysis with more detailed keyword matching"""
        weather_keywords = ["weather", "temperature", "rain", "sunny", "cloudy", "forecast", "天气", "温度", "下雨", "晴天"]
        is_weather = any(keyword in query.lower() for keyword in weather_keywords)
        
        # Determine information type
        if any(word in query.lower() for word in ["how", "why", "explain", "什么", "为什么", "怎么"]):
            info_type = "procedural"
        elif any(word in query.lower() for word in ["analyze", "compare", "evaluate", "分析", "比较"]):
            info_type = "analytical"
        else:
            info_type = "factual"
        
        return {
            "is_weather_query": is_weather,
            "is_knowledge_query": not is_weather,
            "weather_location": None,
            "main_intent": "weather query" if is_weather else "knowledge query",
            "query_type": "weather" if is_weather else "knowledge",
            "information_type": info_type
        }
    
    @error_handler()
    @log_execution
    def rewrite_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced query rewriting with simplified thinking process"""
        original_query = state.get("original_query", "")
        current_query = state.get("query", original_query)
        query_type = state.get("query_type", "general")
        main_intent = state.get("main_intent", "")
        information_type = state.get("information_type", "factual")
        loop_count = state.get("loop_count", 0)
        thinking_process = state.get("thinking_process", [])
        
        thinking_process.append("🔄 Step 2: Optimizing query")
        
        if loop_count > 0:
            thinking_process.append(f"Attempt {loop_count + 1}, improving based on feedback")
        
        # Different rewriting strategies based on query type
        if query_type == "weather":
            rewrite_prompt = """Rewrite the following weather query to be more specific and clear for weather API:
- Extract clear location/city name
- Make the weather request more specific
- Keep the original intent

Original query: {query}
Main intent: {intent}

Rewritten query (focus on location and weather type):"""
        
        elif query_type == "knowledge":
            rewrite_prompt = """Please rewrite the following query to make it clearer and more natural for document search.
Keep the meaning unchanged. The goal is to make it slightly smoother and easier to match semantically similar documents.

Original query: {query}
Main intent: {intent}

Refined search query:"""
        
        else:
            rewrite_prompt = """Please lightly rewrite the following query to make it clearer, more natural, and easier to understand.
Keep the meaning and tone unchanged, just make the phrasing smoother and more fluent.

Original query: {query}
Main intent: {intent}

Refined query (clear and natural):"""
        
        try:
            messages = [HumanMessage(content=rewrite_prompt.format(
                query=current_query, 
                intent=main_intent,
                info_type=information_type,
                loop_count=loop_count
            ))]
            response = self.llm.invoke(messages)
            rewritten_query = response.content.strip()
            
            # Clean up the response
            if ":" in rewritten_query:
                rewritten_query = rewritten_query.split(":", 1)[-1].strip()
            
            thinking_process.append(f"Original: \"{current_query}\"")
            thinking_process.append(f"Optimized: \"{rewritten_query}\"")
            
            state.update({"rewritten_query": rewritten_query, "thinking_process": thinking_process})
            logger.info(f"Query rewritten: '{current_query}' -> '{rewritten_query}'")
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {str(e)}")
            thinking_process.append(f"Query optimization failed: {str(e)}, using original")
            state.update({"rewritten_query": current_query, "thinking_process": thinking_process})
        
        return state
    
    @error_handler()
    @log_execution
    def retrieve_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced document retrieval with simplified thinking process"""
        rewritten_query = state.get("rewritten_query", state.get("query", ""))
        use_rag = state.get("use_rag", True)
        similarity_threshold = state.get("similarity_threshold", settings.default_similarity_threshold)
        thinking_process = state.get("thinking_process", [])
        
        retrieved_docs = []
        context = ""
        
        thinking_process.append("📚 Step 3: Retrieving documents")
        
        if not use_rag:
            thinking_process.append("RAG disabled, skipping document retrieval")
        elif state.get("query_type") == "weather":
            thinking_process.append("Weather query, skipping document retrieval")
        else:
            thinking_process.append(f"Search query: \"{rewritten_query}\"")
            thinking_process.append(f"Similarity threshold: {similarity_threshold}")
            
            try:
                # Use enhanced search with thinking process and fallback
                retrieval_thinking = []
                retrieved_docs = self.vector_store.similarity_search_with_fallback(
                    query=rewritten_query,
                    threshold=similarity_threshold,
                    k=settings.max_retrieved_docs,
                    thinking_process=retrieval_thinking
                )
                
                # Add only key retrieval results to main thinking process
                for line in retrieval_thinking:
                    if any(keyword in line for keyword in ["Found", "documents", "Lowering", "No documents", "Error"]):
                        thinking_process.append(line)
                
                if retrieved_docs:
                    context = self.vector_store.get_context(retrieved_docs)
                    thinking_process.append(f"Successfully retrieved {len(retrieved_docs)} relevant documents")
                else:
                    thinking_process.append("No relevant documents found, will use general knowledge")
                    
            except Exception as e:
                logger.error(f"Document retrieval failed: {str(e)}")
                thinking_process.append(f"Document retrieval failed: {str(e)}")
        
        state.update({
            "retrieved_docs": retrieved_docs,
            "context": context,
            "thinking_process": thinking_process
        })
        
        return state
    
    @error_handler()
    @log_execution
    def query_weather(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced weather query with simplified thinking process"""
        is_weather_query = state.get("is_weather_query", False)
        weather_location = state.get("weather_location")
        rewritten_query = state.get("rewritten_query", "")
        thinking_process = state.get("thinking_process", [])
        
        weather_info = ""
        
        thinking_process.append("🌤️ Step 3: Querying weather information")
        
        if not is_weather_query:
            thinking_process.append("Not a weather query, skipping weather API")
        else:
            # Try to extract location from rewritten query if not found in analysis
            if not weather_location and rewritten_query:
                import re
                location_patterns = [
                    r"(?:in|for|of|at)\s+([A-Za-z\u4e00-\u9fff\s]+)",
                    r"([A-Za-z\u4e00-\u9fff]+)(?:\s+weather|\s+temperature|\s+forecast)"
                ]
                
                for pattern in location_patterns:
                    match = re.search(pattern, rewritten_query, re.IGNORECASE)
                    if match:
                        weather_location = match.group(1).strip()
                        thinking_process.append(f"Extracted location from query: {weather_location}")
                        break
                
                if not weather_location:
                    thinking_process.append("Could not extract location from query")
            
            if weather_location:
                thinking_process.append(f"Querying weather for {weather_location}")
                
                try:
                    weather_info = self.weather_tools.query_weather(weather_location)
                    thinking_process.append("Weather information retrieved successfully")
                    logger.info(f"Weather information retrieved for {weather_location}")
                except Exception as e:
                    logger.error(f"Weather query failed: {str(e)}")
                    weather_info = f"Sorry, I couldn't retrieve weather information for {weather_location}."
                    thinking_process.append(f"Weather API query failed: {str(e)}")
            else:
                weather_info = "Please specify a location for the weather query."
                thinking_process.append("Missing location information, cannot query weather")
        
        state.update({
            "weather_info": weather_info,
            "weather_location": weather_location,
            "thinking_process": thinking_process
        })
        
        return state
    
    @error_handler()
    @log_execution
    def generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced response generation with simplified thinking process"""
        original_query = state.get("original_query", "")
        context = state.get("context", "")
        weather_info = state.get("weather_info", "")
        query_type = state.get("query_type", "general")
        main_intent = state.get("main_intent", "")
        loop_count = state.get("loop_count", 0)
        thinking_process = state.get("thinking_process", [])
        
        thinking_process.append("💭 Step 4: Generating response")
        
        # Analyze available information sources
        sources = []
        if context:
            sources.append("document knowledge")
        if weather_info:
            sources.append("weather data")
        if not sources:
            sources.append("general knowledge")
        
        thinking_process.append(f"Information sources: {', '.join(sources)}")
        
        # Enhanced system prompt
        system_prompt = """You are IntelliFlow, an intelligent assistant that provides accurate and helpful responses.

IMPORTANT RULES:
- Always answer the user's ORIGINAL question, not any rewritten version
- Prioritize the most relevant and recent information
- For knowledge queries: Use document content when available, clearly cite sources
- For weather queries: Use provided weather information comprehensively
- Be comprehensive but concise
- Include specific details and examples when relevant
- If information is insufficient, clearly state limitations

RESPONSE QUALITY:
- Directly address the user's question
- Provide actionable information when possible
- Maintain conversational tone
- Structure information clearly
- If uncertain about anything, acknowledge it"""
        
        # Prepare user message with comprehensive information
        if query_type == "weather" and weather_info:
            user_prompt = f"""Weather Information Available:
{weather_info}

Original User Question: {original_query}

Please provide a comprehensive weather response based on the information above. Include all relevant details like temperature, conditions, and any forecasts available."""
        
        elif query_type == "knowledge" and context:
            user_prompt = f"""Retrieved Document Content:
{context}

Original User Question: {original_query}

Please answer based on the retrieved content. Cite specific information from the documents and indicate if additional information might be helpful."""
        
        else:
            user_prompt = f"""Original User Question: {original_query}

Please provide an accurate and helpful answer using your knowledge. Be comprehensive and include relevant details."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_content = response.content.strip()
            
            # Simple response analysis
            word_count = len(response_content.split())
            thinking_process.append(f"Response generated successfully, length: {word_count} words")
            
            # Format thinking process with natural styling
            formatted_thinking = self.thinking_formatter.format_thinking_process(thinking_process)
            
            state.update({
                "response": response_content,
                "response_generated": True,
                "thinking_process": thinking_process,
                "formatted_thinking": formatted_thinking
            })
            
            logger.info("Enhanced response generated successfully")
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            thinking_process.append(f"Response generation failed: {str(e)}")
            
            formatted_thinking = self.thinking_formatter.format_thinking_process(thinking_process)
            
            state.update({
                "response": "I apologize, but I encountered an error while processing your request. Please try rephrasing your question.",
                "response_generated": True,
                "thinking_process": thinking_process,
                "formatted_thinking": formatted_thinking
            })
        
        return state
    
    @error_handler()
    @log_execution
    def reflect_on_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced reflection with simplified thinking process"""
        original_query = state.get("original_query", "")
        response = state.get("response", "")
        context = state.get("context", "")
        weather_info = state.get("weather_info", "")
        query_type = state.get("query_type", "general")
        loop_count = state.get("loop_count", 0)
        thinking_process = state.get("thinking_process", [])
        
        thinking_process.append("🤔 Step 5: Evaluating response quality")
        thinking_process.append(f"Current attempt: {loop_count + 1}/3")
        
        reflection_prompt = f"""Evaluate if the following response adequately answers the user's question:

Original Question: {original_query}
Query Type: {query_type}
Generated Response: {response}

Available Information Sources:
- Document Context: {"Yes (" + str(len(context.split())) + " words)" if context else "No"}
- Weather Info: {"Yes" if weather_info else "No"}

Evaluation Criteria:
1. Does the response directly answer the user's question?
2. Is the response specific and informative enough?
3. Are the most relevant information sources properly utilized?
4. For knowledge queries: Is document context used when available?
5. For weather queries: Is weather information properly included?
6. Are there any obvious gaps, inaccuracies, or areas for improvement?
7. Is the response length appropriate for the question complexity?

Respond with JSON:
{{
    "adequately_answered": true/false,
    "used_best_sources": true/false,
    "appropriate_detail": true/false,
    "specific_issues": ["list of specific problems if any"],
    "improvement_suggestions": ["what could be improved"],
    "overall_quality": "excellent/good/fair/poor",
    "needs_retry": true/false
}}"""
        
        try:
            messages = [HumanMessage(content=reflection_prompt)]
            reflection_response = self.llm.invoke(messages)
            
            import json
            import re
            json_match = re.search(r'\{.*\}', reflection_response.content, re.DOTALL)
            
            if json_match:
                reflection = json.loads(json_match.group())
                
                adequately_answered = reflection.get("adequately_answered", True)
                used_best_sources = reflection.get("used_best_sources", True)
                appropriate_detail = reflection.get("appropriate_detail", True)
                overall_quality = reflection.get("overall_quality", "good")
                
                thinking_process.append(f"Adequately answered: {'Yes' if adequately_answered else 'No'}")
                thinking_process.append(f"Used best sources: {'Yes' if used_best_sources else 'No'}")
                thinking_process.append(f"Appropriate detail: {'Yes' if appropriate_detail else 'No'}")
                thinking_process.append(f"Overall quality: {overall_quality}")
                
                needs_improvement = not (adequately_answered and used_best_sources and appropriate_detail)
                
                if reflection.get("specific_issues"):
                    thinking_process.append("Issues identified:")
                    for issue in reflection["specific_issues"][:2]:  # Only show top 2 issues
                        thinking_process.append(f"- {issue}")
                
            else:
                needs_improvement = self._enhanced_quality_check(original_query, response, context, weather_info)
                reflection = {"needs_retry": needs_improvement, "overall_quality": "unknown"}
                thinking_process.append("Using heuristic quality check")
                thinking_process.append(f"Needs improvement: {'Yes' if needs_improvement else 'No'}")
            
            # Final decision
            if needs_improvement and loop_count < 2:
                thinking_process.append("Response needs improvement, preparing retry")
            else:
                thinking_process.append("Response quality meets standards" if loop_count < 2 else "Maximum attempts reached")
            
            # Format thinking process
            formatted_thinking = self.thinking_formatter.format_thinking_process(thinking_process)
            
            state.update({
                "needs_improvement": needs_improvement,
                "reflection": reflection,
                "reflection_complete": True,
                "thinking_process": thinking_process,
                "formatted_thinking": formatted_thinking
            })
            
            logger.info(f"Enhanced reflection completed. Needs improvement: {needs_improvement}")
            
        except Exception as e:
            logger.error(f"Reflection failed: {str(e)}")
            thinking_process.append(f"Evaluation failed: {str(e)}, accepting current response")
            
            formatted_thinking = self.thinking_formatter.format_thinking_process(thinking_process)
            
            state.update({
                "needs_improvement": False,
                "reflection_complete": True,
                "thinking_process": thinking_process,
                "formatted_thinking": formatted_thinking
            })
        
        return state
    
    def _enhanced_quality_check(self, query: str, response: str, context: str, weather_info: str) -> bool:
        """Enhanced heuristic quality check"""
        # Basic quality checks
        if len(response.split()) < 15:
            return True
        
        # Check for error indicators
        error_indicators = ["sorry", "error", "couldn't", "unable", "don't know", "apologize"]
        if any(indicator in response.lower() for indicator in error_indicators):
            return True
        
        # Check if available information sources were used
        if context and len(context) > 100:
            context_words = set(context.lower().split()[:50])  # First 50 words
            response_words = set(response.lower().split())
            overlap = len(context_words.intersection(response_words))
            if overlap < 5:  # Very little overlap suggests context wasn't used
                return True
        
        if weather_info and len(weather_info) > 50:
            weather_words = set(weather_info.lower().split()[:30])  # First 30 words
            response_words = set(response.lower().split())
            overlap = len(weather_words.intersection(response_words))
            if overlap < 3:  # Weather info not used
                return True
        
        return False
    
    @error_handler()
    @log_execution
    def prepare_retry(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced retry preparation with simplified thinking process"""
        loop_count = state.get("loop_count", 0)
        original_query = state.get("original_query", "")
        reflection = state.get("reflection", {})
        thinking_process = state.get("thinking_process", [])
        
        thinking_process.append(f"🔄 Step 6: Preparing attempt {loop_count + 2}")
        
        # Increment loop count
        new_loop_count = loop_count + 1
        
        # Create a more targeted query for retry based on reflection
        improvement_suggestions = reflection.get("improvement_suggestions", [])
        
        if improvement_suggestions:
            retry_query = f"{original_query} (Focus on: {', '.join(improvement_suggestions[:2])})"
            thinking_process.append(f"Optimizing query focus based on feedback: {', '.join(improvement_suggestions[:2])}")
        else:
            retry_query = original_query
            thinking_process.append("Query remains unchanged")
        
        thinking_process.append("Clearing previous results, restarting process")
        
        state.update({
            "loop_count": new_loop_count,
            "query": retry_query,
            "response_generated": False,
            "needs_improvement": False,
            "reflection_complete": False,
            # Clear previous results
            "rewritten_query": "",
            "context": "",
            "retrieved_docs": [],
            "weather_info": "",
            "response": "",
            "thinking_process": thinking_process
        })
        
        logger.info(f"Preparing retry attempt {new_loop_count + 1}")
        return state
    
    @error_handler()
    @log_execution
    def generate_fallback_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate fallback response using general knowledge when all retries fail
        
        Args:
            state: Current state
            
        Returns:
            Updated state with fallback response
        """
        original_query = state.get("original_query", "")
        failed_response = state.get("response", "")
        query_type = state.get("query_type", "general")
        main_intent = state.get("main_intent", "")
        thinking_process = state.get("thinking_process", [])
        
        thinking_process.append("🔄 Step 6: Generating fallback response using general knowledge")
        thinking_process.append("Previous attempts were unsatisfactory, switching to general knowledge mode")
        
        # Enhanced system prompt for fallback mode
        fallback_system_prompt = """You are IntelliFlow in fallback mode. The system has attempted to answer using retrieved documents but the quality was insufficient after multiple retries.

    FALLBACK MODE RULES:
    - Use your general knowledge to provide a comprehensive answer
    - Be transparent that this response is based on general knowledge, not retrieved documents
    - Provide accurate, helpful information even without specific document context
    - Structure your response clearly and include relevant details
    - If the topic requires very specific or recent information, acknowledge limitations

    Your goal is to be helpful while being honest about the information source."""

        # Prepare fallback prompt
        fallback_prompt = f"""The system attempted to answer this question using document retrieval but was unable to provide a satisfactory response after multiple attempts.

Original Question: {original_query}
Query Type: {query_type}
Main Intent: {main_intent}

Previous Response (unsatisfactory): {failed_response}

Please provide a comprehensive answer using general knowledge. Be clear that this response is based on general knowledge rather than specific retrieved documents, and provide the best possible answer you can."""

        try:
            messages = [
                SystemMessage(content=fallback_system_prompt),
                HumanMessage(content=fallback_prompt)
            ]
            
            response = self.llm.invoke(messages)
            fallback_content = response.content.strip()
            
            # Combine failed response with fallback response
            combined_response = f"""**Based on Retrieved Documents:**
{failed_response}

**Using General Knowledge:**
{fallback_content}"""
            logger.info(combined_response)
            thinking_process.append(f"Fallback response generated successfully, length: {len(fallback_content)} words")
            thinking_process.append("Combined original response with general knowledge response")
            
            # Format thinking process
            formatted_thinking = self.thinking_formatter.format_thinking_process(thinking_process)
            
            state.update({
                "response": combined_response,
                "response_generated": True,
                "thinking_process": thinking_process,
                "formatted_thinking": formatted_thinking
            })
            
            logger.info("Fallback response generated successfully")
            
        except Exception as e:
            logger.error(f"Fallback response generation failed: {str(e)}")
            thinking_process.append(f"Fallback response generation failed: {str(e)}")
            
            # If even fallback fails, keep the original response
            formatted_thinking = self.thinking_formatter.format_thinking_process(thinking_process)
            
            state.update({
                "response": failed_response,
                "response_generated": True,
                "thinking_process": thinking_process,
                "formatted_thinking": formatted_thinking
            })
        
        return state