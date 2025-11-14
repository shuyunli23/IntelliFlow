# debug_graph.py
"""
Debug script for monitoring graph state changes
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
os.makedirs("test_outputs", exist_ok=True)

from src.agents.graph import IntelliFlowGraph
from src.database.connection import init_database, test_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphDebugger:
    """Debug graph execution with detailed state tracking"""
    
    def __init__(self):
        """Initialize debugger"""
        self.graph = IntelliFlowGraph()
        self.state_history = []
        
        # Ensure database is ready
        if not test_connection():
            logger.error("Database connection failed!")
            sys.exit(1)
        init_database()
    
    def debug_single_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Debug a single query with detailed state tracking
        
        Args:
            query: Query to debug
            **kwargs: Additional parameters
            
        Returns:
            Debug information
        """
        logger.info(f"🔍 Debugging query: {query}")
        
        # Monkey patch the graph to capture state changes
        original_invoke = self.graph.graph.invoke
        
        def debug_invoke(state):
            self.state_history.append({
                "step": "initial",
                "state": self._clean_state_for_debug(state.copy())
            })
            
            # Run original invoke
            result = original_invoke(state)
            
            self.state_history.append({
                "step": "final",
                "state": self._clean_state_for_debug(result.copy())
            })
            
            return result
        
        # Apply monkey patch
        self.graph.graph.invoke = debug_invoke
        
        try:
            # Run the query
            result = self.graph.run(query, **kwargs)
            
            # Analyze state changes
            debug_info = self._analyze_state_changes()
            debug_info["final_result"] = result
            
            return debug_info
            
        finally:
            # Restore original method
            self.graph.graph.invoke = original_invoke
    
    def _clean_state_for_debug(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Clean state for debugging (remove large objects)"""
        cleaned = {}
        
        for key, value in state.items():
            if key == "retrieved_docs":
                cleaned[key] = f"[{len(value)} documents]" if value else "[]"
            elif key == "context":
                cleaned[key] = f"[{len(value)} characters]" if value else ""
            elif key == "thinking_process":
                cleaned[key] = f"[{len(value)} steps]" if value else "[]"
            elif key == "formatted_thinking":
                cleaned[key] = f"[{len(value)} characters]" if value else ""
            elif isinstance(value, str) and len(value) > 200:
                cleaned[key] = value[:200] + "..."
            else:
                cleaned[key] = value
        
        return cleaned
    
    def _analyze_state_changes(self) -> Dict[str, Any]:
        """Analyze state changes during execution"""
        if len(self.state_history) < 2:
            return {"error": "Insufficient state history"}
        
        initial_state = self.state_history[0]["state"]
        final_state = self.state_history[-1]["state"]
        
        # Track key changes
        changes = {}
        all_keys = set(initial_state.keys()) | set(final_state.keys())
        
        for key in all_keys:
            initial_val = initial_state.get(key, "NOT_SET")
            final_val = final_state.get(key, "NOT_SET")
            
            if initial_val != final_val:
                changes[key] = {
                    "initial": initial_val,
                    "final": final_val,
                    "changed": True
                }
            else:
                changes[key] = {
                    "value": final_val,
                    "changed": False
                }
        
        return {
            "state_changes": changes,
            "total_state_snapshots": len(self.state_history),
            "key_metrics": {
                "query_rewritten": changes.get("rewritten_query", {}).get("changed", False),
                "loop_count": final_state.get("loop_count", 0),
                "response_generated": final_state.get("response_generated", False),
                "reflection_performed": final_state.get("reflection_complete", False)
            }
        }
    
    def compare_queries(self, queries: list) -> pd.DataFrame:
        """
        Compare multiple queries and their execution patterns
        
        Args:
            queries: List of queries to compare
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for query in queries:
            self.state_history = []  # Reset history
            
            try:
                debug_info = self.debug_single_query(query)
                
                comparison_data.append({
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "query_type": debug_info["final_result"].get("query_type", "unknown"),
                    "execution_time": debug_info["final_result"].get("execution_metadata", {}).get("total_loops", 1),
                    "loops": debug_info["key_metrics"]["loop_count"],
                    "query_rewritten": debug_info["key_metrics"]["query_rewritten"],
                    "reflection_performed": debug_info["key_metrics"]["reflection_performed"],
                    "response_length": len(debug_info["final_result"].get("response", "")),
                    "retrieved_docs": len(debug_info["final_result"].get("retrieved_docs", [])),
                    "success": "error" not in debug_info["final_result"]
                })
                
            except Exception as e:
                comparison_data.append({
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "error": str(e),
                    "success": False
                })
        
        return pd.DataFrame(comparison_data)
    
    def print_debug_summary(self, debug_info: Dict[str, Any]):
        """Print formatted debug summary"""
        print("\n" + "="*80)
        print("🔍 GRAPH DEBUG SUMMARY")
        print("="*80)
        
        # Key metrics
        metrics = debug_info["key_metrics"]
        print(f"🔄 Query rewritten: {metrics['query_rewritten']}")
        print(f"🔁 Loop count: {metrics['loop_count']}")
        print(f"💭 Response generated: {metrics['response_generated']}")
        print(f"🤔 Reflection performed: {metrics['reflection_performed']}")
        
        # State changes
        print(f"\n📊 STATE CHANGES:")
        changes = debug_info["state_changes"]
        
        for key, change_info in changes.items():
            if change_info.get("changed", False):
                print(f"  ✏️  {key}:")
                print(f"    Initial: {change_info['initial']}")
                print(f"    Final: {change_info['final']}")
        
        print("="*80)


def main():
    """Main debug function"""
    debugger = GraphDebugger()
    
    # Debug single query
    print("🔍 Debugging single query...")
    debug_info = debugger.debug_single_query(
        "What is the weather in Beijing?",
        use_rag=True,
        similarity_threshold=0.7
    )
    
    debugger.print_debug_summary(debug_info)
    
    # Compare multiple queries
    print("\n🔍 Comparing multiple queries...")
    test_queries = [
        "What is artificial intelligence?",
        "Weather in Shanghai?",
        "Who is Ong Leong Chin?",
        "Hello, how are you?"
    ]
    
    comparison_df = debugger.compare_queries(test_queries)
    print("\n📊 QUERY COMPARISON:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison to CSV
    comparison_df.to_csv("test_outputs/query_comparison.csv", index=False)
    print("\n💾 Comparison saved to test_outputs/query_comparison.csv")


if __name__ == "__main__":
    main()