# test_graph.py
"""
IntelliFlow Graph Testing and Visualization Script
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from pathlib import Path

os.makedirs("test_outputs", exist_ok=True)

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.settings import settings
from src.database.connection import init_database, test_connection
from src.agents.graph import IntelliFlowGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_outputs/test_graph.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GraphTester:
    """Test and visualize IntelliFlow graph"""
    
    def __init__(self):
        """Initialize the tester"""
        self.graph = IntelliFlowGraph()
        self.test_results = []
        
        # Ensure database is ready
        if not test_connection():
            logger.error("Database connection failed!")
            sys.exit(1)
        
        init_database()
        logger.info("GraphTester initialized successfully")
    
    def visualize_graph(self, save_path: str = "graph_visualization.png"):
        """
        Visualize the graph structure
        
        Args:
            save_path: Path to save the visualization
        """
        graph_png = self.graph.graph.get_graph().draw_mermaid_png()
        with open(save_path, "wb") as f:
            f.write(graph_png)
    
    def _create_hierarchical_layout(self, G) -> Dict[str, tuple]:
        """Create hierarchical layout for the graph"""
        # Define levels for hierarchical layout
        levels = {
            0: ["START"],
            1: ["analyze_query"],
            2: ["rewrite_query"],
            3: ["retrieve_documents", "query_weather"],
            4: ["generate_response"],
            5: ["reflect_on_response"],
            6: ["prepare_retry", "END"]
        }
        
        pos = {}
        for level, nodes in levels.items():
            y = -level * 1.5  # Vertical spacing
            if len(nodes) == 1:
                pos[nodes[0]] = (0, y)
            else:
                # Distribute nodes horizontally
                x_positions = [i * 2 - (len(nodes) - 1) for i in range(len(nodes))]
                for i, node in enumerate(nodes):
                    pos[node] = (x_positions[i], y)
        
        return pos
    
    def run_test_queries(self, test_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run test queries and collect results
        
        Args:
            test_queries: List of test query dictionaries
            
        Returns:
            List of test results
        """
        results = []
        
        for i, test_case in enumerate(test_queries, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Running Test Case {i}: {test_case['name']}")
            logger.info(f"Query: {test_case['query']}")
            logger.info(f"{'='*60}")
            
            try:
                # Run the query
                start_time = datetime.now()
                result = self.graph.run(
                    query=test_case['query'],
                    use_rag=test_case.get('use_rag', True),
                    similarity_threshold=test_case.get('similarity_threshold', 0.7)
                )
                end_time = datetime.now()
                
                # Calculate execution time
                execution_time = (end_time - start_time).total_seconds()
                
                # Prepare test result
                test_result = {
                    "test_case": test_case['name'],
                    "query": test_case['query'],
                    "execution_time": execution_time,
                    "success": "error" not in result,
                    "response": result.get("response", ""),
                    "query_type": result.get("query_type", "unknown"),
                    "total_loops": result.get("execution_metadata", {}).get("total_loops", 1),
                    "query_rewritten": result.get("execution_metadata", {}).get("query_rewritten", False),
                    "reflection_performed": result.get("execution_metadata", {}).get("reflection_performed", False),
                    "retrieved_docs_count": len(result.get("retrieved_docs", [])),
                    "weather_info_available": bool(result.get("weather_info", "")),
                    "thinking_process": result.get("thinking_process", []),
                    "formatted_thinking": result.get("formatted_thinking", ""),
                    "full_result": result
                }
                
                results.append(test_result)
                
                # Log summary
                logger.info(f"✅ Test completed successfully")
                logger.info(f"⏱️  Execution time: {execution_time:.2f}s")
                logger.info(f"🔄 Total loops: {test_result['total_loops']}")
                logger.info(f"📝 Query rewritten: {test_result['query_rewritten']}")
                logger.info(f"🤔 Reflection performed: {test_result['reflection_performed']}")
                logger.info(f"📚 Retrieved docs: {test_result['retrieved_docs_count']}")
                logger.info(f"🌤️ Weather info: {test_result['weather_info_available']}")
                
                # Print thinking process if available
                if test_result['formatted_thinking']:
                    logger.info(f"\n🧠 Thinking Process:\n{test_result['formatted_thinking']}")
                
                # Print response
                logger.info(f"\n💬 Response:\n{test_result['response']}")
                
            except Exception as e:
                logger.error(f"❌ Test case {i} failed: {str(e)}")
                results.append({
                    "test_case": test_case['name'],
                    "query": test_case['query'],
                    "success": False,
                    "error": str(e),
                    "execution_time": 0
                })
        
        return results
    
    def generate_test_report(self, results: List[Dict[str, Any]], save_path: str = "test_report.json"):
        """
        Generate comprehensive test report
        
        Args:
            results: Test results
            save_path: Path to save the report
        """
        try:
            # Calculate summary statistics
            total_tests = len(results)
            successful_tests = sum(1 for r in results if r.get("success", False))
            failed_tests = total_tests - successful_tests
            
            avg_execution_time = sum(r.get("execution_time", 0) for r in results) / total_tests if total_tests > 0 else 0
            
            # Query type distribution
            query_types = {}
            for result in results:
                qtype = result.get("query_type", "unknown")
                query_types[qtype] = query_types.get(qtype, 0) + 1
            
            # Create comprehensive report
            report = {
                "test_summary": {
                    "timestamp": datetime.now().isoformat(),
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "failed_tests": failed_tests,
                    "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                    "average_execution_time": avg_execution_time,
                    "query_type_distribution": query_types
                },
                "detailed_results": results
            }
            
            # Save report
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            # Print summary
            logger.info(f"\n{'='*60}")
            logger.info("TEST SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"📊 Total tests: {total_tests}")
            logger.info(f"✅ Successful: {successful_tests}")
            logger.info(f"❌ Failed: {failed_tests}")
            logger.info(f"📈 Success rate: {report['test_summary']['success_rate']:.1f}%")
            logger.info(f"⏱️  Average execution time: {avg_execution_time:.2f}s")
            logger.info(f"📋 Query types: {query_types}")
            logger.info(f"💾 Report saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate test report: {str(e)}")


def main():
    """Main test function"""
    # Initialize tester
    tester = GraphTester()
    
    # Visualize the graph
    logger.info("Generating graph visualization...")
    tester.visualize_graph("test_outputs/intelliflow_graph.png")
    
    # Define test cases
    test_queries = [
        {
            "name": "Simple Knowledge Query",
            "query": "What is artificial intelligence?",
            "use_rag": True,
            "similarity_threshold": 0.7
        },
        {
            "name": "Weather Query - Beijing",
            "query": "What's the weather like in Beijing?",
            "use_rag": True,
            "similarity_threshold": 0.7
        },
        {
            "name": "Complex Knowledge Query",
            "query": "How does machine learning differ from deep learning and what are the practical applications?",
            "use_rag": True,
            "similarity_threshold": 0.6
        },
        {
            "name": "Weather Query - Shanghai",
            "query": "Tell me about the weather forecast for Shanghai",
            "use_rag": True,
            "similarity_threshold": 0.7
        },
        {
            "name": "General Conversation",
            "query": "Hello, how are you today?",
            "use_rag": False,
            "similarity_threshold": 0.7
        },
        {
            "name": "Technical Query",
            "query": "Explain the concept of vector databases and their applications in RAG systems",
            "use_rag": True,
            "similarity_threshold": 0.5
        },
        {
            "name": "Weather Query - Multiple Cities",
            "query": "Compare the weather between Beijing and Shanghai",
            "use_rag": True,
            "similarity_threshold": 0.7
        }
    ]
    
    # Run tests
    logger.info("Starting test execution...")
    results = tester.run_test_queries(test_queries)
    
    # Generate report
    logger.info("Generating test report...")
    tester.generate_test_report(results, "test_outputs/intelliflow_test_report.json")
    
    logger.info("Testing completed successfully!")


if __name__ == "__main__":
    main()