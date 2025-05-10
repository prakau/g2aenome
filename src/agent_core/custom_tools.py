from typing import Type, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from loguru import logger

# Assuming VectorStoreManager and KnowledgeGraphManager are accessible
# This might require adjusting import paths depending on how the project is structured
# and how these managers are instantiated and passed to the tools.
# For this example, we'll assume they are passed during tool initialization.

# from ..knowledge_services.vector_store_manager import VectorStoreManager
# from ..knowledge_services.knowledge_graph_manager import KnowledgeGraphManager
# from ..bio_tools import run_sequence_alignment # Placeholder for GenomicAnalysisTool

# --- Input Schemas for Tools ---
class SemanticSearchInput(BaseModel):
    query: str = Field(description="Natural language query for semantic search in documents.")
    k: Optional[int] = Field(default=5, description="Number of top results to return.")

class KnowledgeGraphLookupInput(BaseModel):
    entity_name: str = Field(description="Name or ID of the entity to look up in the knowledge graph.")
    relation_type: Optional[str] = Field(default=None, description="Optional: Filter relationships by this type.")
    depth: Optional[int] = Field(default=1, description="Query depth (currently supports 1-hop).")

# (Optional Advanced) GenomicAnalysisTool Input Schema
# class GenomicAnalysisInput(BaseModel):
#     sequence1: str = Field(description="First DNA/protein sequence.")
#     sequence2: Optional[str] = Field(default=None, description="Second DNA/protein sequence for alignment.")
#     analysis_type: str = Field(description="Type of analysis, e.g., 'align', 'orf_find'.")


# --- Custom Tools ---

class SemanticSearchTool(BaseTool):
    """
    Tool to perform semantic search over a vector store of text chunks.
    """
    name: str = "semantic_search_documents"
    description: str = (
        "Performs semantic search on a collection of documents (e.g., Dr. Kaushik's publications) "
        "to find text segments most relevant to a natural language query. "
        "Useful for finding contextual information, definitions, or discussions related to a topic."
    )
    args_schema: Type[BaseModel] = SemanticSearchInput
    vector_store_manager: Any # Expects an instance of VectorStoreManager

    def _run(self, query: str, k: Optional[int] = 5) -> List[Dict[str, Any]]:
        """Use the tool."""
        logger.info(f"SemanticSearchTool called with query: '{query}', k={k}")
        if not self.vector_store_manager:
            logger.error("VectorStoreManager not initialized for SemanticSearchTool.")
            return [{"error": "VectorStoreManager not available."}]
        try:
            results = self.vector_store_manager.search_vector_store(query_text=query, k=k)
            # Format results for better readability by LLM
            formatted_results = []
            for res in results:
                formatted_results.append({
                    "document_source": res.get("source_document"),
                    "page_number": res.get("page_number"),
                    "chunk_id": res.get("chunk_id"),
                    "relevant_text_snippet": res.get("text"),
                    "relevance_score": res.get("score"),
                    "score_type": res.get("score_type")
                })
            return formatted_results if formatted_results else [{"info": "No relevant text segments found."}]
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return [{"error": f"Semantic search failed: {str(e)}"}]

    async def _arun(self, query: str, k: Optional[int] = 5) -> List[Dict[str, Any]]:
        """Use the tool asynchronously."""
        # For simplicity, reusing the synchronous version.
        # For true async, VectorStoreManager would need async methods.
        return self._run(query, k)


class KnowledgeGraphLookupTool(BaseTool):
    """
    Tool to query the knowledge graph for entities and their relationships.
    """
    name: str = "knowledge_graph_lookup"
    description: str = (
        "Queries the knowledge graph to find information about specific entities (like genes, proteins, traits, crop species) "
        "and their relationships. Useful for structured data retrieval, finding connections between entities, "
        "or getting attributes of a known entity."
    )
    args_schema: Type[BaseModel] = KnowledgeGraphLookupInput
    kg_manager: Any # Expects an instance of KnowledgeGraphManager

    def _run(self, entity_name: str, relation_type: Optional[str] = None, depth: Optional[int] = 1) -> List[Dict[str, Any]]:
        """Use the tool."""
        logger.info(f"KnowledgeGraphLookupTool called for entity: '{entity_name}', relation: {relation_type}, depth: {depth}")
        if not self.kg_manager:
            logger.error("KnowledgeGraphManager not initialized for KnowledgeGraphLookupTool.")
            return [{"error": "KnowledgeGraphManager not available."}]
        try:
            # First, get details of the primary entity itself
            node_details = self.kg_manager.get_node_details(entity_name)
            
            # Then, get its relationships
            related_info = self.kg_manager.query_graph(entity_name=entity_name, relation_type=relation_type, depth=depth)
            
            results = []
            if node_details:
                results.append({"entity_queried": entity_name, "attributes": node_details})
            
            if related_info:
                # If node_details was None, results might be empty or contain an "info" message.
                # If we only want to add related_info if node_details was also found, adjust logic.
                # For now, extend regardless, or clear "info" message if related_info is found.
                if not node_details and results and "info" in results[0]: # If only "info" message exists
                    results.pop() # Remove the "info" message if we found relations

                results.extend(related_info)

            if not results: 
                # This means neither node_details (actual attributes) nor related_info were found.
                return [{"info": f"No information or specific relationships found for entity '{entity_name}' (relation: {relation_type if relation_type else 'ANY'})."}]
            
            # If only node_details was found but no specific related_info for the given relation_type
            if node_details and not related_info and relation_type:
                 # Check if results only contains the node_details part
                if len(results) == 1 and "attributes" in results[0]:
                    results.append({"info": f"Entity '{entity_name}' found, but no specific relationships of type '{relation_type}'."})

            return results

        except Exception as e:
            logger.error(f"Error during knowledge graph lookup: {e}")
            return [{"error": f"Knowledge graph lookup failed: {str(e)}"}]

    async def _arun(self, entity_name: str, relation_type: Optional[str] = None, depth: Optional[int] = 1) -> List[Dict[str, Any]]:
        """Use the tool asynchronously."""
        return self._run(entity_name, relation_type, depth)


# (Optional Advanced) GenomicAnalysisTool Placeholder
# class GenomicAnalysisTool(BaseTool):
#     name: str = "genomic_sequence_analysis"
#     description: str = (
#         "Performs basic genomic sequence analyses, such as sequence alignment (pairwise), "
#         "finding Open Reading Frames (ORFs), or other BioPython-powered functions. "
#         "Use when direct sequence manipulation or comparison is needed."
#     )
#     args_schema: Type[BaseModel] = GenomicAnalysisInput
#     # This tool might not need a manager if it directly calls BioPython functions.
#     # Or it could have a BioToolsWrapper.

#     def _run(self, sequence1: str, analysis_type: str, sequence2: Optional[str] = None) -> Dict[str, Any]:
#         logger.info(f"GenomicAnalysisTool called. Type: {analysis_type}")
#         if analysis_type == "align" and sequence2:
#             # result = run_sequence_alignment(sequence1, sequence2) # Placeholder
#             result = {"status": "alignment_stub_success", "score": "dummy_score_100"}
#             return result
#         elif analysis_type == "orf_find":
#             # result = find_orfs(sequence1) # Placeholder
#             result = {"status": "orf_find_stub_success", "orfs_found": ["orf1_dummy", "orf2_dummy"]}
#             return result
#         else:
#             return {"error": f"Unsupported analysis type '{analysis_type}' or missing sequence2 for alignment."}

#     async def _arun(self, sequence1: str, analysis_type: str, sequence2: Optional[str] = None) -> Dict[str, Any]:
#         return self._run(sequence1, analysis_type, sequence2)


if __name__ == '__main__':
    # Example of how tools might be instantiated and used (requires mock managers)
    # This is conceptual; actual instantiation happens in agent_orchestrator.py
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    import sys

    # --- Mock Managers for testing the tools ---
    class MockVectorStoreManager:
        def search_vector_store(self, query_text: str, k: int):
            logger.debug(f"MockVSM: Searching for '{query_text}' (k={k})")
            return [
                {"text": f"Relevant text snippet about {query_text} 1", "source_document": "mock_doc1.pdf", "page_number": 1, "score": 0.9, "score_type": "similarity"},
                {"text": f"Relevant text snippet about {query_text} 2", "source_document": "mock_doc2.pdf", "page_number": 5, "score": 0.85, "score_type": "similarity"}
            ]

    class MockKnowledgeGraphManager:
        def get_node_details(self, entity_name: str):
            logger.debug(f"MockKGM: Getting details for '{entity_name}'")
            if entity_name == "GeneX":
                return {"type": "GENE", "label": "GeneX", "description": "A hypothetical gene."}
            return None
        
        def query_graph(self, entity_name: str, relation_type: Optional[str] = None, depth: int = 1):
            logger.debug(f"MockKGM: Querying for '{entity_name}', relation '{relation_type}'")
            if entity_name == "GeneX" and (relation_type is None or relation_type == "REGULATES"):
                return [
                    {"source_entity": "GeneX", "relation": "REGULATES", "target_entity": "GeneY", "target_entity_type": "GENE", "target_entity_label": "GeneY", "evidence": "GeneX regulates GeneY.", "direction": "outgoing"},
                    {"source_entity": "ProteinA", "relation": "INTERACTS_WITH", "target_entity": "GeneX", "source_entity_type": "PROTEIN", "source_entity_label": "ProteinA", "evidence": "ProteinA interacts with GeneX.", "direction": "incoming"}
                ]
            return []

    # --- Instantiate Tools with Mock Managers ---
    mock_vsm = MockVectorStoreManager()
    mock_kgm = MockKnowledgeGraphManager()

    semantic_tool = SemanticSearchTool(vector_store_manager=mock_vsm)
    kg_tool = KnowledgeGraphLookupTool(kg_manager=mock_kgm)

    logger.info("\n--- Testing SemanticSearchTool ---")
    semantic_results = semantic_tool.run({"query": "eggplant drought", "k": 2})
    logger.info(f"Semantic Search Results: {semantic_results}")

    logger.info("\n--- Testing KnowledgeGraphLookupTool ---")
    kg_results_geneX = kg_tool.run({"entity_name": "GeneX", "relation_type": "REGULATES"})
    logger.info(f"KG Lookup Results for GeneX (REGULATES): {kg_results_geneX}")
    
    kg_results_geneZ = kg_tool.run({"entity_name": "GeneZ"}) # Non-existent entity
    logger.info(f"KG Lookup Results for GeneZ: {kg_results_geneZ}")
    
    logger.info("\nCustom tools example usage finished.")
