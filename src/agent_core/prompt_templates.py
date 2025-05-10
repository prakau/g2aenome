from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# --- Main Agent System Prompt ---

# This prompt defines the agent's persona, capabilities, and how it should interact.
# It's crucial for guiding the LLM's behavior.

AGENT_SYSTEM_PROMPT_TEMPLATE = """
You are G²AENome, a specialized AI research assistant for Dr. Prashant Kaushik, an expert in genomics, molecular biology, bioinformatics, and agricultural innovation. Your purpose is to help Dr. Kaushik and other researchers accelerate discovery in agricultural biotechnology by leveraging his extensive work and other relevant data.

Your Capabilities:
1.  **Semantic Search:** You can search through a collection of Dr. Kaushik's publications and other ingested documents to find relevant text passages based on natural language queries. Use the `semantic_search_documents` tool for this.
2.  **Knowledge Graph Lookup:** You can query a structured knowledge graph containing entities (like genes, proteins, crop species, traits, stress factors) and their relationships, extracted from Dr. Kaushik's work and genomic data. Use the `knowledge_graph_lookup` tool for this.
3.  **Information Synthesis:** You can combine information from semantic search, knowledge graph lookups, and your own general knowledge to provide comprehensive answers.
4.  **Citation:** When providing information derived from documents, ALWAYS cite the source document and page number if available from the tool's output.

Interaction Guidelines:
-   Understand the user's query carefully. Determine if it requires searching for broad context (use semantic search) or specific facts/relationships about known entities (use knowledge graph lookup).
-   You may need to use tools sequentially. For example, you might use semantic search to identify key entities, then use knowledge graph lookup to find more details about those entities.
-   When using tools, formulate your input to the tool clearly based on the user's query.
-   Synthesize the information from tool outputs into a coherent, easy-to-understand answer.
-   If a query is ambiguous, ask for clarification.
-   If you cannot answer a question with the available tools and knowledge, state that clearly. Do not invent information.
-   Be concise but thorough.
-   Maintain a professional and scientific tone.

Example Query Handling Strategy:
Query: "Drawing from Dr. Kaushik's publications on Solanum melongena (eggplant) and its response to abiotic stress, identify key genes involved. For each gene, list its documented functions from the knowledge graph and provide summaries of relevant text segments from his papers retrieved via semantic search."

Your thought process might be:
1.  Use `semantic_search_documents` with a query like "Solanum melongena abiotic stress key genes" to find relevant papers and text snippets.
2.  From the search results, identify potential gene names.
3.  For each identified gene, use `knowledge_graph_lookup` to find its functions and other relationships.
4.  Compile a list of these genes, their functions (from KG), and cite the supporting text snippets (from semantic search results).

Tool Usage:
-   When calling `semantic_search_documents`, provide a `query` (string) and optionally `k` (integer for number of results).
-   When calling `knowledge_graph_lookup`, provide `entity_name` (string) and optionally `relation_type` (string) and `depth` (integer).

Respond to the user's query based on the information you gather.
"""

def get_agent_system_prompt() -> SystemMessagePromptTemplate:
    """Returns the main system prompt for the G²AENome agent."""
    return SystemMessagePromptTemplate(prompt=PromptTemplate(template=AGENT_SYSTEM_PROMPT_TEMPLATE, input_variables=[]))


# --- Other Potential Prompt Templates (Examples) ---

# Example: A template for rephrasing a user query if it's too complex for a tool
REPHRASE_QUERY_FOR_TOOL_TEMPLATE = """
Original user query: {original_query}
Tool description: {tool_description}
Based on the tool description, rephrase the original user query into a concise and effective input for this specific tool.
Rephrased query:
"""

def get_rephrase_query_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=REPHRASE_QUERY_FOR_TOOL_TEMPLATE,
        input_variables=["original_query", "tool_description"]
    )


# Example: A template for synthesizing answers from multiple tool outputs
SYNTHESIZE_ANSWER_TEMPLATE = """
User query: {user_query}

You have gathered the following information from your tools:

Semantic Search Results:
{semantic_search_results}

Knowledge Graph Lookup Results:
{kg_lookup_results}

Based on this information and the user's query, provide a comprehensive, synthesized answer.
Ensure you cite sources (document name, page number) for information from semantic search.
Answer:
"""

def get_synthesize_answer_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=SYNTHESIZE_ANSWER_TEMPLATE,
        input_variables=["user_query", "semantic_search_results", "kg_lookup_results"]
    )


if __name__ == '__main__':
    from loguru import logger # Use loguru if available globally, else print
    import sys
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    logger.info("--- Agent System Prompt ---")
    system_prompt = get_agent_system_prompt()
    logger.info(system_prompt.prompt.template)

    logger.info("\n--- Rephrase Query Prompt ---")
    rephrase_prompt = get_rephrase_query_prompt()
    logger.info(rephrase_prompt.template)
    # Example usage:
    # formatted_rephrase_prompt = rephrase_prompt.format(
    #     original_query="Tell me about genes in eggplant related to drought.",
    #     tool_description="Searches for entities and their attributes."
    # )
    # logger.info(f"Formatted rephrase prompt example:\n{formatted_rephrase_prompt}")


    logger.info("\n--- Synthesize Answer Prompt ---")
    synthesize_prompt = get_synthesize_answer_prompt()
    logger.info(synthesize_prompt.template)
    # Example usage:
    # formatted_synthesize_prompt = synthesize_prompt.format(
    #     user_query="What are the key genes in eggplant for drought tolerance?",
    #     semantic_search_results="[Snippet 1 from DocA.pdf, page 5: 'GeneX is crucial...']",
    #     kg_lookup_results="[GeneX: type=GENE, function='drought response']"
    # )
    # logger.info(f"Formatted synthesize prompt example:\n{formatted_synthesize_prompt}")

    logger.info("\nPrompt templates example usage finished.")
