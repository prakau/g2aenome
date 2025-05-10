from typing import List, Dict, Any, Optional
from loguru import logger
import yaml
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain.agents import AgentExecutor, create_react_agent # Example agent type
# from langchain.agents import create_openai_tools_agent # Alternative for OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory # For conversation history

# Custom tools and prompts
from .custom_tools import SemanticSearchTool, KnowledgeGraphLookupTool #, GenomicAnalysisTool
from .prompt_templates import get_agent_system_prompt

# Managers (passed during initialization)
# from ..knowledge_services import KnowledgeGraphManager, VectorStoreManager


class G2AenomeAgent:
    """
    Orchestrates the G²AENome agent, integrating LLM, tools, and prompts.
    """

    def __init__(self, models_config_path: str = None, 
                 knowledge_graph_manager: Any = None, 
                 vector_store_manager: Any = None):
        """
        Initializes the G2AenomeAgent.
        Args:
            models_config_path (str, optional): Path to models_config.yaml.
            knowledge_graph_manager (Any): Initialized KnowledgeGraphManager instance.
            vector_store_manager (Any): Initialized VectorStoreManager instance.
        """
        self.models_config = {}
        self._load_models_config(models_config_path)

        self.kg_manager = knowledge_graph_manager
        self.vector_store_manager = vector_store_manager

        self.llm = self._initialize_llm()
        self.tools = self._initialize_tools()
        
        self.agent_executor = None
        if self.llm and self.tools:
            self.agent_executor = self._create_agent_executor()
        else:
            logger.error("LLM or Tools could not be initialized. Agent executor not created.")

        # Memory for conversational context
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Number of past interactions to remember
            memory_key="chat_history",
            return_messages=True # Important for ReAct agent with MessagesPlaceholder
        )
        # For more advanced, persistent memory, especially with LangGraph,
        # consider using `langgraph.checkpoint.memory.MemorySaver` or other
        # persistent checkpointers.

    def _load_models_config(self, config_path: str = None):
        """Loads LLM and other model configurations."""
        if config_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            config_path = project_root / "configs" / "models_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                self.models_config = yaml.safe_load(f).get("llm", {})
            logger.info("G2AenomeAgent: LLM configuration loaded.")
        except FileNotFoundError:
            logger.error(f"Models configuration file not found at {config_path} for Agent. Using defaults.")
            self.models_config = {"provider": "OpenAI", "openai_model_name": "gpt-3.5-turbo"} # Basic default
        except Exception as e:
            logger.error(f"Error loading LLM configuration from {config_path}: {e}. Using defaults.")
            self.models_config = {"provider": "OpenAI", "openai_model_name": "gpt-3.5-turbo"}


    def _initialize_llm(self) -> Optional[Any]:
        """Initializes the LLM based on configuration."""
        provider = self.models_config.get("provider", "OpenAI").lower()
        temperature = self.models_config.get("temperature", 0.1)
        max_tokens = self.models_config.get("max_tokens", 2048)

        logger.info(f"Initializing LLM provider: {provider}")
        try:
            if provider == "openai":
                model_name = self.models_config.get("openai_model_name", "gpt-3.5-turbo")
                # OPENAI_API_KEY should be in environment variables
                return ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
            
            elif provider == "huggingfaceendpoint":
                endpoint_url = self.models_config.get("endpoint_url")
                hf_token = self.models_config.get("hf_token") # Best practice: from env
                if not endpoint_url:
                    logger.error("HuggingFaceEndpoint URL not provided in models_config.yaml.")
                    return None
                # HUGGINGFACEHUB_API_TOKEN for the endpoint can be set in env
                return HuggingFaceEndpoint(
                    endpoint_url=endpoint_url,
                    huggingfacehub_api_token=hf_token, # Pass token if needed by endpoint
                    task="text-generation", # Or appropriate task
                    temperature=temperature,
                    max_new_tokens=max_tokens # Note: HuggingFaceEndpoint uses max_new_tokens
                )
            
            elif provider == "huggingfacehub": # For models directly on HF Hub via API
                repo_id = self.models_config.get("repo_id")
                if not repo_id:
                    logger.error("HuggingFace Hub repo_id not provided in models_config.yaml.")
                    return None
                # HUGGINGFACEHUB_API_TOKEN should be in environment variables
                # This often uses HuggingFacePipeline for more control
                return HuggingFacePipeline.from_model_id(
                    model_id=repo_id,
                    task="text-generation", # Adjust task as needed
                    pipeline_kwargs={"max_new_tokens": max_tokens, "temperature": temperature}
                    # device_map="auto" # if using local transformers and have GPU
                )

            # Add other providers like AzureOpenAI, Anthropic, etc. as needed
            else:
                logger.error(f"Unsupported LLM provider: {provider}. Defaulting to OpenAI gpt-3.5-turbo if possible.")
                # Fallback attempt, assuming OPENAI_API_KEY might be set
                return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, max_tokens=max_tokens)
        
        except Exception as e:
            logger.critical(f"Failed to initialize LLM (Provider: {provider}): {e}")
            return None

    def _initialize_tools(self) -> List[Any]:
        """Initializes and returns a list of tools for the agent."""
        tools = []
        if self.vector_store_manager:
            tools.append(SemanticSearchTool(vector_store_manager=self.vector_store_manager))
        else:
            logger.warning("VectorStoreManager not provided. SemanticSearchTool will not be available.")

        if self.kg_manager:
            tools.append(KnowledgeGraphLookupTool(kg_manager=self.kg_manager))
        else:
            logger.warning("KnowledgeGraphManager not provided. KnowledgeGraphLookupTool will not be available.")
        
        # Example for GenomicAnalysisTool (if implemented)
        # tools.append(GenomicAnalysisTool())
        
        logger.info(f"Initialized {len(tools)} tools for the agent.")
        return tools

    def _create_agent_executor(self) -> Optional[AgentExecutor]:
        """Creates and returns the LangChain agent executor."""
        if not self.llm or not self.tools:
            logger.error("Cannot create agent executor: LLM or tools are missing.")
            return None

        # Define the prompt structure for ReAct agent
        # Includes system message, chat history, human input, and agent scratchpad
        system_prompt_template = get_agent_system_prompt() # From prompt_templates.py
        
        prompt = ChatPromptTemplate.from_messages([
            system_prompt_template,
            MessagesPlaceholder(variable_name="chat_history", optional=True), # For conversation memory
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad") # Where agent stores its thoughts/tool calls
        ])

        try:
            # Using the ReAct agent framework
            # This agent uses a ReAct (Reasoning and Acting) prompting style.
            react_agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
            
            agent_executor = AgentExecutor(
                agent=react_agent, 
                tools=self.tools, 
                verbose=True, # Set to True for detailed logging of agent's thoughts and actions
                handle_parsing_errors=True, # Gracefully handle if LLM output is not parsable
                max_iterations=10, # Prevent overly long chains of thought
                # early_stopping_method="generate", # Stop if LLM generates a final answer
                memory=self.memory # Add memory here
            )
            logger.info("ReAct AgentExecutor created successfully.")
            return agent_executor
        except Exception as e:
            logger.critical(f"Failed to create AgentExecutor: {e}")
            return None

    def ask_question(self, query_text: str) -> Dict[str, Any]:
        """
        Asks a question to the G²AENome agent.
        Args:
            query_text (str): The user's question.
        Returns:
            Dict[str, Any]: The agent's response, typically including an 'output' field.
        """
        if not self.agent_executor:
            logger.error("Agent executor is not available. Cannot process question.")
            return {"error": "Agent is not properly initialized."}

        logger.info(f"Processing query: '{query_text}'")
        try:
            # The input to invoke must match the variables in the HumanMessagePromptTemplate ('input')
            # and potentially 'chat_history' if the agent expects it directly (though memory handles it)
            response = self.agent_executor.invoke({
                "input": query_text,
                # chat_history is managed by the memory object passed to AgentExecutor
            })
            
            # The actual answer is usually in the 'output' key of the response dictionary
            logger.info(f"Agent response received. Output: {response.get('output')}")
            return response
            
        except Exception as e:
            logger.error(f"Error during agent invocation for query '{query_text}': {e}")
            return {"error": f"An error occurred: {str(e)}"}


if __name__ == '__main__':
    import sys
    # Example usage of G2AenomeAgent
    # This requires actual or mocked KGM and VSM, and a configured LLM.
    # For a simple run, ensure configs/models_config.yaml points to a usable LLM (e.g., OpenAI with API key in env).
    
    logger.remove()
    logger.add(sys.stderr, level="INFO") # Use INFO for cleaner example output

    # --- Mock Managers for example ---
    class MockKGM:
        def get_node_details(self, entity_name): return {"type": "MOCK_ENTITY", "name": entity_name}
        def query_graph(self, entity_name, relation_type=None, depth=1): 
            return [{"mock_relation": f"{entity_name} related_to MockTarget"}]

    class MockVSM:
        def search_vector_store(self, query_text, k=5): 
            return [{"text": f"Mock semantic result for '{query_text}'", "source_document": "mock.pdf"}]

    mock_kg_manager = MockKGM()
    mock_vector_store_manager = MockVSM()

    # --- Initialize Agent ---
    # Ensure your configs/models_config.yaml is set up for an LLM you can access
    # (e.g., OpenAI with OPENAI_API_KEY in environment)
    # Or, modify the _initialize_llm method to use a local model if preferred.
    
    project_root_example = Path(__file__).resolve().parent.parent.parent
    models_cfg_path_example = project_root_example / "configs" / "models_config.yaml"

    # Create a dummy models_config.yaml if it doesn't exist, pointing to OpenAI
    if not models_cfg_path_example.exists():
        models_cfg_path_example.parent.mkdir(parents=True, exist_ok=True)
        dummy_llm_config = {
            "llm": { # Ensure it's under 'llm' key as expected by _load_models_config
                "provider": "OpenAI", 
                "openai_model_name": "gpt-3.5-turbo-instruct", # Cheaper instruct model for testing
                "temperature": 0.1,
                "max_tokens": 500
            }
        }
        with open(models_cfg_path_example, 'w') as f:
            yaml.dump(dummy_llm_config, f)
        logger.info(f"Created dummy models_config.yaml for Agent example at {models_cfg_path_example}")
        logger.info("Ensure OPENAI_API_KEY is set in your environment for this example to run with OpenAI.")


    logger.info("Initializing G2AenomeAgent...")
    agent = G2AenomeAgent(
        models_config_path=str(models_cfg_path_example),
        knowledge_graph_manager=mock_kg_manager,
        vector_store_manager=mock_vector_store_manager
    )

    if agent.agent_executor:
        logger.info("Agent initialized successfully.")
        
        test_query = "What is known about Solanum melongena and abiotic stress?"
        logger.info(f"\n--- Asking question to agent: '{test_query}' ---")
        
        # First interaction
        response1 = agent.ask_question(test_query)
        logger.info(f"Agent's final answer (1st query): {response1.get('output')}")

        # Second interaction (to test memory)
        # test_query2 = "Tell me more about the genes involved."
        # logger.info(f"\n--- Asking a follow-up question: '{test_query2}' ---")
        # response2 = agent.ask_question(test_query2)
        # logger.info(f"Agent's final answer (2nd query): {response2.get('output')}")

    else:
        logger.error("Agent could not be initialized. Check LLM configuration and API keys.")

    logger.info("\nG2AenomeAgent example usage finished.")
