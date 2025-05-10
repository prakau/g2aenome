import streamlit as st
from pathlib import Path
from loguru import logger
import sys
import os

# Adjust Python path to include the 'src' directory
# This allows importing modules from src like 'common', 'agent_core', etc.
PROJECT_ROOT_APP = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT_APP / "src"))

# Import necessary components from the project
try:
    from common.logging_config import setup_logging
    from knowledge_services.knowledge_graph_manager import KnowledgeGraphManager
    from knowledge_services.vector_store_manager import VectorStoreManager
    from agent_core.agent_orchestrator import G2AenomeAgent
except ImportError as e:
    st.error(f"Failed to import project modules: {e}. Ensure PYTHONPATH is set correctly or app is run from project root.")
    # Fallback for local dev: if run from app/ directory, src/ is one level up.
    # This is a bit of a hack; proper packaging or running from root is better.
    if str(PROJECT_ROOT_APP / "src") not in sys.path:
         alt_src_path = Path(__file__).resolve().parent.parent / "src"
         if alt_src_path.exists() and str(alt_src_path) not in sys.path:
              sys.path.insert(0, str(alt_src_path))
              logger.info(f"Added {alt_src_path} to sys.path for Streamlit app.")
              # Retry imports
              from common.logging_config import setup_logging
              from knowledge_services.knowledge_graph_manager import KnowledgeGraphManager
              from knowledge_services.vector_store_manager import VectorStoreManager
              from agent_core.agent_orchestrator import G2AenomeAgent
         else:
             raise e # Re-raise if alternative path also doesn't work


# --- Configuration & Initialization ---
# Setup logging (will use configs/main_config.yaml by default)
# Note: Streamlit has its own way of handling logs in deployment,
# but for local dev, this ensures our app's logs are captured.
MAIN_CONFIG_FILE_PATH = PROJECT_ROOT_APP / "configs" / "main_config.yaml" # Renamed for clarity
MODELS_CONFIG_PATH = PROJECT_ROOT_APP / "configs" / "models_config.yaml"

# Initialize logger (it's okay if main_pipeline also calls this; loguru handles it)
setup_logging(config_path=MAIN_CONFIG_FILE_PATH) # Use the new clear name

# --- Page Configuration (Streamlit) ---
st.set_page_config(
    page_title="G²AENome AI Nexus",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your_username/g2aenome_dr_kaushik_adv_poc', # Replace
        'Report a bug': "https://github.com/your_username/g2aenome_dr_kaushik_adv_poc/issues", # Replace
        'About': "# G²AENome: Genomic & Agri-Phenomic AI Nexus\nAdvanced PoC for Dr. Prashant Kaushik."
    }
)

# --- Caching for expensive operations ---
@st.cache_resource(show_spinner="Loading Knowledge Graph Manager...")
def load_kg_manager():
    logger.info("Attempting to load KnowledgeGraphManager for Streamlit app.")
    kgm = KnowledgeGraphManager(main_config_path=str(MAIN_CONFIG_FILE_PATH)) # Use the new clear name
    if not kgm.load_graph(): # Tries to load from path configured in main_config.yaml
        st.warning("Knowledge Graph data not found or failed to load. KG-related queries might not work. Please run `src/main_pipeline.py` first.")
    else:
        logger.info(f"KG loaded with {kgm.graph.number_of_nodes()} nodes and {kgm.graph.number_of_edges()} edges.")
    return kgm

@st.cache_resource(show_spinner="Loading Vector Store Manager...")
def load_vector_store_manager():
    logger.info("Attempting to load VectorStoreManager for Streamlit app.")
    vsm = VectorStoreManager(main_config_path=str(MAIN_CONFIG_FILE_PATH), models_config_path=str(MODELS_CONFIG_PATH)) # Use the new clear name
    # create_or_load_vector_store will try to load if index exists, otherwise it needs data to build.
    # For the app, we assume it's pre-built by the pipeline.
    vsm.create_or_load_vector_store(force_recreate=False) # force_recreate=False to load existing
    if vsm.index is None or vsm.index.ntotal == 0:
        st.warning("Vector Store not found, empty, or failed to load. Semantic search might not work. Please run `src/main_pipeline.py` first.")
    else:
         logger.info(f"Vector Store loaded with {vsm.index.ntotal} vectors.")
    return vsm

@st.cache_resource(show_spinner="Initializing G²AENome Agent...")
def initialize_agent(_kg_manager, _vector_store_manager):
    logger.info("Initializing G²AENomeAgent for Streamlit app.")
    # Ensure API keys (e.g., OPENAI_API_KEY) are in the environment if needed by the LLM config
    # Streamlit Cloud allows setting secrets. For local, use .env file (python-dotenv should handle it).
    agent = G2AenomeAgent(
        models_config_path=str(MODELS_CONFIG_PATH),
        knowledge_graph_manager=_kg_manager,
        vector_store_manager=_vector_store_manager
    )
    if not agent.agent_executor:
        st.error("G²AENome Agent could not be initialized. Check LLM configuration and logs.")
        return None
    logger.info("G²AENome Agent initialized successfully for Streamlit app.")
    return agent

# --- Load components ---
# These are cached, so they only load once per session or until cache invalidates.
kg_manager_instance = load_kg_manager()
vector_store_manager_instance = load_vector_store_manager()
g2aenome_agent = initialize_agent(kg_manager_instance, vector_store_manager_instance)


# --- Streamlit UI Layout ---
st.title("🧬 G²AENome: Genomic & Agri-Phenomic AI Nexus")
st.subheader("Advanced Proof-of-Concept for Dr. Prashant Kaushik")
st.markdown("Ask questions related to Dr. Kaushik's publications, genomics, and agricultural biotechnology.")

# Sidebar for options or info (optional)
with st.sidebar:
    st.header("About")
    st.markdown(
        "This application demonstrates the capabilities of the G²AENome platform "
        "to process and query scientific literature and genomic data."
    )
    st.markdown("---")
    st.header("Status")
    if kg_manager_instance and kg_manager_instance.graph and kg_manager_instance.graph.number_of_nodes() > 0:
        st.success(f"Knowledge Graph: Loaded ({kg_manager_instance.graph.number_of_nodes()} nodes)")
    else:
        st.warning("Knowledge Graph: Not loaded or empty.")

    if vector_store_manager_instance and vector_store_manager_instance.index and vector_store_manager_instance.index.ntotal > 0:
        st.success(f"Vector Store: Loaded ({vector_store_manager_instance.index.ntotal} vectors)")
    else:
        st.warning("Vector Store: Not loaded or empty.")
    
    if g2aenome_agent and g2aenome_agent.agent_executor:
        st.success("G²AENome Agent: Ready")
    else:
        st.error("G²AENome Agent: Not available")
    
    st.markdown("---")
    if st.button("Clear Chat History & Cache"):
        st.session_state.messages = []
        if g2aenome_agent:
            g2aenome_agent.memory.clear() # Clear agent's internal memory
        st.cache_resource.clear() # Clear Streamlit's resource cache
        st.rerun()


# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask G²AENome... (e.g., 'What are key genes in eggplant for abiotic stress?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if g2aenome_agent and g2aenome_agent.agent_executor:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_text = ""
            
            # Streamlit's way to show "thinking..."
            with st.spinner("G²AENome is thinking..."):
                try:
                    logger.info(f"Streamlit App: Sending query to agent: {prompt}")
                    # The agent's ask_question method should handle memory internally
                    agent_response = g2aenome_agent.ask_question(prompt)
                    
                    if "output" in agent_response:
                        full_response_text = agent_response["output"]
                    elif "error" in agent_response:
                        full_response_text = f"An error occurred: {agent_response['error']}"
                    else:
                        full_response_text = "Sorry, I received an unexpected response format from the agent."
                    
                    logger.info(f"Streamlit App: Agent raw response: {agent_response}")

                except Exception as e:
                    logger.error(f"Error during agent query from Streamlit app: {e}", exc_info=True)
                    full_response_text = f"An application error occurred: {str(e)}"
            
            message_placeholder.markdown(full_response_text)
        st.session_state.messages.append({"role": "assistant", "content": full_response_text})
    else:
        st.error("The G²AENome agent is not available. Please check the logs or ensure the backend pipeline has run.")

# To run this app:
# 1. Ensure all dependencies from requirements.txt are installed.
# 2. Make sure the main pipeline (src/main_pipeline.py) has been run at least once to generate
#    the knowledge graph and vector store artifacts in the 'outputs/' directory (or as configured).
# 3. Set any necessary environment variables (e.g., OPENAI_API_KEY if using OpenAI).
# 4. From the project root directory (g2aenome_dr_kaushik_adv_poc), run:
#    streamlit run app/streamlit_app.py
