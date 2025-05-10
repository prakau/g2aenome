# G²AENome (Genomic & Agri-Phenomic AI Nexus) - Dr. Prashant Kaushik Advanced PoC

## Project Vision: Accelerating Agricultural Biotechnology Discovery

**G²AENome** aims to be a transformative, multi-modal, generative AI-powered platform designed to accelerate discovery in agricultural biotechnology. This initiative is deeply rooted in the extensive expertise of **Dr. Prashant Kaushik**, a distinguished figure in genomics, molecular biology, bioinformatics, and agricultural innovation. With over 350 publications, 12 books, numerous patents, and a significant scholarly impact (Google Scholar h-index 35, i10-index 125), Dr. Kaushik's work provides a rich foundation for this project. His research spans critical areas such as RNA sequencing, biomarker analysis, and studies on vital crops like eggplant, tomato, and rice, particularly concerning stress response and agricultural improvement.

The G²AENome platform envisions a future where researchers can rapidly synthesize information from vast datasets—genomic sequences, phenomic data, and the entire corpus of scientific literature—to uncover novel insights, predict gene function, identify stress-resilient traits, and ultimately contribute to global food security and sustainable agriculture.

This repository, **`g2aenome_dr_kaushik_adv_poc`**, represents an **Advanced Proof-of-Concept (PoC)**. It lays the foundational software architecture for the G²AENome platform, demonstrating core capabilities in data processing, knowledge extraction, and intelligent querying.

## This Demonstrable Core (Advanced Proof-of-Concept)

This Advanced PoC showcases the following key components and capabilities:

1.  **Multi-Modal Data Fabric:**
    *   Ingestion and processing of Dr. Kaushik's publications (PDFs) to extract textual knowledge.
    *   Parsing and structuring of sample genomic data (FASTA, GFF) for crops like eggplant and tomato.

2.  **Advanced Knowledge Services:**
    *   **Entity Recognition:** Utilizes advanced NLP models (e.g., `scispaCy`) to identify key biological and agricultural entities (genes, proteins, crop species, traits, stress factors) from processed text.
    *   **Relation Extraction:** Implements mechanisms to identify and extract relationships between these entities (e.g., gene-regulates-gene, crop-exhibits-trait, stressor-affects-gene), moving beyond simple co-occurrence to more structured relational data.
    *   **Knowledge Graph Construction:** Builds a `networkx`-based knowledge graph, integrating entities and relations from both literature and genomic data. This serves as a structured, queryable representation of domain knowledge.
    *   **Vector Store Management:** Creates a semantic search index (using `sentence-transformers` and `FAISS`) from text chunks, enabling powerful similarity-based retrieval of information from Dr. Kaushik's work.

3.  **Intelligent Agent Core (LangChain-Powered):**
    *   A modular LangChain agent equipped with **custom tools** for:
        *   `SemanticSearchTool`: Querying the vector store for relevant text passages.
        *   `KnowledgeGraphLookupTool`: Querying the knowledge graph for specific entities and their relationships.
    *   The agent can understand complex queries, strategically use its tools (potentially in sequence), and synthesize information from both the KG and vector store to provide comprehensive, cited answers.

4.  **Configurability and Extensibility:**
    *   System behavior, model choices, and paths are managed through **YAML configuration files** (`configs/main_config.yaml`, `configs/models_config.yaml`) and environment variables (`.env.example`).
    *   The codebase is designed with modularity and clear separation of concerns to facilitate future expansion.

5.  **Orchestration and Application:**
    *   A main pipeline script (`src/main_pipeline.py`) to manage data ingestion, processing, KG/vector store creation, and artifact saving.
    *   Integration with `MLflow` for tracking pipeline runs, parameters, and metrics.
    *   A basic `Streamlit` application (`app/streamlit_app.py`) to interact with the G²AENome agent and query the knowledge base.

## Example Usage / Query Scenario

This PoC is designed to handle queries like the following, demonstrating its integrated knowledge retrieval capabilities:

> *"Drawing from Dr. Kaushik's publications on Solanum melongena (eggplant) and its response to abiotic stress, identify key genes involved. For each gene, list its documented functions from the knowledge graph and provide summaries of relevant text segments from his papers retrieved via semantic search."*

The G²AENome agent would:
1.  Use semantic search to find relevant sections in Dr. Kaushik's papers discussing *Solanum melongena* and abiotic stress.
2.  Extract gene names (entities) from these sections.
3.  Query the knowledge graph for these genes to find their known functions and relationships (e.g., involvement in stress response pathways, interactions with other genes/proteins).
4.  Synthesize this information, providing a list of genes, their functions (from KG), and supporting evidence (cited text snippets from semantic search).

## Repository Structure

```
g2aenome_dr_kaushik_adv_poc/
├── .env.example              # Template for environment variables
├── .github/                  # CI/CD workflows
│   └── workflows/
│       └── python-ci.yml     # Basic GitHub Actions CI
├── configs/                  # YAML configuration files
│   ├── main_config.yaml
│   └── models_config.yaml
├── data/                     # Sample data
│   ├── publications_dr_kaushik/ # Dr. Kaushik's sample PDFs
│   └── genomic_data_samples/  # Sample FASTA, GFF files
├── notebooks/                # Jupyter notebooks for exploration
├── src/                      # Source code
│   ├── common/               # Shared utilities (e.g., logging)
│   ├── data_fabric/          # Data ingestion and processing
│   ├── knowledge_services/   # NER, RE, KG, Vector Store
│   ├── agent_core/           # LangChain agent, tools, prompts
│   └── main_pipeline.py      # Main data processing script
├── experiments/              # MLflow outputs (if not configured elsewhere)
├── tests/                    # Pytest unit tests
├── app/                      # Streamlit/FastAPI application
│   └── streamlit_app.py
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Basic setup script
├── .gitignore
└── LICENSE
```

## Getting Started

### Prerequisites

*   Python 3.8+
*   Access to Dr. Kaushik's sample publications (PDFs) and relevant genomic data (FASTA, GFF).
*   API keys (e.g., OpenAI, HuggingFace) if using proprietary models (optional, configure in `.env`).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/g2aenome_dr_kaushik_adv_poc.git # Replace with actual URL
    cd g2aenome_dr_kaushik_adv_poc
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will install core libraries. Specific spaCy/scispaCy models or large transformer models might need separate download steps as indicated in `configs/models_config.yaml` or during their first use.*
    For example, to download the default `scispaCy` model:
    ```bash
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz
    ```


4.  **Set up environment variables:**
    Copy `.env.example` to `.env` and fill in any necessary API keys or configuration:
    ```bash
    cp .env.example .env
    # Edit .env with your details
    ```

5.  **Place sample data:**
    *   Put sample PDF publications from Dr. Kaushik into the `data/publications_dr_kaushik/` directory.
    *   Place sample FASTA and GFF files (e.g., for eggplant, tomato) into the `data/genomic_data_samples/` directory.

### Running the Core Pipeline

This script processes the data, builds the knowledge graph, and creates the vector store.

```bash
python src/main_pipeline.py
```
This run will be tracked by MLflow (by default, results will be in a local `mlruns` directory).

### Running the Streamlit Application

After running the main pipeline to generate the KG and vector store artifacts:

```bash
streamlit run app/streamlit_app.py
```
This will launch a web interface where you can interact with the G²AENome agent.

## Development and Testing

*   **Configuration:** Modify `configs/main_config.yaml` and `configs/models_config.yaml` to change paths, models, and other parameters.
*   **Notebooks:** Use Jupyter notebooks in the `notebooks/` directory for experimentation and component testing.
*   **Testing:** Run unit tests using pytest:
    ```bash
    pytest tests/
    ```

## Future Directions

This Advanced PoC is a stepping stone towards the full G²AENome vision. Future enhancements could include:

*   Integration with persistent graph databases (e.g., Neo4j, Amazon Neptune).
*   Support for a wider range of genomic and phenomic data types.
*   More sophisticated Relation Extraction models.
*   Advanced agent reasoning capabilities and multi-step tool usage.
*   User authentication and collaborative features.
*   Deployment to cloud platforms.
*   Integration of more specialized bioinformatics tools and databases.

## Contribution

This project is currently in its foundational stages. Contributions and collaborations aligned with Dr. Kaushik's research vision are welcome as the project matures.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project leverages the extensive body of work and expertise of Dr. Prashant Kaushik in agricultural biotechnology. It also builds upon the capabilities of numerous open-source libraries and the LangChain framework.
