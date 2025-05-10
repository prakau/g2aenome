import yaml
from pathlib import Path
from loguru import logger
import mlflow
import time
import os

# Import project modules
from common.logging_config import setup_logging, PROJECT_ROOT
from data_fabric.pdf_processor import PDFProcessor
from data_fabric.genomic_processor import GenomicProcessor
from knowledge_services.entity_recognition import EntityRecognizer
from knowledge_services.relation_extraction import RelationExtractor
from knowledge_services.knowledge_graph_manager import KnowledgeGraphManager
from knowledge_services.vector_store_manager import VectorStoreManager

# Load configurations
CONFIGS_DIR = PROJECT_ROOT / "configs"
MAIN_CONFIG_PATH = CONFIGS_DIR / "main_config.yaml"
MODELS_CONFIG_PATH = CONFIGS_DIR / "models_config.yaml"

def load_config_file(path: Path) -> dict:
    if not path.exists():
        logger.error(f"Configuration file not found: {path}")
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Setup Logging (call this first)
    # The logger instance returned by setup_logging can be used, or loguru can be imported directly.
    # For simplicity, other modules will import loguru directly after it's configured here.
    setup_logging(config_path=MAIN_CONFIG_PATH)
    logger.info("G²AENome Main Pipeline Started")

    # 2. Load Configurations
    logger.info("Loading configurations...")
    try:
        main_config = load_config_file(MAIN_CONFIG_PATH)
        models_config = load_config_file(MODELS_CONFIG_PATH) # models_config is loaded by individual services as needed
    except FileNotFoundError:
        logger.critical("Core configuration files missing. Exiting pipeline.")
        return

    # Extract relevant paths from main_config
    data_base_dir = PROJECT_ROOT / main_config.get("data_base_dir", "data")
    publications_path = data_base_dir / main_config.get("publications_dir", "publications_dr_kaushik")
    genomic_data_path = data_base_dir / main_config.get("genomic_data_dir", "genomic_data_samples")
    
    output_base_dir = PROJECT_ROOT / main_config.get("output_base_dir", "outputs")
    output_base_dir.mkdir(parents=True, exist_ok=True) # Ensure output base dir exists

    # MLflow setup
    mlflow_experiment_name = main_config.get("mlflow_tracking", {}).get("experiment_name", "G2Aenome_Advanced_PoC_Pipeline")
    mlflow.set_experiment(mlflow_experiment_name)
    
    # Check for MLFLOW_TRACKING_URI in environment, otherwise it defaults to local ./mlruns
    if os.getenv("MLFLOW_TRACKING_URI"):
        logger.info(f"Using MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    else:
        logger.info(f"MLFLOW_TRACKING_URI not set. Using default local tracking (./mlruns).")


    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        mlflow.log_param("run_id", run_id)
        
        # Log config files as artifacts
        mlflow.log_artifact(str(MAIN_CONFIG_PATH), artifact_path="configs")
        mlflow.log_artifact(str(MODELS_CONFIG_PATH), artifact_path="configs")

        # --- 3. Initialize Processors and Services ---
        logger.info("Initializing data processors and knowledge services...")
        pdf_processor = PDFProcessor(config=main_config) # Pass full main_config
        genomic_processor = GenomicProcessor(config=main_config)
        
        # Services need paths to model configs
        entity_recognizer = EntityRecognizer(models_config_path=str(MODELS_CONFIG_PATH))
        relation_extractor = RelationExtractor(models_config_path=str(MODELS_CONFIG_PATH)) # Uses NER's spacy model by default
        
        # Managers need paths to main_config for their output locations
        kg_manager = KnowledgeGraphManager(main_config_path=str(MAIN_CONFIG_PATH))
        vector_store_manager = VectorStoreManager(main_config_path=str(MAIN_CONFIG_PATH), models_config_path=str(MODELS_CONFIG_PATH))

        # --- 4. Data Ingestion and Processing (Data Fabric) ---
        start_time_data_fabric = time.time()
        
        # Process PDFs
        logger.info(f"Processing PDFs from: {publications_path}")
        all_text_chunks = pdf_processor.process_all_pdfs_in_directory(str(publications_path))
        num_pdfs_processed = len(set(chunk['source_document'] for chunk in all_text_chunks)) # Approximate
        num_text_chunks = len(all_text_chunks)
        logger.info(f"PDF Processing: {num_pdfs_processed} PDFs processed, {num_text_chunks} text chunks generated.")
        mlflow.log_metric("num_pdfs_processed", num_pdfs_processed)
        mlflow.log_metric("num_text_chunks_from_pdf", num_text_chunks)

        # Process Genomic Data
        logger.info(f"Processing FASTA files from: {genomic_data_path}")
        fasta_data = genomic_processor.parse_fasta_files(str(genomic_data_path))
        logger.info(f"Genomic Processing: {len(fasta_data)} FASTA records parsed.")
        mlflow.log_metric("num_fasta_records_parsed", len(fasta_data))

        logger.info(f"Processing GFF files from: {genomic_data_path}")
        gff_data = genomic_processor.parse_gff_files(str(genomic_data_path))
        logger.info(f"Genomic Processing: {len(gff_data)} GFF features processed.")
        mlflow.log_metric("num_gff_features_processed", len(gff_data))
        
        mlflow.log_metric("data_fabric_duration_sec", time.time() - start_time_data_fabric)

        # --- 5. Knowledge Extraction (Knowledge Services) ---
        start_time_knowledge_services = time.time()

        # Entity Recognition
        logger.info("Extracting entities from text chunks...")
        # We need to map entities to their chunks for relation extraction
        entities_by_chunk_id = {} # Store entities grouped by their chunk_id
        all_extracted_entities = [] # Flat list of all entities

        if all_text_chunks:
            # EntityRecognizer expects List[Dict] where each dict has 'text', 'source_document', etc.
            extracted_entities_tuples = entity_recognizer.extract_entities(all_text_chunks)
            all_extracted_entities = extracted_entities_tuples # Keep the flat list for KG building
            
            for ent_tuple in extracted_entities_tuples:
                # ent_tuple: (text, label, start, end, chunk_id, source_doc, page_num)
                chunk_id = ent_tuple[4]
                if chunk_id not in entities_by_chunk_id:
                    entities_by_chunk_id[chunk_id] = []
                entities_by_chunk_id[chunk_id].append(ent_tuple)
            
            logger.info(f"Entity Recognition: {len(all_extracted_entities)} entities extracted.")
            mlflow.log_metric("num_entities_extracted", len(all_extracted_entities))
            # TODO: Log entity types counts if desired
        else:
            logger.warning("No text chunks available for entity recognition.")
            mlflow.log_metric("num_entities_extracted", 0)

        # Relation Extraction
        logger.info("Extracting relations from text chunks...")
        if all_text_chunks and entities_by_chunk_id:
            extracted_relations = relation_extractor.extract_relations(all_text_chunks, entities_by_chunk_id)
            logger.info(f"Relation Extraction: {len(extracted_relations)} relations extracted.")
            mlflow.log_metric("num_relations_extracted", len(extracted_relations))
            # TODO: Log relation types counts if desired
        else:
            logger.warning("No text chunks or entities available for relation extraction.")
            extracted_relations = []
            mlflow.log_metric("num_relations_extracted", 0)
            
        mlflow.log_metric("knowledge_extraction_duration_sec", time.time() - start_time_knowledge_services)

        # --- 6. Build/Update Knowledge Graph ---
        start_time_kg = time.time()
        logger.info("Building/Updating Knowledge Graph...")
        kg_manager.load_graph() # Load existing graph if any, or starts fresh
        kg_manager.build_graph_from_entities_relations(all_extracted_entities, extracted_relations)
        kg_manager.add_genomic_data_to_graph(fasta_data, data_type="fasta_record")
        kg_manager.add_genomic_data_to_graph(gff_data, data_type="gff_feature")
        kg_manager.save_graph()
        logger.info(f"Knowledge Graph updated and saved. Nodes: {kg_manager.graph.number_of_nodes()}, Edges: {kg_manager.graph.number_of_edges()}")
        mlflow.log_metric("kg_num_nodes", kg_manager.graph.number_of_nodes())
        mlflow.log_metric("kg_num_edges", kg_manager.graph.number_of_edges())
        mlflow.log_artifact(str(kg_manager.graph_file_path), artifact_path="knowledge_graph")
        mlflow.log_metric("kg_build_duration_sec", time.time() - start_time_kg)

        # --- 7. Build/Update Vector Store ---
        start_time_vs = time.time()
        logger.info("Building/Updating Vector Store...")
        # VectorStoreManager handles loading or creating. Pass force_recreate=True if always rebuilding.
        vector_store_manager.create_or_load_vector_store(all_text_chunks, force_recreate=True)
        if vector_store_manager.index:
            logger.info(f"Vector Store updated and saved. Index size: {vector_store_manager.index.ntotal} vectors.")
            mlflow.log_metric("vector_store_num_vectors", vector_store_manager.index.ntotal)
            mlflow.log_artifact(str(vector_store_manager.index_file_path), artifact_path="vector_store")
            mlflow.log_artifact(str(vector_store_manager.metadata_file_path), artifact_path="vector_store")
        else:
            logger.error("Vector Store index is None after creation/loading attempt.")
            mlflow.log_metric("vector_store_num_vectors", 0)
        mlflow.log_metric("vector_store_build_duration_sec", time.time() - start_time_vs)

        # --- 8. (Optional) Agent Evaluation Placeholder ---
        # This would involve a predefined set of Q&A pairs to test the agent.
        # For now, this is a placeholder.
        # logger.info("Agent evaluation placeholder...")
        # agent = G2AenomeAgent(models_config_path=str(MODELS_CONFIG_PATH), 
        #                        knowledge_graph_manager=kg_manager, 
        #                        vector_store_manager=vector_store_manager)
        # if agent.agent_executor:
        #     # Define some test queries
        #     test_queries = [
        #         "What are the functions of SmMYB1 in eggplant?",
        #         "Summarize Dr. Kaushik's work on abiotic stress in rice."
        #     ]
        #     for i, t_query in enumerate(test_queries):
        #         response = agent.ask_question(t_query)
        #         mlflow.log_text(response.get('output', 'No output'), f"agent_response_query_{i+1}.txt")
        #         # Manual or LLM-based scoring would go here
        #         # mlflow.log_metric(f"query_{i+1}_relevance_score", 0.9) # Example
        # else:
        #     logger.error("Agent could not be initialized for evaluation.")
        
        logger.info("G²AENome Main Pipeline Finished Successfully.")
        mlflow.log_param("pipeline_status", "success")

if __name__ == "__main__":
    main()
