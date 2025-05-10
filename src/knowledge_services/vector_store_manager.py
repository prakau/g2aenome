from typing import List, Dict, Any, Optional
from loguru import logger
import yaml
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss # For FAISS vector store

# from ..common.logging_config import setup_logging
# setup_logging()

class VectorStoreManager:
    """
    Manages the creation, loading, and searching of a vector store (e.g., FAISS)
    for semantic search over text chunks.
    """

    def __init__(self, main_config_path: str = None, models_config_path: str = None):
        """
        Initializes the VectorStoreManager.
        Args:
            main_config_path (str, optional): Path to the main configuration YAML file.
            models_config_path (str, optional): Path to the models configuration YAML file.
        """
        self.main_config = {}
        self.models_config = {}
        self._load_configs(main_config_path, models_config_path)

        self.embedding_model_name = self.models_config.get("embedding_model", {}).get("hf_model_name", "sentence-transformers/all-mpnet-base-v2")
        self.embedding_dim = self.models_config.get("embedding_model", {}).get("embedding_dim", 768) # Default for all-mpnet-base-v2
        self.embedding_model = None # Loaded on demand

        self.output_dir = Path(self.main_config.get("output_base_dir", "outputs")) / \
                          Path(self.main_config.get("vector_store_dir", "vector_store"))
        self.index_file_name = self.main_config.get("vector_store", {}).get("index_file_name", "faiss_index.idx")
        self.index_file_path = self.output_dir / self.index_file_name
        self.metadata_file_path = self.output_dir / f"{self.index_file_name}.meta.pkl" # For storing metadata associated with vectors

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.index = None # FAISS index
        self.chunk_metadata_store = [] # List to store metadata for each vector by its index ID

        logger.info(f"VectorStoreManager initialized. Embedding model: {self.embedding_model_name}, Dim: {self.embedding_dim}")
        logger.info(f"Vector store index will be at: {self.index_file_path}")

    def _load_configs(self, main_config_path: str = None, models_config_path: str = None):
        """Loads configurations from YAML files."""
        project_root = Path(__file__).resolve().parent.parent.parent
        
        if main_config_path is None:
            main_config_path = project_root / "configs" / "main_config.yaml"
        if models_config_path is None:
            models_config_path = project_root / "configs" / "models_config.yaml"

        try:
            with open(main_config_path, 'r') as f:
                self.main_config = yaml.safe_load(f)
            logger.info("VectorStoreManager: Main configuration loaded.")
        except FileNotFoundError:
            logger.error(f"Main configuration file not found at {main_config_path} for VectorStoreManager.")
        except Exception as e:
            logger.error(f"Error loading main configuration from {main_config_path}: {e}")

        try:
            with open(models_config_path, 'r') as f:
                self.models_config = yaml.safe_load(f)
            logger.info("VectorStoreManager: Models configuration loaded.")
        except FileNotFoundError:
            logger.error(f"Models configuration file not found at {models_config_path} for VectorStoreManager.")
        except Exception as e:
            logger.error(f"Error loading models configuration from {models_config_path}: {e}")


    def load_embedding_model(self):
        """Loads the sentence-transformer model specified in the configuration."""
        if self.embedding_model is None:
            try:
                logger.info(f"Loading sentence-transformer model: {self.embedding_model_name}...")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                # Verify embedding dimension if possible (some models don't expose it easily)
                # test_emb_dim = self.embedding_model.get_sentence_embedding_dimension()
                # if test_emb_dim != self.embedding_dim:
                #    logger.warning(f"Configured embedding_dim ({self.embedding_dim}) does not match model's reported dim ({test_emb_dim}). Using model's dim.")
                #    self.embedding_dim = test_emb_dim
                logger.info(f"Successfully loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.critical(f"Failed to load sentence-transformer model {self.embedding_model_name}: {e}")
                self.embedding_model = None
        return self.embedding_model

    def create_or_load_vector_store(self, text_chunks_with_metadata: Optional[List[Dict[str, Any]]] = None, force_recreate: bool = False):
        """
        Creates a new FAISS vector store from text chunks or loads an existing one from disk.
        Args:
            text_chunks_with_metadata (Optional[List[Dict[str, Any]]]):
                List of dictionaries, each containing 'text' and other metadata
                (e.g., 'source_document', 'page_number', 'chunk_id').
                Required if creating a new store or force_recreate is True.
            force_recreate (bool): If True, always create a new store even if one exists on disk.
        """
        if not force_recreate and self.index_file_path.exists() and self.metadata_file_path.exists():
            try:
                logger.info(f"Loading existing FAISS index from {self.index_file_path}")
                self.index = faiss.read_index(str(self.index_file_path))
                
                import pickle
                with open(self.metadata_file_path, "rb") as f_meta:
                    self.chunk_metadata_store = pickle.load(f_meta)
                
                logger.info(f"Successfully loaded vector store. Index size: {self.index.ntotal} vectors. Metadata items: {len(self.chunk_metadata_store)}")
                # Ensure embedding model is loaded if we plan to search later
                if self.embedding_model is None: self.load_embedding_model()
                return
            except Exception as e:
                logger.error(f"Failed to load existing vector store from {self.index_file_path}: {e}. Will attempt to recreate.")
        
        if text_chunks_with_metadata is None and force_recreate:
             logger.error("Cannot recreate vector store: text_chunks_with_metadata is None but force_recreate is True.")
             return
        if text_chunks_with_metadata is None:
            logger.info("No existing vector store found and no new data provided. Initializing empty store.")
            # Initialize an empty index if no data and no existing store
            self.index = faiss.IndexFlatL2(self.embedding_dim) # Basic L2 distance index
            # For larger datasets, consider IndexIVFFlat or other more advanced FAISS indexes.
            # self.index = faiss.IndexIDMap2(self.index) # To map to original IDs, useful if not sequential
            self.chunk_metadata_store = []
            return


        if self.embedding_model is None:
            if not self.load_embedding_model(): # Try to load the model
                logger.error("Embedding model could not be loaded. Cannot create vector store.")
                return

        logger.info(f"Creating new FAISS vector store. Processing {len(text_chunks_with_metadata)} text chunks.")
        
        texts_to_embed = [chunk['text'] for chunk in text_chunks_with_metadata if chunk.get('text')]
        valid_metadata = [chunk for chunk in text_chunks_with_metadata if chunk.get('text')]

        if not texts_to_embed:
            logger.warning("No valid text found in chunks to create embeddings.")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.chunk_metadata_store = []
            return

        embeddings = self.embedding_model.encode(texts_to_embed, show_progress_bar=True)
        embeddings_np = np.array(embeddings).astype('float32') # FAISS expects float32
        
        # Ensure embedding dimension matches
        if embeddings_np.shape[1] != self.embedding_dim:
            logger.warning(f"Actual embedding dimension ({embeddings_np.shape[1]}) differs from configured ({self.embedding_dim}). Updating config.")
            self.embedding_dim = embeddings_np.shape[1]

        self.index = faiss.IndexFlatL2(self.embedding_dim)
        # self.index = faiss.IndexIDMap2(self.index) # Optional: if you want to use non-sequential IDs later

        self.index.add(embeddings_np)
        self.chunk_metadata_store = valid_metadata # Store corresponding metadata

        logger.info(f"FAISS index created with {self.index.ntotal} vectors.")
        self.save_vector_store()


    def save_vector_store(self):
        """Saves the FAISS index and associated metadata to disk."""
        if self.index is None:
            logger.warning("No FAISS index to save.")
            return

        try:
            logger.info(f"Saving FAISS index to {self.index_file_path}")
            faiss.write_index(self.index, str(self.index_file_path))
            
            import pickle
            with open(self.metadata_file_path, "wb") as f_meta:
                pickle.dump(self.chunk_metadata_store, f_meta)
            
            logger.info("Vector store (index and metadata) saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")


    def search_vector_store(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the vector store for the top-k most similar text chunks to the query.
        Args:
            query_text (str): The natural language query.
            k (int): The number of top similar chunks to retrieve.
        Returns:
            List[Dict[str, Any]]: A list of top-k matching chunks, each a dictionary
                                  containing the chunk's metadata and similarity score.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is not initialized or is empty. Cannot perform search.")
            return []
        
        if self.embedding_model is None:
            if not self.load_embedding_model():
                logger.error("Embedding model not loaded. Cannot perform search.")
                return []

        logger.info(f"Searching vector store for query: '{query_text[:100]}...' (top {k})")
        query_embedding = self.embedding_model.encode([query_text])
        query_embedding_np = np.array(query_embedding).astype('float32')

        distances, indices = self.index.search(query_embedding_np, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < 0 or idx >= len(self.chunk_metadata_store): # faiss can return -1 if not enough neighbors
                logger.warning(f"Invalid index {idx} returned by FAISS search. Skipping.")
                continue

            metadata = self.chunk_metadata_store[idx]
            # FAISS L2 returns squared L2 distances. Convert to similarity if needed, or just use distance.
            # For cosine similarity with SentenceTransformers, it's often better to normalize embeddings
            # and use IndexFlatIP (Inner Product). For L2, smaller distance is better.
            result_item = {
                "text": metadata.get("text"),
                "source_document": metadata.get("source_document"),
                "page_number": metadata.get("page_number"),
                "chunk_id": metadata.get("chunk_id"),
                "start_char_offset": metadata.get("start_char_offset"),
                "end_char_offset": metadata.get("end_char_offset"),
                "score_type": "L2_distance", # Or "cosine_similarity" if using IndexFlatIP
                "score": float(distances[0][i]) 
            }
            results.append(result_item)
            
        logger.info(f"Found {len(results)} results for the query.")
        return results

if __name__ == '__main__':
    import sys
    import pickle # For metadata_file_path in example

    # Example usage of VectorStoreManager
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    project_root_example = Path(__file__).resolve().parent.parent.parent
    main_cfg_path = project_root_example / "configs" / "main_config.yaml"
    models_cfg_path = project_root_example / "configs" / "models_config.yaml"

    # Ensure configs exist for the example
    if not main_cfg_path.exists():
        main_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_main_config = {
            "output_base_dir": "outputs_vs_example",
            "vector_store_dir": "vs_store",
            "vector_store": {"index_file_name": "example_vs.idx", "default_top_k": 3}
        }
        with open(main_cfg_path, 'w') as f: yaml.dump(dummy_main_config, f)
        logger.info(f"Created dummy main_config.yaml for VS example at {main_cfg_path}")

    if not models_cfg_path.exists():
        models_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_models_config = {
            "embedding_model": {
                "hf_model_name": "sentence-transformers/all-MiniLM-L6-v2", # Smaller model for faster testing
                "embedding_dim": 384 
            }
        }
        with open(models_cfg_path, 'w') as f: yaml.dump(dummy_models_config, f)
        logger.info(f"Created dummy models_config.yaml for VS example at {models_cfg_path}")

    vs_manager = VectorStoreManager(main_config_path=str(main_cfg_path), models_config_path=str(models_cfg_path))

    # Sample text chunks (e.g., from PDFProcessor)
    sample_chunks = [
        {"text": "Eggplant (Solanum melongena) is a species of nightshade grown for its edible fruit.", 
         "source_document": "doc1.pdf", "page_number": 1, "chunk_id": "doc1_p1_c1"},
        {"text": "Abiotic stress, such as drought, significantly impacts eggplant yield.", 
         "source_document": "doc1.pdf", "page_number": 2, "chunk_id": "doc1_p2_c1"},
        {"text": "Key genes in tomato (Solanum lycopersicum) show homology to those in eggplant.", 
         "source_document": "doc2.pdf", "page_number": 3, "chunk_id": "doc2_p3_c1"},
        {"text": "Rice (Oryza sativa) is a staple food for a large part of the world's human population.", 
         "source_document": "doc3.pdf", "page_number": 1, "chunk_id": "doc3_p1_c1"}
    ]

    logger.info("\n--- Creating or loading vector store ---")
    # Use force_recreate=True to ensure it builds for the example, 
    # or False to test loading if it already exists.
    vs_manager.create_or_load_vector_store(sample_chunks, force_recreate=True) 

    if vs_manager.index and vs_manager.index.ntotal > 0:
        logger.info("\n--- Searching vector store ---")
        query = "impact of drought on eggplant"
        search_results = vs_manager.search_vector_store(query, k=2)
        
        if search_results:
            logger.info(f"Search results for '{query}':")
            for i, res in enumerate(search_results):
                logger.info(f"  Result {i+1}:")
                logger.info(f"    Text: '{res['text'][:100]}...'")
                logger.info(f"    Source: {res['source_document']}, Page: {res['page_number']}")
                logger.info(f"    Score ({res['score_type']}): {res['score']:.4f}")
        else:
            logger.warning("No search results found.")
    else:
        logger.error("Vector store is empty or not loaded. Cannot perform search example.")

    # Clean up dummy config and output if created by this script
    if "outputs_vs_example" in str(main_cfg_path):
        import shutil
        if (project_root_example / "outputs_vs_example").exists():
            shutil.rmtree(project_root_example / "outputs_vs_example")
            logger.info("Cleaned up dummy output directory for VS example.")
        # main_cfg_path.unlink(missing_ok=True)
        # models_cfg_path.unlink(missing_ok=True)
        # logger.info("Cleaned up dummy config files if created by this script.")

    logger.info("VectorStoreManager example usage finished.")
