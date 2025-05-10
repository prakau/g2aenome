import spacy
from spacy.tokens import Doc
from typing import List, Dict, Tuple, Any, Optional # Added Optional
from loguru import logger
import yaml
from pathlib import Path

# from ..common.logging_config import setup_logging # Assuming logger is set up globally
# setup_logging()

class EntityRecognizer:
    """
    Recognizes entities in text chunks using a spaCy model.
    """

    def __init__(self, models_config_path: str = None, main_config_path: str = None):
        """
        Initializes the EntityRecognizer.
        Args:
            models_config_path (str, optional): Path to the models configuration YAML file.
            main_config_path (str, optional): Path to the main configuration YAML file (not directly used here but good for consistency).
        """
        self.nlp = None
        self.model_name = None
        self.target_labels = None
        self._load_config(models_config_path)
        self.load_model()

    def _load_config(self, config_path: str = None):
        """Loads NER model name and target labels from the models_config.yaml."""
        if config_path is None:
            # Assuming this file is in src/knowledge_services/entity_recognition.py
            project_root = Path(__file__).resolve().parent.parent.parent
            config_path = project_root / "configs" / "models_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                models_config = yaml.safe_load(f)
            
            self.model_name = models_config.get("entity_recognition", {}).get("spacy_model_name", "en_core_web_sm")
            self.target_labels = models_config.get("entity_recognition", {}).get("target_entity_labels", [])
            logger.info(f"EntityRecognizer configured. Model: {self.model_name}, Target Labels: {self.target_labels if self.target_labels else 'ALL'}")

        except FileNotFoundError:
            logger.error(f"Models configuration file not found at {config_path}. Using default spaCy model (en_core_web_sm).")
            self.model_name = "en_core_web_sm"
            self.target_labels = []
        except Exception as e:
            logger.error(f"Error loading models configuration from {config_path}: {e}. Using default spaCy model (en_core_web_sm).")
            self.model_name = "en_core_web_sm"
            self.target_labels = []


    def load_model(self):
        """
        Loads the spaCy NLP model specified in the configuration.
        """
        if self.nlp is None and self.model_name:
            try:
                logger.info(f"Loading spaCy model: {self.model_name}...")
                self.nlp = spacy.load(self.model_name)
                logger.info(f"Successfully loaded spaCy model: {self.model_name}")
            except OSError:
                logger.error(f"Could not load spaCy model '{self.model_name}'. "
                             f"Make sure it's downloaded (e.g., python -m spacy download {self.model_name}).")
                # Fallback or raise error
                logger.warning("Falling back to en_core_web_sm. Please install the configured model for better performance.")
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    self.model_name = "en_core_web_sm" # Update model_name if fallback occurs
                    logger.info("Successfully loaded fallback spaCy model: en_core_web_sm")
                except OSError:
                    logger.critical("Fallback model en_core_web_sm also not found. NER will not function.")
                    self.nlp = None # Ensure nlp is None if loading fails critically
            except Exception as e:
                logger.critical(f"An unexpected error occurred while loading spaCy model {self.model_name}: {e}")
                self.nlp = None


    def extract_entities(self, text_chunks_with_metadata: List[Dict[str, Any]]) -> List[Tuple[str, str, int, int, str, str, Optional[int]]]:
        """
        Processes a list of text chunks (with metadata) and extracts entities.
        Args:
            text_chunks_with_metadata (List[Dict[str, Any]]): A list of dictionaries,
                where each dict contains at least 'text' and 'source_document'.
                Optionally 'page_number' and 'chunk_id'.
        Returns:
            List[Tuple[str, str, int, int, str, str, Optional[int]]]]: A list of tuples, where each tuple is
            (entity_text, entity_label, start_char_in_chunk, end_char_in_chunk,
             chunk_id, source_document, page_number (or None)).
        """
        if self.nlp is None:
            logger.error("spaCy NLP model not loaded. Cannot extract entities.")
            return []

        all_entities = []
        
        # Prepare texts for batch processing with nlp.pipe for efficiency
        texts = [chunk_data['text'] for chunk_data in text_chunks_with_metadata if chunk_data.get('text')]
        chunk_metadata_list = [chunk_data for chunk_data in text_chunks_with_metadata if chunk_data.get('text')]

        if not texts:
            logger.warning("No text provided in chunks to extract_entities.")
            return []

        logger.info(f"Processing {len(texts)} text chunks for entity recognition...")
        
        # Using nlp.pipe for efficient batch processing
        # Consider adjusting n_process and batch_size for very large datasets
        docs = self.nlp.pipe(texts, batch_size=50) # n_process=-1 for using all cores

        for i, doc in enumerate(docs):
            chunk_data = chunk_metadata_list[i]
            chunk_id = chunk_data.get('chunk_id', f"chunk_{i}")
            source_document = chunk_data.get('source_document', 'unknown_source')
            page_number = chunk_data.get('page_number') # Can be None

            for ent in doc.ents:
                # Filter by target labels if specified
                if not self.target_labels or ent.label_ in self.target_labels:
                    all_entities.append((
                        ent.text,
                        ent.label_,
                        ent.start_char,
                        ent.end_char,
                        chunk_id,
                        source_document,
                        page_number
                    ))
        
        logger.info(f"Extracted {len(all_entities)} entities from {len(texts)} chunks.")
        return all_entities

if __name__ == '__main__':
    import sys
    # Example usage of EntityRecognizer
    logger.remove() # Remove default handlers, if any
    logger.add(sys.stderr, level="DEBUG") # Add a simple handler for the example

    # Assume project root is two levels up from src/knowledge_services
    project_root_example = Path(__file__).resolve().parent.parent.parent
    models_cfg_path = project_root_example / "configs" / "models_config.yaml"

    if not models_cfg_path.exists():
        logger.error(f"Models config not found at {models_cfg_path} for example. Ensure it exists.")
        # Create a dummy models_config.yaml for the example to run if it doesn't exist
        models_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_models_config = {
            "entity_recognition": {
                "spacy_model_name": "en_core_web_sm", # Small model for quick testing
                "target_entity_labels": ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
            }
        }
        with open(models_cfg_path, 'w') as f:
            yaml.dump(dummy_models_config, f)
        logger.info(f"Created dummy models_config.yaml at {models_cfg_path} with en_core_web_sm.")


    recognizer = EntityRecognizer(models_config_path=str(models_cfg_path))

    if recognizer.nlp is None:
        logger.error("Failed to initialize EntityRecognizer or load spaCy model. Exiting example.")
        sys.exit(1)

    sample_text_chunks = [
        {
            "text": "Dr. Prashant Kaushik published a study on Solanum melongena (eggplant) in 2023. "
                    "The study focused on abiotic stress response, particularly drought tolerance. "
                    "Key genes like SmMYB1 and SmNAC1 were identified. The research was conducted at ICAR.",
            "source_document": "Kaushik_et_al_2023.pdf",
            "page_number": 1,
            "chunk_id": "Kaushik_et_al_2023_p1_c1"
        },
        {
            "text": "Further analysis of Oryza sativa (rice) showed similar stress response mechanisms. "
                    "The experiments involved RNA sequencing and were performed in New Delhi.",
            "source_document": "Another_Paper_2024.pdf",
            "page_number": 5,
            "chunk_id": "Another_Paper_2024_p5_c1"
        }
    ]

    logger.info("\nExtracting entities from sample text chunks...")
    extracted_entities = recognizer.extract_entities(sample_text_chunks)

    if extracted_entities:
        logger.info(f"Found {len(extracted_entities)} entities:")
        for entity in extracted_entities:
            logger.info(
                f"  Text: '{entity[0]}', Label: {entity[1]}, Start: {entity[2]}, End: {entity[3]}, "
                f"Chunk: {entity[4]}, Doc: {entity[5]}, Page: {entity[6] if entity[6] is not None else 'N/A'}"
            )
    else:
        logger.warning("No entities extracted from the sample text.")
        if recognizer.target_labels:
             logger.warning(f"Current target labels are: {recognizer.target_labels}. "
                            "If using en_core_web_sm, these might not match its default entity types.")
             logger.warning("For biomedical entities, ensure a scispaCy model is configured and installed.")


    # Example with specific biomedical labels (requires a scispaCy model like en_core_sci_lg)
    # To run this, you'd need to:
    # 1. Modify configs/models_config.yaml to use "en_core_sci_lg" or similar.
    # 2. Add relevant biomedical target_entity_labels.
    # 3. Download the model: python -m spacy download en_core_sci_lg
    
    # logger.info("\n--- Example with scispaCy (if configured) ---")
    # biomedical_config = {
    #     "entity_recognition": {
    #         "spacy_model_name": "en_core_sci_lg", # Make sure this is installed
    #         "target_entity_labels": ["GENE_OR_GENE_PRODUCT", "CHEMICAL", "ORGANISM", "CELL_LINE"] 
    #     }
    # }
    # # Create a temporary config file for this specific example
    # temp_scispacy_config_path = project_root_example / "configs" / "temp_scispacy_models_config.yaml"
    # with open(temp_scispacy_config_path, 'w') as f:
    #     yaml.dump(biomedical_config, f)
    
    # try:
    #     sci_recognizer = EntityRecognizer(models_config_path=str(temp_scispacy_config_path))
    #     if sci_recognizer.nlp:
    #         sci_entities = sci_recognizer.extract_entities(sample_text_chunks)
    #         if sci_entities:
    #             logger.info(f"Found {len(sci_entities)} biomedical entities:")
    #             for entity in sci_entities:
    #                 logger.info(f"  Sci-Text: '{entity[0]}', Label: {entity[1]}")
    #         else:
    #             logger.warning("No biomedical entities extracted with scispaCy model from sample.")
    #     else:
    #         logger.warning("Could not load scispaCy model for the biomedical example. Ensure it's installed and configured.")
    # finally:
    #     if temp_scispacy_config_path.exists():
    #         temp_scispacy_config_path.unlink() # Clean up temp config

    logger.info("EntityRecognizer example usage finished.")
