from typing import List, Dict, Tuple, Any, Set, Optional # Added Optional
from loguru import logger
import yaml
from pathlib import Path
import spacy # For dependency parsing in rule-based approach
from transformers import pipeline # For Option 2: Transformer-based RE

# from ..common.logging_config import setup_logging
# setup_logging()

class RelationExtractor:
    """
    Extracts relationships between entities in text chunks.
    Offers a rule-based approach and stubs for a transformer-based approach.
    """

    def __init__(self, models_config_path: str = None, spacy_model_name: str = None):
        """
        Initializes the RelationExtractor.
        Args:
            models_config_path (str, optional): Path to the models configuration YAML file.
            spacy_model_name (str, optional): Name of the spaCy model to load for dependency parsing.
                                              If None, it will try to load from models_config.yaml or default.
        """
        self.hf_model_name = None
        self.target_relation_types = []
        self.nlp_spacy = None # For rule-based approach using spaCy's dependency parser
        self.re_pipeline = None # For transformer-based RE

        self._load_config(models_config_path)

        if spacy_model_name: # Override config if explicitly passed
            self.spacy_model_for_rules = spacy_model_name
        elif not hasattr(self, 'spacy_model_for_rules') or not self.spacy_model_for_rules:
             # Fallback if not in config or passed
            self.spacy_model_for_rules = "en_core_web_sm" # Default small model for rules

        self._load_spacy_for_rules()
        self._load_transformer_re_model() # Implement Option 2

    def _load_config(self, config_path: str = None):
        """Loads RE model name and target relation types from the models_config.yaml."""
        if config_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            config_path = project_root / "configs" / "models_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                models_config = yaml.safe_load(f)
            
            re_config = models_config.get("relation_extraction", {})
            self.hf_model_name = re_config.get("hf_model_name")
            self.target_relation_types = re_config.get("target_relation_types", [])
            
            # Also get spaCy model for rule-based from entity_recognition section or a dedicated one
            self.spacy_model_for_rules = models_config.get("entity_recognition", {}).get("spacy_model_name", "en_core_web_sm")

            logger.info(f"RelationExtractor configured. HF Model: {self.hf_model_name if self.hf_model_name else 'N/A (Rule-based default)'}, "
                        f"SpaCy for rules: {self.spacy_model_for_rules}, Target Relations: {self.target_relation_types if self.target_relation_types else 'ALL'}")

        except FileNotFoundError:
            logger.warning(f"Models configuration file not found at {config_path}. Using defaults for RelationExtractor.")
        except Exception as e:
            logger.error(f"Error loading models configuration from {config_path} for RelationExtractor: {e}. Using defaults.")

    def _load_spacy_for_rules(self):
        """Loads the spaCy model for rule-based relation extraction."""
        if self.nlp_spacy is None and self.spacy_model_for_rules:
            try:
                logger.info(f"Loading spaCy model for rule-based RE: {self.spacy_model_for_rules}...")
                self.nlp_spacy = spacy.load(self.spacy_model_for_rules)
                logger.info(f"Successfully loaded spaCy model: {self.spacy_model_for_rules}")
            except OSError:
                logger.error(f"Could not load spaCy model '{self.spacy_model_for_rules}' for rule-based RE. "
                             "Rule-based extraction might be impaired.")
            except Exception as e:
                logger.error(f"Unexpected error loading spaCy model {self.spacy_model_for_rules}: {e}")
                self.nlp_spacy = None


    def _load_transformer_re_model(self):
        """Loads a transformer-based Relation Extraction model (Option 2)."""
        if self.hf_model_name and self.re_pipeline is None:
            try:
                logger.info(f"Loading HuggingFace RE model: {self.hf_model_name}...")
                # The task for relation extraction can vary. Some models might be 'text-classification'
                # (e.g., classifying if a relation exists between entities in a sentence),
                # 'token-classification' (e.g., identifying relation spans), or a custom task.
                # For a generic RE model, "text2text-generation" or a specific RE task might be used.
                # This is a placeholder; the actual task and model compatibility are critical.
                # Many advanced RE models require specific input formatting (e.g., marking entities).
                self.re_pipeline = pipeline(task="text-classification", model=self.hf_model_name, tokenizer=self.hf_model_name)
                # Using "text-classification" as a common pipeline type; actual RE models might need different setup.
                # For example, some RE models might not fit neatly into a standard pipeline task.
                logger.info(f"Successfully loaded HuggingFace RE model/pipeline: {self.hf_model_name}")
                logger.warning("The loaded RE pipeline is generic. Specific input formatting and output parsing will be required in _extract_relations_transformer.")
            except Exception as e:
                logger.error(f"Failed to load HuggingFace RE model {self.hf_model_name}: {e}. Transformer-based RE will be unavailable.")
                self.re_pipeline = None


    def extract_relations(self, text_chunks_with_metadata: List[Dict[str, Any]],
                          entities_by_chunk: Dict[str, List[Tuple[str, str, int, int, str, str, Optional[int]]]]) \
                          -> List[Tuple[str, str, str, str, str, str, Optional[int]]]:
        """
        Extracts relations from text chunks given pre-extracted entities.
        Args:
            text_chunks_with_metadata (List[Dict[str, Any]]): List of chunk data.
                Each dict must have 'text', 'chunk_id', 'source_document'. 'page_number' is optional.
            entities_by_chunk (Dict[str, List[Tuple[str, str, int, int, str, str, Optional[int]]]]): A dictionary mapping chunk_id to a list of its entities.
                Entity tuple format: (entity_text, entity_label, start_char, end_char, chunk_id, source_doc, page_num)
        Returns:
            List[Tuple[str, str, str, str, str, str, Optional[int]]]]: List of relation tuples:
            (subject_entity_text, relation_type, object_entity_text,
             evidence_sentence, chunk_id, source_document, page_number).
        """
        all_relations = []
        
        # Determine which method to use based on configuration/availability
        if self.re_pipeline: # Prioritize transformer model if loaded (Option 2)
           logger.info("Attempting Transformer-based Relation Extraction.")
           all_relations = self._extract_relations_transformer(text_chunks_with_metadata, entities_by_chunk)
           if not all_relations and self.nlp_spacy: # Fallback if transformer yields nothing and rule-based is available
               logger.info("Transformer RE yielded no results. Falling back to Rule-based Relation Extraction.")
               all_relations = self._extract_relations_rule_based(text_chunks_with_metadata, entities_by_chunk)
        elif self.nlp_spacy: # Use rule-based if transformer is not configured/loaded (Option 1)
            logger.info("Using Rule-based Relation Extraction.")
            all_relations = self._extract_relations_rule_based(text_chunks_with_metadata, entities_by_chunk)
        else:
            logger.error("No relation extraction method available (neither Transformer RE model nor spaCy for rules is loaded).")
            return []
            
        logger.info(f"Extracted {len(all_relations)} relations in total (after potential fallback).")
        return all_relations

    def _extract_relations_rule_based(self, text_chunks_with_metadata: List[Dict[str, Any]],
                                      entities_by_chunk: Dict[str, List[Tuple[str, str, int, int, str, str, Optional[int]]]]) \
                                      -> List[Tuple[str, str, str, str, str, str, Optional[int]]]:
        """
        Implements rule-based or co-occurrence relation extraction using spaCy dependency parsing.
        This is a simplified example. More sophisticated rules would be needed for high accuracy.
        """
        relations = []
        if not self.nlp_spacy:
            logger.error("spaCy model for rule-based RE not loaded. Cannot proceed.")
            return []

        # Define some simple patterns for relations (highly domain-specific)
        # Example: GENE/PROTEIN - verb - GENE/PROTEIN/TRAIT/STRESS_FACTOR
        # This needs to be much more robust.
        potential_relation_verbs = {"activate", "inhibit", "regulate", "express", "associate", "cause", "affect", "involve"}
        
        # Entity types that can be subjects or objects
        # This is a very broad example.
        subj_obj_types = {"GENE", "PROTEIN", "CROP_SPECIES", "TRAIT", "PHENOTYPE", "STRESS_FACTOR", "CHEMICAL_COMPOUND"}


        for chunk_data in text_chunks_with_metadata:
            chunk_id = chunk_data['chunk_id']
            text = chunk_data['text']
            source_doc = chunk_data['source_document']
            page_num = chunk_data.get('page_number')

            if chunk_id not in entities_by_chunk or not entities_by_chunk[chunk_id]:
                continue
            
            chunk_entities = entities_by_chunk[chunk_id]
            
            # Process the chunk text with spaCy
            doc = self.nlp_spacy(text)

            # Iterate over sentences in the chunk
            for sent in doc.sents:
                sent_entities = []
                # Find entities within this sentence's span
                for ent_text, ent_label, ent_start, ent_end, _, _, _ in chunk_entities:
                    # Check if entity span (relative to chunk) is within sentence span (relative to chunk)
                    if ent_start >= sent.start_char and ent_end <= sent.end_char:
                        sent_entities.append({'text': ent_text, 'label': ent_label, 
                                              'start': ent_start - sent.start_char, # Adjust to be relative to sentence
                                              'end': ent_end - sent.start_char})
                
                if len(sent_entities) < 2: # Need at least two entities for a binary relation
                    continue

                # Simple co-occurrence within a sentence as a basic "relation"
                # This is very naive and should be expanded with dependency parsing.
                for i in range(len(sent_entities)):
                    for j in range(i + 1, len(sent_entities)):
                        subj = sent_entities[i]
                        obj = sent_entities[j]

                        # Filter by entity types if needed
                        if subj['label'] not in subj_obj_types or obj['label'] not in subj_obj_types:
                            continue
                        
                        # Simplistic: if two relevant entities co-occur, assume "ASSOCIATED_WITH"
                        # More advanced: Check dependency path between entities in `sent`
                        # e.g., find shortest path in dependency graph, look for verb patterns.
                        
                        # Example of a slightly more advanced check (placeholder for real dependency logic):
                        # Find a verb between them in the sentence text
                        verb_found = False
                        # Text between entities (approximate)
                        start_search = min(subj['end'], obj['end'])
                        end_search = max(subj['start'], obj['start'])
                        
                        # This is still very naive. Proper dependency parsing is complex.
                        # For a real system, iterate through tokens in sent.doc, check POS, lemma, and dependency relations.
                        # For example, find tokens that are verbs and connect subj and obj in the dependency tree.
                        
                        # For now, let's just use co-occurrence as a fallback
                        relation_type = "CO_OCCURS_WITH" # Default, very weak relation
                        
                        # A slightly better heuristic: if a known verb is nearby
                        for token in sent: # spaCy sentence object
                            if token.lemma_ in potential_relation_verbs:
                                # Check if this verb is syntactically related to both entities (complex)
                                # For simplicity, if verb is between entities (in text order)
                                if (subj['end'] < token.idx < obj['start']) or \
                                   (obj['end'] < token.idx < subj['start']):
                                    relation_type = f"{token.lemma_.upper()}_candidate" # e.g. REGULATES_candidate
                                    break # Take first verb found

                        # Filter by target relation types if provided in config
                        if self.target_relation_types and relation_type not in self.target_relation_types and "CO_OCCURS_WITH" not in self.target_relation_types:
                            if not any(candidate_type.startswith(verb.upper()) for verb in potential_relation_verbs if candidate_type in self.target_relation_types):
                                continue


                        relations.append((
                            subj['text'],
                            relation_type,
                            obj['text'],
                            sent.text, # Evidence sentence
                            chunk_id,
                            source_doc,
                            page_num
                        ))
        return relations

    def _extract_relations_transformer(self, 
                                       text_chunks_with_metadata: List[Dict[str, Any]],
                                       entities_by_chunk: Dict[str, List[Tuple[str, str, int, int, str, str, Optional[int]]]]) \
                                       -> List[Tuple[str, str, str, str, str, str, Optional[int]]]:
        """ 
        STUB: Implements relation extraction using a pre-trained transformer model.
        This requires significant work to format inputs and parse outputs based on the chosen model.
        """
        logger.warning("Transformer-based relation extraction (_extract_relations_transformer) is a STUB and not fully implemented.")
        relations = []
        if not self.re_pipeline:
            logger.error("Transformer RE pipeline not loaded.")
            return relations

        for chunk_data in text_chunks_with_metadata:
            chunk_id = chunk_data['chunk_id']
            text = chunk_data['text']
            source_doc = chunk_data['source_document']
            page_num = chunk_data.get('page_number')

            if chunk_id not in entities_by_chunk or not entities_by_chunk[chunk_id]:
                continue
            
            chunk_entities = entities_by_chunk[chunk_id]
            
            # Process with spaCy to get sentences if not already sentence-split
            # (or assume chunks are small enough, e.g., sentences themselves)
            doc = self.nlp_spacy(text) # Using spaCy for sentence tokenization here

            for sent in doc.sents:
                sent_entities_details = []
                for ent_text, ent_label, ent_start, ent_end, _, _, _ in chunk_entities:
                    if ent_start >= sent.start_char and ent_end <= sent.end_char:
                        sent_entities_details.append({'text': ent_text, 'label': ent_label, 
                                                      'start_in_sent': ent_start - sent.start_char,
                                                      'end_in_sent': ent_end - sent.start_char})
                
                if len(sent_entities_details) < 2:
                    continue

                # Iterate over pairs of entities in the sentence
                for i in range(len(sent_entities_details)):
                    for j in range(len(sent_entities_details)):
                        if i == j: continue # Don't relate an entity to itself

                        subj_entity = sent_entities_details[i]
                        obj_entity = sent_entities_details[j]

                        # --- Placeholder for model-specific input formatting ---
                        # Many RE models require entities to be marked, e.g., with special tokens
                        # or by providing entity spans.
                        # Example input (highly dependent on model):
                        # "The sentence text where [SUBJ]subj_entity_text[/SUBJ] is related to [OBJ]obj_entity_text[/OBJ]."
                        # Or: {"text": "sentence text", "spans": [(subj_start, subj_end, "SUBJ_TYPE"), (obj_start, obj_end, "OBJ_TYPE")]}
                        
                        # This is a generic input, likely insufficient for most RE models.
                        formatted_input_for_model = f"{sent.text} Entity1: {subj_entity['text']}. Entity2: {obj_entity['text']}."
                        
                        try:
                            # --- Placeholder for model prediction and output parsing ---
                            # model_output = self.re_pipeline(formatted_input_for_model)
                            # Example: model_output might be [{'label': 'REGULATES', 'score': 0.9}]
                            # This parsing is highly model-dependent.
                            
                            # Simulate a model output for stub purposes
                            # This part needs to be replaced with actual model interaction and output parsing.
                            simulated_model_output = []
                            if "regulates" in sent.text.lower() and subj_entity['label'] == "GENE" and obj_entity['label'] == "GENE":
                                simulated_model_output = [{'label': 'REGULATES', 'score': 0.85}]
                            elif "associated" in sent.text.lower():
                                 simulated_model_output = [{'label': 'ASSOCIATED_WITH', 'score': 0.75}]


                            for pred in simulated_model_output: # Assuming model might return multiple relations or one
                                relation_type = pred['label']
                                confidence = pred['score']
                                
                                # Filter by confidence and target relation types
                                if confidence > 0.7 and (not self.target_relation_types or relation_type in self.target_relation_types):
                                    relations.append((
                                        subj_entity['text'],
                                        relation_type,
                                        obj_entity['text'],
                                        sent.text,
                                        chunk_id,
                                        source_doc,
                                        page_num
                                    ))
                        except Exception as e:
                            logger.error(f"Error during transformer RE prediction for chunk {chunk_id}: {e}")
        return relations


if __name__ == '__main__':
    import sys
    # Example usage of RelationExtractor
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    project_root_example = Path(__file__).resolve().parent.parent.parent
    models_cfg_path = project_root_example / "configs" / "models_config.yaml"

    # Ensure a models_config.yaml exists for the example
    if not models_cfg_path.exists():
        models_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_re_config = {
            "entity_recognition": { # RelationExtractor might use spacy model from NER config
                "spacy_model_name": "en_core_web_sm", 
            },
            "relation_extraction": {
                "hf_model_name": None, # Default to rule-based
                "target_relation_types": ["CO_OCCURS_WITH", "REGULATES_candidate", "ASSOCIATED_WITH"] 
            }
        }
        with open(models_cfg_path, 'w') as f:
            yaml.dump(dummy_re_config, f)
        logger.info(f"Created dummy models_config.yaml for RE example at {models_cfg_path}")

    # Sample data (mimicking output from EntityRecognizer)
    sample_entities_by_chunk = {
        "chunk1_doc1": [
            ("Dr. Prashant Kaushik", "PERSON", 0, 20, "chunk1_doc1", "doc1.pdf", 1),
            ("Solanum melongena", "CROP_SPECIES", 38, 55, "chunk1_doc1", "doc1.pdf", 1),
            ("eggplant", "CROP_SPECIES", 58, 66, "chunk1_doc1", "doc1.pdf", 1),
            ("abiotic stress", "STRESS_FACTOR", 95, 110, "chunk1_doc1", "doc1.pdf", 1),
            ("SmMYB1", "GENE", 150, 156, "chunk1_doc1", "doc1.pdf", 1),
            ("SmNAC1", "GENE", 161, 167, "chunk1_doc1", "doc1.pdf", 1),
            ("ICAR", "ORG", 195, 199, "chunk1_doc1", "doc1.pdf", 1),
        ],
        "chunk2_doc2": [
            ("Oryza sativa", "CROP_SPECIES", 19, 31, "chunk2_doc2", "doc2.pdf", 5),
            ("rice", "CROP_SPECIES", 34, 38, "chunk2_doc2", "doc2.pdf", 5),
            ("RNA sequencing", "METHODOLOGY", 75, 89, "chunk2_doc2", "doc2.pdf", 5),
            ("New Delhi", "GPE", 115, 124, "chunk2_doc2", "doc2.pdf", 5),
        ]
    }
    sample_chunks_with_metadata = [
        {
            "text": "Dr. Prashant Kaushik published a study on Solanum melongena (eggplant) in 2023. The study focused on abiotic stress response, particularly drought tolerance. Key genes like SmMYB1 and SmNAC1 were identified. The research was conducted at ICAR.",
            "chunk_id": "chunk1_doc1", "source_document": "doc1.pdf", "page_number": 1
        },
        {
            "text": "Further analysis of Oryza sativa (rice) showed similar stress response mechanisms. The experiments involved RNA sequencing and were performed in New Delhi.",
            "chunk_id": "chunk2_doc2", "source_document": "doc2.pdf", "page_number": 5
        }
    ]
    
    # Initialize RelationExtractor
    # It will try to load 'en_core_web_sm' by default if models_config.yaml doesn't specify or is missing.
    # Ensure 'en_core_web_sm' is downloaded: python -m spacy download en_core_web_sm
    try:
        relation_extractor = RelationExtractor(models_config_path=str(models_cfg_path))
    except Exception as e:
        logger.error(f"Failed to initialize RelationExtractor: {e}")
        logger.error("Make sure you have a spaCy model downloaded, e.g., 'python -m spacy download en_core_web_sm'")
        sys.exit(1)

    if relation_extractor.nlp_spacy is None: # Check if spaCy model loaded for rule-based
        logger.error("SpaCy model for rule-based RE could not be loaded. Exiting example.")
        sys.exit(1)

    logger.info("\nExtracting relations using rule-based approach...")
    extracted_relations = relation_extractor.extract_relations(sample_chunks_with_metadata, sample_entities_by_chunk)

    if extracted_relations:
        logger.info(f"Found {len(extracted_relations)} relations:")
        for rel in extracted_relations:
            logger.info(
                f"  Subject: '{rel[0]}', Relation: '{rel[1]}', Object: '{rel[2]}'\n"
                f"  Evidence: '{rel[3][:100]}...'\n" # Print first 100 chars of evidence
                f"  Chunk: {rel[4]}, Doc: {rel[5]}, Page: {rel[6]}"
            )
    else:
        logger.warning("No relations extracted from the sample text using the rule-based method.")

    logger.info("RelationExtractor example usage finished.")
