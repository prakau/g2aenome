import pytest
from pathlib import Path
import os
import sys
import yaml
import networkx as nx

# Adjust Python path
PROJECT_ROOT_TESTS = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT_TESTS / "src"))

from knowledge_services.entity_recognition import EntityRecognizer
from knowledge_services.relation_extraction import RelationExtractor
from knowledge_services.knowledge_graph_manager import KnowledgeGraphManager
from knowledge_services.vector_store_manager import VectorStoreManager

# --- Fixtures ---

@pytest.fixture(scope="module")
def temp_output_dir(tmp_path_factory):
    """Creates a temporary output directory for KG and VS artifacts."""
    return tmp_path_factory.mktemp("test_outputs")

@pytest.fixture(scope="module")
def dummy_models_config_path(tmp_path_factory):
    """Creates a dummy models_config.yaml for testing."""
    config_dir = tmp_path_factory.mktemp("test_configs_ks")
    config_path = config_dir / "models_config.yaml"
    dummy_config = {
        "entity_recognition": {
            "spacy_model_name": "en_core_web_sm", # Use small model for tests
            "target_entity_labels": ["PERSON", "ORG", "LOC", "GPE", "PRODUCT"] 
        },
        "relation_extraction": {
            "hf_model_name": None, # Rule-based for tests
            "target_relation_types": ["CO_OCCURS_WITH"]
        },
        "embedding_model": {
            "hf_model_name": "sentence-transformers/all-MiniLM-L6-v2", # Small ST model
            "embedding_dim": 384
        }
    }
    with open(config_path, 'w') as f:
        yaml.dump(dummy_config, f)
    return config_path

@pytest.fixture(scope="module")
def dummy_main_config_path(temp_output_dir, tmp_path_factory):
    """Creates a dummy main_config.yaml pointing to temp_output_dir."""
    config_dir = tmp_path_factory.mktemp("test_configs_main_ks") # Ensure different from models_config dir
    config_path = config_dir / "main_config.yaml"
    
    # Ensure subdirectories for KG and VS exist within temp_output_dir
    (temp_output_dir / "kg_test_store").mkdir(exist_ok=True)
    (temp_output_dir / "vs_test_store").mkdir(exist_ok=True)

    dummy_config = {
        "output_base_dir": str(temp_output_dir), # Use the temp_output_dir from fixture
        "knowledge_graph_dir": "kg_test_store",
        "vector_store_dir": "vs_test_store",
        "knowledge_graph": {
            "graph_file_name": "test_kg.graphml",
        },
        "vector_store": {
            "index_file_name": "test_vs.idx",
            "default_top_k": 3
        },
        "logging": {"level": "DEBUG"} # For test verbosity if needed
    }
    with open(config_path, 'w') as f:
        yaml.dump(dummy_config, f)
    return config_path


@pytest.fixture(scope="module")
def entity_recognizer_instance(dummy_models_config_path):
    # This will attempt to load en_core_web_sm. Ensure it's available.
    # `python -m spacy download en_core_web_sm`
    try:
        return EntityRecognizer(models_config_path=str(dummy_models_config_path))
    except Exception as e:
        pytest.skip(f"Skipping ER tests: Failed to init EntityRecognizer (spaCy model likely missing or error): {e}")


@pytest.fixture(scope="module")
def relation_extractor_instance(dummy_models_config_path):
    try:
        return RelationExtractor(models_config_path=str(dummy_models_config_path))
    except Exception as e:
        pytest.skip(f"Skipping RE tests: Failed to init RelationExtractor: {e}")

@pytest.fixture(scope="module")
def kg_manager_instance(dummy_main_config_path):
    return KnowledgeGraphManager(main_config_path=str(dummy_main_config_path))

@pytest.fixture(scope="module")
def vector_store_manager_instance(dummy_main_config_path, dummy_models_config_path):
    # This will attempt to load all-MiniLM-L6-v2.
    try:
        return VectorStoreManager(main_config_path=str(dummy_main_config_path), 
                                  models_config_path=str(dummy_models_config_path))
    except Exception as e:
        pytest.skip(f"Skipping VS tests: Failed to init VectorStoreManager (transformer model likely missing or error): {e}")


# Sample data for tests
sample_text_chunks_ks = [
    {
        "text": "Dr. Kaushik from ICAR studied eggplant in New Delhi. He found GeneA regulates TraitX.",
        "source_document": "paper1.pdf", "page_number": 1, "chunk_id": "paper1_p1_c1"
    },
    {
        "text": "Eggplant, also known as Solanum melongena, is a crop. TraitX is important.",
        "source_document": "paper1.pdf", "page_number": 2, "chunk_id": "paper1_p2_c1"
    }
]

# --- EntityRecognizer Tests ---
def test_er_load_model(entity_recognizer_instance):
    if entity_recognizer_instance: # Check if fixture skipped
        assert entity_recognizer_instance.nlp is not None
        assert entity_recognizer_instance.model_name == "en_core_web_sm" # From dummy_models_config

def test_er_extract_entities(entity_recognizer_instance):
    if not entity_recognizer_instance or not entity_recognizer_instance.nlp:
        pytest.skip("EntityRecognizer or its spaCy model not loaded.")
    
    entities = entity_recognizer_instance.extract_entities(sample_text_chunks_ks)
    assert isinstance(entities, list)
    
    found_persons = [e for e in entities if e[1] == "PERSON"]
    found_orgs = [e for e in entities if e[1] == "ORG"]
    found_gpe = [e for e in entities if e[1] == "GPE"] # Geo-Political Entity
    
    # Based on en_core_web_sm and sample_text_chunks_ks
    assert any(e[0] == "Kaushik" for e in found_persons)
    assert any(e[0] == "ICAR" for e in found_orgs)
    assert any(e[0] == "New Delhi" for e in found_gpe)
    # Note: "eggplant", "GeneA", "TraitX" might not be recognized by en_core_web_sm default entities.
    # This test primarily checks if the process runs and finds some standard entities.

# --- RelationExtractor Tests ---
def test_re_extract_relations_rule_based(relation_extractor_instance, entity_recognizer_instance):
    if not relation_extractor_instance or not relation_extractor_instance.nlp_spacy:
        pytest.skip("RelationExtractor or its spaCy model not loaded.")
    if not entity_recognizer_instance or not entity_recognizer_instance.nlp:
        pytest.skip("EntityRecognizer for pre-requisite entity extraction not loaded.")

    # First, get entities
    raw_entities = entity_recognizer_instance.extract_entities(sample_text_chunks_ks)
    entities_by_chunk = {}
    for ent_tuple in raw_entities:
        chunk_id = ent_tuple[4]
        if chunk_id not in entities_by_chunk: entities_by_chunk[chunk_id] = []
        entities_by_chunk[chunk_id].append(ent_tuple)

    relations = relation_extractor_instance.extract_relations(sample_text_chunks_ks, entities_by_chunk)
    assert isinstance(relations, list)
    # Rule-based is very basic, might find CO_OCCURS_WITH or simple verb candidates.
    # Example: ("Kaushik", "CO_OCCURS_WITH", "ICAR", ...)
    if relations:
        assert len(relations[0]) == 7 # (subj, rel, obj, evidence, chunk_id, source, page)
        # Check if any "CO_OCCURS_WITH" relations were found, as per dummy config
        assert any(r[1] == "CO_OCCURS_WITH" for r in relations)


# --- KnowledgeGraphManager Tests ---
def test_kgm_add_and_query_nodes(kg_manager_instance):
    kg_manager_instance.add_node_if_not_exists("NodeA", node_type="TestType", description="Test node A")
    kg_manager_instance.add_node_if_not_exists("NodeB", node_type="TestType", description="Test node B")
    kg_manager_instance.graph.add_edge("NodeA", "NodeB", key="TEST_REL", relation_type="TEST_REL", evidence="Test evidence")

    assert kg_manager_instance.graph.has_node("NodeA")
    assert kg_manager_instance.graph.nodes["NodeA"]["type"] == "TestType"
    
    query_res = kg_manager_instance.query_graph("NodeA", relation_type="TEST_REL")
    assert len(query_res) == 1
    assert query_res[0]["target_entity"] == "NodeB"

def test_kgm_save_load_graph(kg_manager_instance, temp_output_dir):
    kg_manager_instance.add_node_if_not_exists("PersistNode", node_type="Persistent")
    kg_manager_instance.save_graph() # Uses path from dummy_main_config_path

    # Create new instance to test loading
    new_kgm = KnowledgeGraphManager(main_config_path=kg_manager_instance.config_path_for_test_if_needed) # Need to pass config path
    
    # Hacky way to get the config path used by the fixture instance
    # Ideally, the fixture should return the config path too, or it should be fixed.
    # For now, let's assume the fixture setup the path correctly for new_kgm.
    # This part of the test might be flaky if config path isn't correctly inferred/passed.
    # Let's try to reconstruct the path based on the fixture's logic.
    # This is not ideal. A better fixture design would provide this.
    # For now, we assume kg_manager_instance.graph_file_path is correctly set by its init.
    
    loaded = new_kgm.load_graph(kg_manager_instance.graph_file_path) # Load from where the first instance saved
    assert loaded
    assert new_kgm.graph.has_node("PersistNode")


# --- VectorStoreManager Tests ---
def test_vsm_create_and_search(vector_store_manager_instance):
    if not vector_store_manager_instance or not vector_store_manager_instance.embedding_model:
        pytest.skip("VectorStoreManager or its embedding model not loaded.")

    vector_store_manager_instance.create_or_load_vector_store(sample_text_chunks_ks, force_recreate=True)
    assert vector_store_manager_instance.index is not None
    assert vector_store_manager_instance.index.ntotal == len(sample_text_chunks_ks)

    search_results = vector_store_manager_instance.search_vector_store("eggplant study", k=1)
    assert len(search_results) >= 1 # Should find at least one relevant chunk
    assert "text" in search_results[0]
    assert "eggplant" in search_results[0]["text"].lower() # Check if relevant text is returned

def test_vsm_save_load_store(vector_store_manager_instance, temp_output_dir):
    if not vector_store_manager_instance or not vector_store_manager_instance.embedding_model:
        pytest.skip("VectorStoreManager or its embedding model not loaded.")

    vector_store_manager_instance.create_or_load_vector_store(sample_text_chunks_ks, force_recreate=True)
    vector_store_manager_instance.save_vector_store() # Uses path from dummy_main_config

    new_vsm = VectorStoreManager(
        main_config_path=vector_store_manager_instance.main_config_path_for_test_if_needed, # Needs config path
        models_config_path=vector_store_manager_instance.models_config_path_for_test_if_needed
    )
    new_vsm.create_or_load_vector_store(force_recreate=False) # Should load from disk
    
    assert new_vsm.index is not None
    assert new_vsm.index.ntotal == len(sample_text_chunks_ks)

# Helper to pass config paths to new instances in save/load tests
@pytest.fixture(autouse=True)
def add_config_paths_to_managers(kg_manager_instance, vector_store_manager_instance, dummy_main_config_path, dummy_models_config_path):
    if kg_manager_instance:
        kg_manager_instance.config_path_for_test_if_needed = str(dummy_main_config_path)
    if vector_store_manager_instance:
        vector_store_manager_instance.main_config_path_for_test_if_needed = str(dummy_main_config_path)
        vector_store_manager_instance.models_config_path_for_test_if_needed = str(dummy_models_config_path)
