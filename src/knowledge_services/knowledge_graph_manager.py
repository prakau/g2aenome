import networkx as nx
from typing import List, Dict, Tuple, Any, Optional
from loguru import logger
import yaml
from pathlib import Path
import pickle # For saving/loading networkx graphs

# from ..common.logging_config import setup_logging
# setup_logging()

class KnowledgeGraphManager:
    """
    Manages the creation, modification, querying, and persistence of a knowledge graph
    using NetworkX.
    """

    def __init__(self, main_config_path: str = None):
        """
        Initializes the KnowledgeGraphManager.
        Args:
            main_config_path (str, optional): Path to the main configuration YAML file.
        """
        self.graph = nx.DiGraph() # Directed graph
        self.config = {}
        self._load_config(main_config_path)
        
        self.output_dir = Path(self.config.get("output_base_dir", "outputs")) / \
                          Path(self.config.get("knowledge_graph_dir", "knowledge_graph"))
        self.graph_file_name = self.config.get("knowledge_graph", {}).get("graph_file_name", "g2aenome_kg.graphml")
        self.graph_file_path = self.output_dir / self.graph_file_name
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_node_type = self.config.get("knowledge_graph", {}).get("default_node_type", "Concept")
        self.default_edge_type = self.config.get("knowledge_graph", {}).get("default_edge_type", "RELATED_TO")

        logger.info(f"KnowledgeGraphManager initialized. Graph will be saved to/loaded from: {self.graph_file_path}")


    def _load_config(self, config_path: str = None):
        """Loads KG related configurations from the main_config.yaml."""
        if config_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            config_path = project_root / "configs" / "main_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("KnowledgeGraphManager configuration loaded.")
        except FileNotFoundError:
            logger.error(f"Main configuration file not found at {config_path} for KGManager. Using defaults.")
        except Exception as e:
            logger.error(f"Error loading main configuration from {config_path} for KGManager: {e}. Using defaults.")


    def add_node_if_not_exists(self, node_id: str, node_type: str = None, **attributes):
        """Adds a node to the graph if it doesn't already exist. Updates attributes if it does."""
        if not node_id:
            logger.warning("Attempted to add a node with empty ID. Skipping.")
            return

        node_type = node_type or self.default_node_type
        
        # Attributes to be updated or set
        current_attributes = dict(attributes) # Make a mutable copy
        new_sources_data = current_attributes.pop('sources', None) # Extract 'sources' attribute if provided

        if not self.graph.has_node(node_id):
            # Node doesn't exist, add it
            self.graph.add_node(node_id, type=node_type, **current_attributes)
            self.graph.nodes[node_id]['sources'] = set() # Initialize sources as an empty set
            # logger.debug(f"Added node: {node_id} (Type: {node_type})")
        else:
            # Node exists, update other attributes (overwrite)
            for key, value in current_attributes.items():
                self.graph.nodes[node_id][key] = value
            # Ensure type is set/updated
            if self.graph.nodes[node_id].get('type') != node_type:
                 self.graph.nodes[node_id]['type'] = node_type
            # logger.debug(f"Node {node_id} already exists. Updated attributes.")

        # Ensure 'sources' attribute exists as a set and update it
        if 'sources' not in self.graph.nodes[node_id] or not isinstance(self.graph.nodes[node_id]['sources'], set):
            self.graph.nodes[node_id]['sources'] = set() # Ensure it's a set

        if new_sources_data:
            if isinstance(new_sources_data, set):
                self.graph.nodes[node_id]['sources'].update(new_sources_data)
            elif isinstance(new_sources_data, list): # Handles a list of source tuples
                for item in new_sources_data:
                    if isinstance(item, tuple):
                        self.graph.nodes[node_id]['sources'].add(item)
                    else:
                        logger.warning(f"Skipping non-tuple item in sources list for node {node_id}: {item}")
            elif isinstance(new_sources_data, tuple): # Handles a single source tuple
                 # Basic check to ensure it's likely a source tuple (e.g., 3 elements)
                if len(new_sources_data) == 3: # Assuming (doc, page, chunk_id)
                    self.graph.nodes[node_id]['sources'].add(new_sources_data)
                else:
                    logger.warning(f"Source data for node {node_id} is a tuple but not of expected structure: {new_sources_data}")
            else:
                logger.warning(f"Unsupported type for new_sources_data for node {node_id}: {type(new_sources_data)}. Expected set, list, or tuple.")
        
        # logger.debug(f"Node {node_id} final sources: {self.graph.nodes[node_id]['sources']}")

    def build_graph_from_entities_relations(self, 
                                            entities: List[Tuple[str, str, int, int, str, str, Optional[int]]], 
                                            relations: List[Tuple[str, str, str, str, str, str, Optional[int]]]):
        """
        Populates the knowledge graph from lists of extracted entities and relations.
        Args:
            entities (List[Tuple]): List of entity tuples from EntityRecognizer.
                Format: (entity_text, entity_label, start, end, chunk_id, source_doc, page_num (Optional))
            relations (List[Tuple]): List of relation tuples from RelationExtractor.
                Format: (subj_text, rel_type, obj_text, evidence, chunk_id, source_doc, page_num (Optional))
        """
        logger.info(f"Building graph from {len(entities)} entities and {len(relations)} relations.")
        
        # Add entities as nodes
        for ent_text, ent_label, _, _, chunk_id, source_doc, page_num in entities:
            # Normalize entity text for node ID (e.g., lowercasing, although case might be important)
            node_id = ent_text # Using raw text as ID for now, consider normalization
            self.add_node_if_not_exists(node_id, 
                                        node_type=ent_label, 
                                        label=ent_text, # Keep original text as a label attribute
                                        sources=set([(source_doc, page_num, chunk_id)])) # Store provenance

        # Add relations as edges
        for subj_text, rel_type, obj_text, evidence, chunk_id, source_doc, page_num in relations:
            subj_id = subj_text
            obj_id = obj_text

            # Ensure nodes exist (they should if entities list is comprehensive)
            self.add_node_if_not_exists(subj_id) # Add with default type if not seen as entity
            self.add_node_if_not_exists(obj_id)

            # Add edge
            if not self.graph.has_edge(subj_id, obj_id, key=rel_type): # Using rel_type as key for multigraph behavior
                self.graph.add_edge(subj_id, obj_id, key=rel_type, 
                                    relation_type=rel_type, 
                                    evidence=evidence,
                                    source_document=source_doc,
                                    page_number=page_num,
                                    chunk_id=chunk_id)
                # logger.debug(f"Added edge: {subj_id} -[{rel_type}]-> {obj_id}")
            else:
                # Handle existing edge, e.g., add more evidence or update properties
                # For simplicity, we're not updating if an edge with the same key exists.
                # logger.debug(f"Edge {subj_id} -[{rel_type}]-> {obj_id} already exists.")
                pass
        
        logger.info(f"Graph built. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")

    def add_genomic_data_to_graph(self, parsed_genomic_data: List[Dict[str, Any]], data_type: str = "fasta_record"):
        """
        Integrates structured genomic data into the knowledge graph.
        Args:
            parsed_genomic_data (List[Dict[str, Any]]): List of dicts, from GenomicProcessor.
            data_type (str): "fasta_record" or "gff_feature".
        """
        logger.info(f"Adding {len(parsed_genomic_data)} {data_type} items to the graph.")
        
        if data_type == "fasta_record":
            for record in parsed_genomic_data:
                node_id = record.get("id", record.get("name"))
                if not node_id: continue

                attributes = {
                    "label": record.get("name", node_id),
                    "description": record.get("description"),
                    "sequence_length": record.get("length"),
                    "organism": record.get("organism", self.default_node_type), # Or a specific "GenomicSequence" type
                    "source_file": record.get("source_file")
                }
                self.add_node_if_not_exists(node_id, node_type="GenomicSequence", **attributes)

        elif data_type == "gff_feature":
            for feature in parsed_genomic_data:
                feature_id = feature.get("id", feature.get("name"))
                if not feature_id: continue

                attributes = {k: v for k, v in feature.items() if k not in ["id", "type", "parent_ids"]}
                attributes["label"] = feature.get("name", feature_id)
                
                self.add_node_if_not_exists(feature_id, node_type=feature.get("type", "GenomicFeature"), **attributes)

                # Add relationships based on Parent IDs
                parent_ids = feature.get("parent_ids", [])
                for parent_id_attr in parent_ids:
                    # GFF Parent attribute can be a list of IDs, comma-separated string, etc.
                    # Assuming it's already processed into a list of strings by GenomicProcessor
                    if isinstance(parent_id_attr, str):
                         actual_parent_ids = parent_id_attr.split(',') # Handle comma-separated parents
                         for p_id in actual_parent_ids:
                            p_id = p_id.strip()
                            if p_id:
                                self.add_node_if_not_exists(p_id, node_type="GenomicFeature") # Ensure parent node exists
                                if not self.graph.has_edge(feature_id, p_id, key="IS_PART_OF"): # Child IS_PART_OF Parent
                                    self.graph.add_edge(feature_id, p_id, key="IS_PART_OF", relation_type="IS_PART_OF",
                                                        source_document=feature.get("source_file"))
                                # Or Parent HAS_CHILD Feature
                                # if not self.graph.has_edge(p_id, feature_id, key="HAS_CHILD"):
                                #    self.graph.add_edge(p_id, feature_id, key="HAS_CHILD", relation_type="HAS_CHILD")
                    elif isinstance(parent_id_attr, list): # If it's already a list
                         for p_id in parent_id_attr:
                            p_id = p_id.strip()
                            if p_id:
                                self.add_node_if_not_exists(p_id, node_type="GenomicFeature")
                                if not self.graph.has_edge(feature_id, p_id, key="IS_PART_OF"):
                                    self.graph.add_edge(feature_id, p_id, key="IS_PART_OF", relation_type="IS_PART_OF",
                                                        source_document=feature.get("source_file"))


        else:
            logger.warning(f"Unsupported genomic data_type for KG integration: {data_type}")

        logger.info(f"Graph after adding genomic data. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")


    def query_graph(self, entity_name: str, relation_type: Optional[str] = None, depth: int = 1) -> List[Dict[str, Any]]:
        """
        Performs basic graph traversal queries.
        Args:
            entity_name (str): The name/ID of the entity to start querying from.
            relation_type (Optional[str]): If specified, filter outgoing relationships by this type.
            depth (int): How many hops to traverse (currently supports 1-hop neighbors).
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a related entity
                                  and the relationship connecting them.
        """
        results = []
        if not self.graph.has_node(entity_name):
            logger.warning(f"Entity '{entity_name}' not found in the knowledge graph.")
            return results

        # Get 1-hop neighbors (successors and predecessors)
        # Successors (outgoing edges)
        for successor in self.graph.successors(entity_name):
            edges_data = self.graph.get_edge_data(entity_name, successor)
            for edge_key, edge_attrs in edges_data.items():
                if relation_type is None or edge_attrs.get("relation_type") == relation_type or edge_key == relation_type:
                    results.append({
                        "source_entity": entity_name,
                        "relation": edge_attrs.get("relation_type", edge_key),
                        "target_entity": successor,
                        "target_entity_type": self.graph.nodes[successor].get("type"),
                        "target_entity_label": self.graph.nodes[successor].get("label", successor),
                        "evidence": edge_attrs.get("evidence"),
                        "direction": "outgoing"
                    })
        
        # Predecessors (incoming edges)
        for predecessor in self.graph.predecessors(entity_name):
            edges_data = self.graph.get_edge_data(predecessor, entity_name)
            for edge_key, edge_attrs in edges_data.items():
                 if relation_type is None or edge_attrs.get("relation_type") == relation_type or edge_key == relation_type:
                    results.append({
                        "source_entity": predecessor,
                        "relation": edge_attrs.get("relation_type", edge_key),
                        "target_entity": entity_name,
                        "source_entity_type": self.graph.nodes[predecessor].get("type"),
                        "source_entity_label": self.graph.nodes[predecessor].get("label", predecessor),
                        "evidence": edge_attrs.get("evidence"),
                        "direction": "incoming"
                    })
        
        # TODO: Implement multi-hop traversal if depth > 1 using BFS or DFS.
        # For now, depth parameter is noted but only 1-hop is implemented.
        if depth > 1:
            logger.warning(f"Query depth {depth} requested, but only 1-hop is currently implemented.")

        logger.info(f"Query for '{entity_name}' (relation: {relation_type or 'ANY'}) returned {len(results)} results.")
        return results

    def save_graph(self, path: Optional[str] = None):
        """
        Saves the graph to a file. Supports GraphML or Pickle.
        Args:
            path (Optional[str]): Path to save the graph. If None, uses configured path.
        """
        save_path = Path(path) if path else self.graph_file_path
        save_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        try:
            if save_path.suffix == ".graphml":
                nx.write_graphml(self.graph, save_path)
            elif save_path.suffix == ".pkl":
                with open(save_path, "wb") as f:
                    pickle.dump(self.graph, f)
            else: # Default to GraphML if extension is unknown or missing
                logger.warning(f"Unknown graph file extension '{save_path.suffix}'. Defaulting to GraphML. Path: {save_path.with_suffix('.graphml')}")
                if save_path.suffix == "": save_path = save_path.with_suffix(".graphml") # Add if no suffix
                nx.write_graphml(self.graph, save_path)

            logger.info(f"Knowledge graph saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save knowledge graph to {save_path}: {e}")

    def load_graph(self, path: Optional[str] = None) -> bool:
        """
        Loads the graph from a file. Supports GraphML or Pickle.
        Args:
            path (Optional[str]): Path to load the graph from. If None, uses configured path.
        Returns:
            bool: True if graph loaded successfully, False otherwise.
        """
        load_path = Path(path) if path else self.graph_file_path

        if not load_path.exists():
            logger.warning(f"Knowledge graph file not found at {load_path}. Starting with an empty graph.")
            self.graph = nx.DiGraph()
            return False
        
        try:
            if load_path.suffix == ".graphml":
                self.graph = nx.read_graphml(load_path)
            elif load_path.suffix == ".pkl":
                with open(load_path, "rb") as f:
                    self.graph = pickle.load(f)
            else:
                logger.error(f"Unknown graph file extension '{load_path.suffix}'. Cannot load graph from {load_path}.")
                return False
            
            logger.info(f"Knowledge graph loaded from {load_path}. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
            return True
        except Exception as e:
            logger.error(f"Failed to load knowledge graph from {load_path}: {e}. Starting with an empty graph.")
            self.graph = nx.DiGraph()
            return False

    def get_node_details(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Returns all attributes of a specific node."""
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]
        logger.warning(f"Node '{node_id}' not found in graph.")
        return None

    # Comment on persistent graph DBs:
    # This NetworkX implementation is for in-memory graph operations, suitable for PoCs.
    # For a production system, a persistent graph database like Neo4j, Amazon Neptune,
    # or ArangoDB would be more appropriate.
    # Mapping to such a DB would involve:
    # - Nodes: Creating nodes with labels (e.g., from entity_type) and properties (attributes).
    # - Edges: Creating relationships with types (e.g., from relation_type) and properties.
    # - Indexing: Creating indexes on node properties (e.g., entity_id, name) for faster lookups.
    # - Querying: Using the DB's query language (e.g., Cypher for Neo4j, Gremlin for Neptune)
    #   instead of NetworkX traversal methods.
    # - Schema: Defining a schema for nodes and relationships if the DB supports/requires it.

if __name__ == '__main__':
    import sys
    # Example usage of KnowledgeGraphManager
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    project_root_example = Path(__file__).resolve().parent.parent.parent
    config_path_example = project_root_example / "configs" / "main_config.yaml"

    # Ensure a main_config.yaml exists for the example
    if not config_path_example.exists():
        config_path_example.parent.mkdir(parents=True, exist_ok=True)
        dummy_main_config = {
            "output_base_dir": "outputs_kg_example",
            "knowledge_graph_dir": "kg_store",
            "knowledge_graph": {
                "graph_file_name": "example_kg.graphml",
                "default_node_type": "TestConcept",
                "default_edge_type": "TEST_RELATES_TO"
            }
        }
        with open(config_path_example, 'w') as f:
            yaml.dump(dummy_main_config, f)
        logger.info(f"Created dummy main_config.yaml for KG example at {config_path_example}")


    kg_manager = KnowledgeGraphManager(main_config_path=str(config_path_example))

    # Sample data
    sample_entities = [
        ("SmMYB1", "GENE", 0, 0, "c1", "doc1", 1),
        ("drought tolerance", "TRAIT", 0, 0, "c1", "doc1", 1),
        ("Solanum melongena", "CROP_SPECIES", 0, 0, "c1", "doc1", 1)
    ]
    sample_relations = [
        ("SmMYB1", "CONFERS", "drought tolerance", "Evidence sentence 1", "c1", "doc1", 1),
        ("SmMYB1", "FOUND_IN", "Solanum melongena", "Evidence sentence 2", "c1", "doc1", 1)
    ]
    sample_fasta = [{"id": "fasta_seq1", "name": "SeqA", "description": "A test sequence", "length": 100, "organism": "TestOrganism", "source_file": "test.fasta"}]
    sample_gff = [{"id": "geneX", "type": "gene", "name": "GeneX", "parent_ids": [], "source_file": "test.gff"}]


    logger.info("\n--- Building graph from entities and relations ---")
    kg_manager.build_graph_from_entities_relations(sample_entities, sample_relations)
    
    logger.info("\n--- Adding genomic data to graph ---")
    kg_manager.add_genomic_data_to_graph(sample_fasta, data_type="fasta_record")
    kg_manager.add_genomic_data_to_graph(sample_gff, data_type="gff_feature")

    logger.info(f"\nGraph stats: Nodes={kg_manager.graph.number_of_nodes()}, Edges={kg_manager.graph.number_of_edges()}")

    logger.info("\n--- Querying graph for 'SmMYB1' ---")
    query_results = kg_manager.query_graph("SmMYB1")
    for res in query_results:
        logger.info(f"  {res}")

    logger.info("\n--- Getting node details for 'drought tolerance' ---")
    node_details = kg_manager.get_node_details("drought tolerance")
    logger.info(f"  Details: {node_details}")

    logger.info("\n--- Saving graph ---")
    kg_manager.save_graph() # Saves to configured path

    logger.info("\n--- Loading graph (into a new manager instance for test) ---")
    new_kg_manager = KnowledgeGraphManager(main_config_path=str(config_path_example))
    loaded_successfully = new_kg_manager.load_graph()
    if loaded_successfully:
        logger.info(f"Loaded graph stats: Nodes={new_kg_manager.graph.number_of_nodes()}, Edges={new_kg_manager.graph.number_of_edges()}")
        query_results_loaded = new_kg_manager.query_graph("SmMYB1")
        logger.info(f"Query on loaded graph for 'SmMYB1' found {len(query_results_loaded)} results.")
    else:
        logger.error("Failed to load the saved graph.")

    # Clean up dummy config if it was created by this example script
    if "outputs_kg_example" in str(config_path_example): # A bit of a hack to check if it's the dummy
        import shutil
        if (project_root_example / "outputs_kg_example").exists():
            shutil.rmtree(project_root_example / "outputs_kg_example")
            logger.info("Cleaned up dummy output directory.")
        # config_path_example.unlink(missing_ok=True) # Don't delete main_config if it was pre-existing
        # logger.info("Cleaned up dummy main_config.yaml if it was created by this script.")


    logger.info("KnowledgeGraphManager example usage finished.")
