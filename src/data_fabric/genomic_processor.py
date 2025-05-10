import os
from pathlib import Path
from typing import List, Dict, Any, Iterator
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature
from BCBio import GFF # BCBio.GFF handles GFF3 parsing robustly
from loguru import logger
import re # Moved import re to the top

# Ensure logger is configured
# from ..common.logging_config import setup_logging
# setup_logging()

class GenomicProcessor:
    """
    Handles parsing of genomic data files like FASTA and GFF.
    """

    def __init__(self, config: Dict = None):
        """
        Initializes the GenomicProcessor.
        Args:
            config (Dict, optional): Configuration dictionary, typically from main_config.yaml.
                                     Expected keys: genomic_processing.default_organism, etc.
        """
        self.config = config or {}
        self.default_organism = self.config.get("genomic_processing", {}).get("default_organism", "Unknown")
        logger.info(f"GenomicProcessor initialized. Default organism: {self.default_organism}")

    def parse_fasta_files(self, fasta_directory_path: str) -> List[Dict[str, Any]]:
        """
        Parses all FASTA files in a given directory.
        Args:
            fasta_directory_path (str): Path to the directory containing FASTA files.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  represents a sequence record with id, description, sequence,
                                  and source filename.
        """
        directory = Path(fasta_directory_path)
        if not directory.is_dir():
            logger.error(f"FASTA directory not found or is not a directory: {fasta_directory_path}")
            return []

        fasta_files = list(directory.glob("*.fasta")) + list(directory.glob("*.fa")) + \
                      list(directory.glob("*.fna")) + list(directory.glob("*.ffn")) + \
                      list(directory.glob("*.faa")) # Common FASTA extensions
        
        all_sequences = []
        logger.info(f"Found {len(fasta_files)} FASTA files in {fasta_directory_path}.")

        for fasta_file in fasta_files:
            try:
                logger.debug(f"Parsing FASTA file: {fasta_file.name}")
                count = 0
                for record in SeqIO.parse(fasta_file, "fasta"):
                    all_sequences.append({
                        "id": record.id,
                        "name": record.name,
                        "description": record.description,
                        "sequence": str(record.seq),
                        "length": len(record.seq),
                        "source_file": fasta_file.name,
                        "organism": self._extract_organism_from_description(record.description) # Basic heuristic
                    })
                    count += 1
                logger.info(f"Parsed {count} sequences from {fasta_file.name}.")
            except FileNotFoundError:
                logger.error(f"FASTA file not found during parsing: {fasta_file}")
            except Exception as e:
                logger.error(f"Error parsing FASTA file {fasta_file.name}: {e}")
        
        logger.info(f"Total sequences parsed from all FASTA files: {len(all_sequences)}")
        return all_sequences

    def _extract_organism_from_description(self, description: str) -> str:
        """Helper to attempt to extract organism name from FASTA description."""
        # This is a very basic heuristic, might need significant improvement
        # e.g., looking for [Organism Name] or specific keywords.
        # For now, it's a placeholder.
        match = re.search(r"\[organism=([^\]]+)\]", description, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Could add more patterns here for common model organisms
        # e.g. Solanum melongena, Oryza sativa, Arabidopsis thaliana
        
        return self.default_organism


    def parse_gff_files(self, gff_directory_path: str) -> List[Dict[str, Any]]:
        """
        Parses all GFF (version 3) files in a given directory.
        Args:
            gff_directory_path (str): Path to the directory containing GFF files.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  represents a genomic feature with its attributes.
        """
        directory = Path(gff_directory_path)
        if not directory.is_dir():
            logger.error(f"GFF directory not found or is not a directory: {gff_directory_path}")
            return []

        gff_files = list(directory.glob("*.gff")) + list(directory.glob("*.gff3"))
        
        all_features_data = []
        logger.info(f"Found {len(gff_files)} GFF files in {gff_directory_path}.")

        for gff_file in gff_files:
            try:
                logger.debug(f"Parsing GFF file: {gff_file.name}")
                in_handle = open(gff_file)
                limit_info = dict(gff_type = self.config.get("genomic_processing", {}).get("feature_types_to_extract", [])) # Example: ["gene", "mRNA", "CDS"]
                
                file_feature_count = 0
                for rec in GFF.parse(in_handle, limit_info=limit_info if limit_info["gff_type"] else None):
                    # rec is a Bio.SeqRecord.SeqRecord object
                    for feature in rec.features:
                        # Process feature and its sub_features recursively
                        all_features_data.extend(self._process_gff_feature(feature, rec.id, gff_file.name))
                        file_feature_count +=1 # Counting top-level features
                
                logger.info(f"Parsed {file_feature_count} top-level features from GFF file: {gff_file.name}")
                in_handle.close()
            except FileNotFoundError:
                logger.error(f"GFF file not found during parsing: {gff_file}")
            except Exception as e:
                logger.error(f"Error parsing GFF file {gff_file.name}: {e}")
        
        logger.info(f"Total features (including sub-features) processed from all GFF files: {len(all_features_data)}")
        return all_features_data

    def _process_gff_feature(self, feature: SeqFeature, chromosome_id: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Recursively processes a GFF feature and its sub-features.
        Args:
            feature (SeqFeature): The BioPython SeqFeature object.
            chromosome_id (str): The ID of the chromosome/contig the feature belongs to.
            source_file (str): The name of the GFF file.
        Returns:
            List[Dict[str, Any]]: A list containing the processed feature and its sub-features.
        """
        feature_data_list = []
        
        # Basic feature information
        feature_info = {
            "id": feature.id if feature.id and feature.id != "<unknown id>" else feature.qualifiers.get("ID", [None])[0],
            "type": feature.type,
            "location_start": int(feature.location.start) + 1, # GFF is 1-based
            "location_end": int(feature.location.end),
            "strand": int(feature.location.strand) if feature.location else None,
            "score": feature.qualifiers.get("score", [None])[0],
            "phase": feature.qualifiers.get("phase", [None])[0], # For CDS features
            "source": feature.qualifiers.get("source", [None])[0],
            "chromosome": chromosome_id,
            "source_file": source_file,
            "attributes": {k: v[0] if len(v) == 1 else v for k, v in feature.qualifiers.items()} # Flatten single-item lists
        }
        
        # Extract common attributes like Name, Parent, Alias, Note
        feature_info["name"] = feature.qualifiers.get("Name", [feature_info["id"]])[0] # Default to ID if Name not present
        feature_info["parent_ids"] = feature.qualifiers.get("Parent", [])
        
        feature_data_list.append(feature_info)

        # Recursively process sub-features
        if feature.sub_features:
            for sub_feature in feature.sub_features:
                feature_data_list.extend(self._process_gff_feature(sub_feature, chromosome_id, source_file))
        
        return feature_data_list


if __name__ == '__main__':
    # Example usage of GenomicProcessor
    import sys
    import yaml
    import re # Added for _extract_organism_from_description

    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    project_root = Path(__file__).resolve().parent.parent.parent
    main_config_path = project_root / "configs" / "main_config.yaml"

    if not main_config_path.exists():
        logger.error(f"Main config not found at {main_config_path} for example usage.")
        sys.exit(1)
        
    with open(main_config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    genomic_proc_config = config_data.get("genomic_processing", {})
    data_dir = project_root / config_data.get("data_base_dir", "data")
    genomic_data_dir_name = config_data.get("genomic_data_dir", "genomic_data_samples")
    actual_genomic_data_dir = data_dir / genomic_data_dir_name

    if not actual_genomic_data_dir.exists():
        logger.warning(f"Sample genomic data directory not found: {actual_genomic_data_dir}. Creating it.")
        actual_genomic_data_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy FASTA and GFF files for testing if they don't exist
    sample_fasta_path = actual_genomic_data_dir / "sample.fasta"
    if not sample_fasta_path.exists():
        with open(sample_fasta_path, "w") as f:
            f.write(">seq1 [organism=Solanum melongena] Sample sequence 1\n")
            f.write("ATGCGTAGCATCGATCGATCG\n")
            f.write(">seq2 [organism=Oryza sativa] Sample sequence 2\n")
            f.write("CGATCGATCGATGCGCGATCG\n")
        logger.info(f"Created dummy FASTA file: {sample_fasta_path}")

    sample_gff_path = actual_genomic_data_dir / "sample.gff3"
    if not sample_gff_path.exists():
        with open(sample_gff_path, "w") as f:
            f.write("##gff-version 3\n")
            f.write("chr1\tSource\tgene\t1000\t2000\t.\t+\t.\tID=gene001;Name=GeneA\n")
            f.write("chr1\tSource\tmRNA\t1000\t2000\t.\t+\t.\tID=mRNA001;Parent=gene001;Name=mRNA_A\n")
            f.write("chr1\tSource\tCDS\t1050\t1500\t.\t+\t0\tID=cds001;Parent=mRNA001;Name=CDS_A\n")
        logger.info(f"Created dummy GFF3 file: {sample_gff_path}")


    processor = GenomicProcessor(config={"genomic_processing": genomic_proc_config})

    logger.info(f"\n--- Parsing FASTA files from: {actual_genomic_data_dir} ---")
    fasta_data = processor.parse_fasta_files(str(actual_genomic_data_dir))
    if fasta_data:
        logger.info(f"Total FASTA records parsed: {len(fasta_data)}")
        logger.info("Sample FASTA record:")
        logger.info(fasta_data[0])
    else:
        logger.warning("No FASTA data parsed.")

    logger.info(f"\n--- Parsing GFF files from: {actual_genomic_data_dir} ---")
    gff_data = processor.parse_gff_files(str(actual_genomic_data_dir))
    if gff_data:
        logger.info(f"Total GFF features processed: {len(gff_data)}")
        logger.info("Sample GFF feature:")
        logger.info(gff_data[0])
    else:
        logger.warning("No GFF data parsed.")
    
    logger.info("GenomicProcessor example usage finished.")
