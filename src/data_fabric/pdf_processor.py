import os
from pathlib import Path
from typing import List, Dict, Tuple, Any
import PyPDF2
from bs4 import BeautifulSoup # For potential HTML-like content or advanced cleaning
from loguru import logger
import unicodedata
import re

# Ensure logger is configured, typically at application entry point
# from ..common.logging_config import setup_logging
# setup_logging() # Or get logger instance if already configured

class PDFProcessor:
    """
    Handles loading, extracting text from, cleaning, and chunking PDF documents.
    """

    def __init__(self, config: Dict = None):
        """
        Initializes the PDFProcessor.
        Args:
            config (Dict, optional): Configuration dictionary, typically from main_config.yaml.
                                     Expected keys: pdf_processing.chunk_size, pdf_processing.chunk_overlap.
        """
        self.config = config or {}
        try:
            self.chunk_size = int(self.config.get("pdf_processing", {}).get("chunk_size", 1000))
            self.chunk_overlap = int(self.config.get("pdf_processing", {}).get("chunk_overlap", 200))
        except ValueError:
            logger.warning("Chunk size or overlap in config is not an integer. Using defaults (1000, 200).")
            self.chunk_size = 1000
            self.chunk_overlap = 200

        if self.chunk_size <= self.chunk_overlap:
            logger.warning(f"Chunk overlap ({self.chunk_overlap}) is greater than or equal to chunk size ({self.chunk_size}). "
                           f"This may lead to inefficient chunking or errors. Adjusting overlap to be less than chunk size.")
            # Adjust overlap to be, for example, 20% of chunk_size, or 0 if chunk_size is too small
            self.chunk_overlap = min(self.chunk_overlap, self.chunk_size -1) if self.chunk_size > 0 else 0
            if self.chunk_overlap < 0: self.chunk_overlap = 0 # Ensure non-negative

        logger.info(f"PDFProcessor initialized. Chunk size: {self.chunk_size}, Chunk overlap: {self.chunk_overlap}")

    def load_pdfs(self, pdf_directory_path: str) -> List[Path]:
        """
        Iterates through PDF files in the specified directory.
        Args:
            pdf_directory_path (str): The path to the directory containing PDF files.
        Returns:
            List[Path]: A list of Path objects for each PDF file found.
        """
        directory = Path(pdf_directory_path)
        if not directory.is_dir():
            logger.error(f"PDF directory not found or is not a directory: {pdf_directory_path}")
            return []

        pdf_files = list(directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_directory_path}.")
        return pdf_files

    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extracts text from a single PDF file, attempting to preserve page information.
        Args:
            pdf_path (Path): The path to the PDF file.
        Returns:
            Tuple[str, List[Dict[str, Any]]]: A tuple containing:
                - The full extracted text from the PDF.
                - A list of dictionaries, where each dictionary represents a page
                  and contains 'page_number' and 'text'.
        """
        full_text = ""
        pages_content = []
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                logger.debug(f"Extracting text from {pdf_path.name} ({num_pages} pages).")
                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            cleaned_page_text = self.clean_text(page_text)
                            full_text += cleaned_page_text + "\n\n" # Add space between pages
                            pages_content.append({
                                "page_number": i + 1,
                                "text": cleaned_page_text,
                                "source_document": pdf_path.name
                            })
                        else:
                            logger.warning(f"Page {i+1} in {pdf_path.name} has no extractable text.")
                    except Exception as e:
                        logger.error(f"Error extracting text from page {i+1} of {pdf_path.name}: {e}")
                        pages_content.append({
                            "page_number": i + 1,
                            "text": "", # Empty text on error for this page
                            "source_document": pdf_path.name
                        })
            logger.info(f"Successfully extracted text from {pdf_path.name}.")
        except FileNotFoundError:
            logger.error(f"PDF file not found: {pdf_path}")
        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"Error reading PDF file {pdf_path.name} (possibly corrupted or password-protected): {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {pdf_path.name}: {e}")
        
        return full_text.strip(), pages_content

    def clean_text(self, raw_text: str) -> str:
        """
        Performs advanced cleaning of raw extracted text.
        - Unicode normalization
        - Hyphen de-duplication (attempt to rejoin words broken across lines)
        - Ligature replacement (e.g., ﬁ -> fi)
        - Whitespace normalization
        Args:
            raw_text (str): The raw text extracted from a PDF page.
        Returns:
            str: The cleaned text.
        """
        if not raw_text:
            return ""

        # Unicode normalization (NFKC is good for compatibility)
        text = unicodedata.normalize('NFKC', raw_text)

        # Common ligatures (add more if needed)
        ligatures = {
            "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
            "ﬅ": "ft", "ﬆ": "st",
        }
        for lig, repl in ligatures.items():
            text = text.replace(lig, repl)

        # Attempt to fix words broken by hyphens at line breaks
        # This is a common issue in PDF text extraction.
        # Example: "experi-\nment" -> "experiment"
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
        
        # Remove remaining isolated hyphens that might be artifacts,
        # but be careful not to remove hyphens in compound words.
        # This regex looks for a hyphen surrounded by spaces or at line ends/starts
        # and might need refinement based on observed text.
        # text = re.sub(r"\s+-\s+", " ", text) # Example, might be too aggressive

        # Normalize whitespace: replace multiple spaces/tabs/newlines with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Further cleaning steps can be added here, e.g.,
        # - Removing headers/footers if patterns are identifiable (complex)
        # - Handling specific OCR errors if applicable

        return text

    def chunk_text(self, document_text: str, document_source: str, page_number: int = None) -> List[Dict[str, Any]]:
        """
        Chunks a single document's text into smaller pieces with overlap.
        This is a fixed-size chunking implementation. Semantic chunking can be more complex.
        Args:
            document_text (str): The text of the document (or a large section like a page).
            document_source (str): Identifier for the source document (e.g., PDF filename).
            page_number (int, optional): Page number if chunking page by page.
        Returns:
            List[Dict[str, Any]]: A list of chunks, where each chunk is a dictionary
                                  containing 'text', 'source_document', 'chunk_id',
                                  and optionally 'page_number'.
        """
        if not document_text:
            return []

        chunks = []
        # text_words = document_text.split() # This line is unused with character-based chunking
        
        chunk_id_counter = 0

        # Using word count for chunk_size and overlap for simplicity here.
        # For character-based, use len(text_words[i]) and string slicing.
        # The config values chunk_size/chunk_overlap are assumed to be character counts.
        # Let's adapt to character counts.

        doc_len = len(document_text)
        start_index = 0
        
        while start_index < doc_len:
            end_index = min(start_index + self.chunk_size, doc_len)
            chunk_text = document_text[start_index:end_index]
            
            chunk_metadata = {
                "text": chunk_text,
                "source_document": document_source,
                "chunk_id": f"{document_source}_p{page_number if page_number else 'doc'}_c{chunk_id_counter}",
                "start_char_offset": start_index,
                "end_char_offset": end_index
            }
            if page_number is not None:
                chunk_metadata["page_number"] = page_number
            
            chunks.append(chunk_metadata)
            chunk_id_counter += 1
            
            step = self.chunk_size - self.chunk_overlap
            if step <= 0: # Safeguard, though __init__ should prevent this
                logger.warning(f"Chunk step is non-positive ({step}). Advancing by 1 to avoid infinite loop.")
                step = 1
            start_index += step

        logger.debug(f"Chunked text from {document_source} (page {page_number if page_number else 'N/A'}) into {len(chunks)} chunks.")
        return chunks

    def process_all_pdfs_in_directory(self, pdf_directory_path: str) -> List[Dict[str, Any]]:
        """
        Loads all PDFs from a directory, extracts text, cleans it, and chunks it.
        Args:
            pdf_directory_path (str): Path to the directory containing PDFs.
        Returns:
            List[Dict[str, Any]]: A list of all text chunks from all processed PDFs.
        """
        all_chunks = []
        pdf_files = self.load_pdfs(pdf_directory_path)

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory_path}. Returning empty list of chunks.")
            return []

        for pdf_path in pdf_files:
            logger.info(f"Processing PDF: {pdf_path.name}")
            _, pages_content = self.extract_text_from_pdf(pdf_path)
            
            if not pages_content:
                logger.warning(f"No content extracted from {pdf_path.name}. Skipping.")
                continue

            for page_data in pages_content:
                page_text = page_data.get("text", "")
                page_num = page_data.get("page_number")
                if page_text:
                    page_chunks = self.chunk_text(
                        document_text=page_text,
                        document_source=pdf_path.name,
                        page_number=page_num
                    )
                    all_chunks.extend(page_chunks)
                else:
                    logger.debug(f"Skipping empty page {page_num} from {pdf_path.name}.")
        
        logger.info(f"Total chunks generated from all PDFs: {len(all_chunks)}")
        return all_chunks


if __name__ == '__main__':
    # This is an example of how to use the PDFProcessor
    # Ensure you have a 'configs/main_config.yaml' and 'data/publications_dr_kaushik/'
    # with some PDFs for this example to run.

    import sys # Add this import
    import yaml # Add this import

    # Setup basic logging for the example if not already configured globally
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    # Assuming project root is the parent of 'src'
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Load main configuration to get paths
    main_config_path = project_root / "configs" / "main_config.yaml"
    if not main_config_path.exists():
        logger.error(f"Main config not found at {main_config_path} for example usage.")
        sys.exit(1)
        
    with open(main_config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    pdf_proc_config = config_data.get("pdf_processing", {})
    processor_config_for_init = {"pdf_processing": pdf_proc_config}


    data_dir = project_root / config_data.get("data_base_dir", "data")
    pubs_dir_name = config_data.get("publications_dir", "publications_dr_kaushik")
    pdf_dir = data_dir / pubs_dir_name

    if not pdf_dir.exists():
        logger.error(f"Sample PDF directory not found: {pdf_dir}. Please create it and add PDFs.")
        # Create dummy PDF for testing if dir exists but is empty
        pdf_dir.mkdir(parents=True, exist_ok=True)
        if not list(pdf_dir.glob("*.pdf")):
             logger.warning(f"No PDFs in {pdf_dir}. PDFProcessor will not process any files.")


    processor = PDFProcessor(config=processor_config_for_init)
    
    # Example 1: Load PDF paths
    # pdf_files_paths = processor.load_pdfs(str(pdf_dir))
    # logger.info(f"Found PDF files: {pdf_files_paths}")

    # Example 2: Process a single PDF (if one exists)
    # sample_pdf_path = next(pdf_dir.glob("*.pdf"), None)
    # if sample_pdf_path:
    #     logger.info(f"\nProcessing single PDF: {sample_pdf_path.name}")
    #     full_doc_text, page_contents = processor.extract_text_from_pdf(sample_pdf_path)
    #     logger.debug(f"Full text (first 200 chars): {full_doc_text[:200]}...")
        
    #     if page_contents:
    #         first_page_text = page_contents[0]['text']
    #         logger.debug(f"First page text (first 200 chars): {first_page_text[:200]}...")
    #         chunks = processor.chunk_text(first_page_text, sample_pdf_path.name, page_contents[0]['page_number'])
    #         logger.info(f"Generated {len(chunks)} chunks from the first page.")
    #         if chunks:
    #             logger.debug(f"First chunk: {chunks[0]['text'][:100]}...")
    # else:
    #     logger.warning("No sample PDF found to demonstrate single PDF processing.")

    # Example 3: Process all PDFs in the directory
    logger.info(f"\nProcessing all PDFs in directory: {pdf_dir}")
    all_document_chunks = processor.process_all_pdfs_in_directory(str(pdf_dir))
    
    if all_document_chunks:
        logger.info(f"Total chunks from all documents: {len(all_document_chunks)}")
        logger.info(f"Sample chunk (first one):")
        logger.info(f"  Source: {all_document_chunks[0]['source_document']}")
        logger.info(f"  Page: {all_document_chunks[0].get('page_number', 'N/A')}")
        logger.info(f"  Chunk ID: {all_document_chunks[0]['chunk_id']}")
        logger.info(f"  Text: {all_document_chunks[0]['text'][:200]}...") # Print first 200 chars of the chunk
    else:
        logger.warning("No chunks were generated. Check PDF directory and content.")

    logger.info("PDFProcessor example usage finished.")
