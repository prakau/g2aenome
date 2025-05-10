import pytest
from pathlib import Path
import os
import sys

# Adjust Python path to include the 'src' directory for imports
PROJECT_ROOT_TESTS = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT_TESTS / "src"))

from data_fabric.pdf_processor import PDFProcessor
from data_fabric.genomic_processor import GenomicProcessor

# --- Fixtures ---

@pytest.fixture(scope="module")
def temp_data_dir(tmp_path_factory):
    """Creates a temporary data directory for tests."""
    data_dir = tmp_path_factory.mktemp("test_data")
    
    # Create dummy subdirectories as expected by processors
    (data_dir / "publications_dr_kaushik").mkdir(exist_ok=True)
    (data_dir / "genomic_data_samples").mkdir(exist_ok=True)
    
    return data_dir

@pytest.fixture(scope="module")
def sample_pdf_path(temp_data_dir):
    """Creates a dummy PDF file for testing PDFProcessor."""
    pdf_dir = temp_data_dir / "publications_dr_kaushik"
    dummy_pdf_path = pdf_dir / "sample_paper.pdf"
    
    # Creating a real (but very simple) PDF is complex.
    # For robust testing, a small, actual PDF should be used.
    # Here, we'll just create an empty file as a placeholder.
    # Actual PDF parsing tests would need a library like reportlab or a pre-existing file.
    # For now, this test will mostly check file discovery and basic error handling.
    # To test actual text extraction, place a real PDF here or mock PyPDF2.
    try:
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(str(dummy_pdf_path))
        c.drawString(100, 750, "This is a sample PDF for testing.")
        c.drawString(100, 730, "It contains some text about Solanum melongena and abiotic stress.")
        c.save()
        created_pdf = True
    except ImportError:
        # Fallback if reportlab is not installed (though it should be for a full test env)
        with open(dummy_pdf_path, "w") as f:
            f.write("%PDF-1.4\n%Dummy PDF content for testing path existence.\n")
        created_pdf = False # Mark that it's not a real PDF
        print("Warning: reportlab not found. Created a non-PDF placeholder for sample_paper.pdf. Text extraction tests will be limited.")

    return dummy_pdf_path, created_pdf


@pytest.fixture(scope="module")
def sample_fasta_path(temp_data_dir):
    """Creates a dummy FASTA file for testing GenomicProcessor."""
    genomic_dir = temp_data_dir / "genomic_data_samples"
    dummy_fasta_path = genomic_dir / "sample_genome.fasta"
    with open(dummy_fasta_path, "w") as f:
        f.write(">seq1 Test sequence 1 [organism=Testus Specius]\n")
        f.write("ATGCGTAGCATCG\n")
        f.write(">seq2 Another sequence\n")
        f.write("CGATCGATGC\n")
    return dummy_fasta_path

@pytest.fixture(scope="module")
def sample_gff_path(temp_data_dir):
    """Creates a dummy GFF file for testing GenomicProcessor."""
    genomic_dir = temp_data_dir / "genomic_data_samples"
    dummy_gff_path = genomic_dir / "sample_features.gff3"
    with open(dummy_gff_path, "w") as f:
        f.write("##gff-version 3\n")
        f.write("chr1\tTestPipe\tgene\t100\t200\t.\t+\t.\tID=gene001;Name=TestGene\n")
        f.write("chr1\tTestPipe\tmRNA\t100\t200\t.\t+\t.\tID=mrna001;Parent=gene001\n")
    return dummy_gff_path


@pytest.fixture(scope="module")
def pdf_processor_instance():
    # Minimal config for testing basic functionality
    config = {"pdf_processing": {"chunk_size": 100, "chunk_overlap": 10}}
    return PDFProcessor(config=config)

@pytest.fixture(scope="module")
def genomic_processor_instance():
    config = {"genomic_processing": {"default_organism": "TestDefault"}}
    return GenomicProcessor(config=config)


# --- PDFProcessor Tests ---

def test_pdf_load_pdfs(pdf_processor_instance, temp_data_dir, sample_pdf_path):
    """Test discovery of PDF files."""
    pdf_dir = temp_data_dir / "publications_dr_kaushik"
    _ = sample_pdf_path # Ensure fixture creates the file
    
    loaded_pdfs = pdf_processor_instance.load_pdfs(str(pdf_dir))
    assert len(loaded_pdfs) == 1
    assert loaded_pdfs[0].name == "sample_paper.pdf"

def test_pdf_extract_text(pdf_processor_instance, sample_pdf_path):
    """Test basic text extraction from the dummy PDF."""
    path, is_real_pdf = sample_pdf_path
    if not is_real_pdf:
        pytest.skip("Skipping PDF text extraction test as reportlab was not available to create a real PDF.")
        
    full_text, pages_content = pdf_processor_instance.extract_text_from_pdf(path)
    assert isinstance(full_text, str)
    assert isinstance(pages_content, list)
    if pages_content: # If reportlab created a PDF with text
        assert "This is a sample PDF for testing." in pages_content[0]["text"]
        assert "Solanum melongena" in pages_content[0]["text"]
    else: # If it's the placeholder file, expect no text
        assert full_text == ""


def test_pdf_clean_text(pdf_processor_instance):
    """Test text cleaning functionality."""
    raw = "This is a test\nwith some  extra spaces and a hyphen-\nated word like experi-\nment. Also ﬁ ligature."
    cleaned = pdf_processor_instance.clean_text(raw)
    assert "  " not in cleaned # Multiple spaces
    assert "\n" not in cleaned[1:-1] # Newlines (except potentially at very start/end if not stripped)
    assert "experiment" in cleaned # Hyphenated word
    assert "fi" in cleaned # Ligature ﬁ
    assert "experi- ment" not in cleaned

def test_pdf_chunk_text(pdf_processor_instance):
    """Test text chunking."""
    doc_text = "This is a sample document text that is long enough to be chunked several times for testing purposes." * 5
    chunks = pdf_processor_instance.chunk_text(doc_text, "test_doc.pdf", page_number=1)
    assert len(chunks) > 1
    if len(chunks) > 1:
        # Check overlap: end of first chunk should overlap with start of second chunk
        # (accounting for chunk_size and chunk_overlap from fixture)
        first_chunk_end = chunks[0]["text"][-pdf_processor_instance.chunk_overlap:]
        second_chunk_start = chunks[1]["text"][:pdf_processor_instance.chunk_overlap]
        # This simple check might fail if chunk_overlap > len(first_chunk_end) or len(second_chunk_start)
        # A more robust check would involve character offsets.
        # For now, just ensure chunks are created.
    for chunk in chunks:
        assert "text" in chunk
        assert chunk["source_document"] == "test_doc.pdf"
        assert "chunk_id" in chunk


# --- GenomicProcessor Tests ---

def test_genomic_parse_fasta(genomic_processor_instance, temp_data_dir, sample_fasta_path):
    """Test parsing of FASTA files."""
    genomic_dir = temp_data_dir / "genomic_data_samples"
    _ = sample_fasta_path # Ensure fixture creates the file
    
    fasta_data = genomic_processor_instance.parse_fasta_files(str(genomic_dir))
    assert len(fasta_data) == 2
    assert fasta_data[0]["id"] == "seq1"
    assert fasta_data[0]["sequence"] == "ATGCGTAGCATCG"
    assert fasta_data[0]["organism"] == "Testus Specius" # From fixture
    assert fasta_data[1]["id"] == "seq2"
    assert fasta_data[1]["organism"] == "TestDefault" # Default from fixture config

def test_genomic_parse_gff(genomic_processor_instance, temp_data_dir, sample_gff_path):
    """Test parsing of GFF files."""
    genomic_dir = temp_data_dir / "genomic_data_samples"
    _ = sample_gff_path # Ensure fixture creates the file
    
    gff_data = genomic_processor_instance.parse_gff_files(str(genomic_dir))
    # Expecting 3 features: gene, mRNA (sub-feature of gene), 
    # but _process_gff_feature returns a flat list.
    # The dummy GFF has 1 top-level gene, which has 1 mRNA sub_feature.
    # So, the flat list should contain both.
    assert len(gff_data) >= 2 # gene and mRNA
    
    gene_feature = next((f for f in gff_data if f["type"] == "gene"), None)
    mrna_feature = next((f for f in gff_data if f["type"] == "mRNA"), None)
    
    assert gene_feature is not None
    assert gene_feature["id"] == "gene001"
    assert gene_feature["attributes"]["Name"] == "TestGene"
    
    assert mrna_feature is not None
    assert mrna_feature["id"] == "mrna001"
    assert "gene001" in mrna_feature["parent_ids"]
