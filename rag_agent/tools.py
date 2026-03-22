# rag_agent/tools.py
# ─────────────────────────────────────────────────────────────────────
# This file contains helper functions for creating and managing
# the Vertex AI RAG corpus. These are run once at setup time.
# ─────────────────────────────────────────────────────────────────────
 
import vertexai                          # Main Vertex AI Python SDK
from vertexai.preview import rag          # RAG Engine API (preview namespace)
import os                                 # For reading environment variables
from dotenv import load_dotenv            # Loads .env file into os.environ
 
load_dotenv()  # Execute this first — reads .env into environment
 
# Read configuration from environment
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
LOCATION   = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
 
# Initialize the Vertex AI SDK
# This sets up authentication and routing for all subsequent API calls
vertexai.init(project=PROJECT_ID, location=LOCATION)
 
 
def create_corpus(display_name: str, description: str) -> str:
    '''
    Creates a new RAG corpus in Vertex AI.
 
    Args:
        display_name: Human-readable name shown in the Cloud Console
        description:  What documents this corpus contains
 
    Returns:
        corpus_name: The full resource name used to reference this corpus.
                     Format: projects/{project}/locations/{loc}/ragCorpora/{id}
    '''
    # rag.create_corpus() makes an API call to Vertex AI RAG Engine
    # It provisions storage and sets up the embedding index
    corpus = rag.create_corpus(
        display_name=display_name,
        description=description
        # embedding_model_config defaults to text-embedding-004 (Google's best)
        # You can override with: embedding_model_config=rag.EmbeddingModelConfig(...)
    )
 
    print(f'Corpus created: {corpus.name}')
    return corpus.name
 
 
def import_files_from_gcs(corpus_name: str, gcs_uri: str) -> None:
    '''
    Imports all documents from a GCS bucket/path into the corpus.
 
    Args:
        corpus_name: Full resource name of the target corpus
        gcs_uri:     GCS path e.g. 'gs://my-bucket/' or 'gs://my-bucket/subdir/'
    '''
    # rag.import_files() reads documents from GCS, chunks them,
    # generates embeddings, and stores vectors in the corpus index.
    # This is an async operation — it returns an import response object.
    response = rag.import_files(
        corpus_name,
        [gcs_uri],   # List of GCS URIs — can import from multiple buckets
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=512,        # Max tokens per chunk (512 is a good default)
                chunk_overlap=50,      # Tokens shared between adjacent chunks
                                       # Overlap preserves context at chunk boundaries
            )
        ),
    )
    print(f'Import completed. Imported {response.imported_rag_files_count} files.')
 
 
def list_corpora() -> list:
    '''Lists all RAG corpora in your project. Useful for finding existing corpus IDs.'''
    corpora = list(rag.list_corpora())
    for c in corpora:
        print(f'  Name: {c.name}')
        print(f'  Display: {c.display_name}')
        print()
    return corpora
 
 
def delete_corpus(corpus_name: str) -> None:
    '''Permanently deletes a corpus and all its indexed documents.'''
    rag.delete_corpus(corpus_name)
    print(f'Deleted corpus: {corpus_name}')
