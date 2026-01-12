# DEFAULT_EMBEDDING_DIMENSION: The default dimension
# size for vector embeddings.
DEFAULT_EMBEDDING_DIMENSION = 128

# DEFAULT_CHUNK_SIZE_IN_TOKENS: The default size of text
# chunks in tokens for processing.
DEFAULT_CHUNK_SIZE_IN_TOKENS = 512

# DEFAULT_LLAMA_STACK_URL: The default URL for
# connecting to the Llama Stack service.
DEFAULT_LLAMA_STACK_URL = "http://localhost:8321"

# DEFAULT_LLAMA_STACK_WAITING_RETRIES: The default number
# of retries for waiting operations in Llama Stack.
DEFAULT_LLAMA_STACK_WAITING_RETRIES = 2

# DEFAULT_LLAMA_STACK_RETRY_DELAY: The default delay in
# seconds between retries in Llama Stack.
DEFAULT_LLAMA_STACK_RETRY_DELAY = 5

# DEFAULT_HTTP_REQUEST_TIMEOUT: The default timeout in
# seconds for HTTP requests.
DEFAULT_HTTP_REQUEST_TIMEOUT = 60

# DEFAULT_INFERENCE_MODEL: The default inference model
# used by the RAGService.
DEFAULT_INFERENCE_MODEL = "vllm/redhataiqwen3-8b-fp8-dynamic"

# DEFAULT_GUARDRAIL_MODEL: The default guardrail model
# used for response validation.
DEFAULT_GUARDRAIL_MODEL = "ollama/llama-guard3:8b"

# DEFAULT_EMBEDDING_MODEL: The default embedding model
# used for generating vector embeddings.
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

# DEFAULT_INGESTION_MODE: The default mode for document ingestion.
# "sync" = sequential (slower but more stable, won't overwhelm embedding model)
# "async" = concurrent (faster but may crash embedding model under load)
DEFAULT_INGESTION_MODE = "sync"

# DEFAULT_MCP_TOOL_MODEL: The default model used for
# MCP tool calls.
DEFAULT_MCP_TOOL_MODEL = "vllm/redhataiqwen3-8b-fp8-dynamic"

# DEFAULT_INGESTION_CONFIG: The default path to the ingestion
# configuration file.
DEFAULT_INGESTION_CONFIG = "config/ingestion-config.yaml"

# DEFAULT_INGESTION_CONFIG_PATHS: The default paths to look for
# the ingestion configuration file.
DEFAULT_INGESTION_CONFIG_PATHS = [
    "ingestion-config.yaml",
    "/config/ingestion-config.yaml",
]

# DEFAULT_RAG_METADATA_FILE_PATHS: The default file location
# for storing RAG metadata.
DEFAULT_RAG_METADATA_FILE_PATHS = [
    "rag_file_metadata.json",
    "/config/rag_file_metadata.json",
]

# PIPELINE_CATEGORIES: The list of pipeline categories
PIPELINE_CATEGORIES = ["legal", "techsupport", "hr", "sales", "procurement"]

# NO_DOCS_INDICATORS: Explicit statements for irelevant info
NO_DOCS_INDICATORS = [
    "no relevant document",
    "could not find any relevant",
    "couldn't find any relevant",
    "did not find any relevant",
    "didn't find any relevant",
    "no information found in",
    "no matching document",
    "unable to find relevant",
    "not found in the knowledge base",
    "no results found",
    # Topic mismatch indicators
    "nothing related to",
    "doesn't have any info on",
    "does not have any info on",
    "documents are unrelated",
    "documents don't cover",
    "documents do not cover",
    "aren't relevant to",
    "are not relevant to",
    "isn't relevant to",
    "is not relevant to",
    "unrelated to your query",
    "unrelated to the query",
    "wrong topic",
    "different topic",
    "available resources don't cover",
    "available documents don't cover",
    "knowledge base doesn't have",
    "knowledge base does not have",
    "knowledge base doesn't cover",
    "knowledge base does not cover",
]
