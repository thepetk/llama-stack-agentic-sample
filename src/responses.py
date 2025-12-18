import json
import os
from typing import Any

from llama_stack_client import LlamaStackClient
from llama_stack_client.types import ResponseObject
from llama_stack_client.types.response_list_response import (
    OutputOpenAIResponseOutputMessageFileSearchToolCallResult as FileSearchToolCallResult,  # noqa
)
from openai import OpenAI

from src.constants import (
    DEFAULT_INGESTION_CONFIG_PATHS,
    DEFAULT_LLAMA_STACK_URL,
    DEFAULT_RAG_METADATA_FILE_PATHS,
    PIPELINE_CATEGORIES,
)
from src.exceptions import NoVectorStoresFoundError
from src.types import Pipeline, SourceTypes
from src.utils import logger


class RAGService:
    """
    service for managing RAG vector stores and providing retrieval capabilities.
    """

    def __init__(
        self,
        pipelines: "list[Pipeline]",
        llama_stack_url: "str" = DEFAULT_LLAMA_STACK_URL,
        ingestion_config_path: "str | None" = None,
        file_metadata_path: "str | None" = None,
    ) -> "None":
        self.llama_stack_url = llama_stack_url
        self.client: "LlamaStackClient | None" = None
        self.openai_client: "OpenAI | None" = None
        self.vector_store_map: "dict[str, list[str]]" = {}
        self.all_vector_store_ids: "list[str]" = []

        # source URL mapping: category -> GitHub base URL for PDFs
        self.source_url_map: "dict[str, str]" = {}
        self.ingestion_config: "dict[str, Any]" = {}

        # file metadata: file_id -> {original_filename, github_url, category}
        self.file_metadata: "dict[str, dict[str, str]]" = {}

        # load ingestion config if available
        if ingestion_config_path is None:
            for path in DEFAULT_INGESTION_CONFIG_PATHS:
                if os.path.exists(path):
                    ingestion_config_path = path
                    break

        # Always load source URL map from pipelines if provided
        if pipelines:
            self._load_source_url_map(pipelines)

        # load file metadata if available
        if file_metadata_path is None:
            for path in DEFAULT_RAG_METADATA_FILE_PATHS:
                if os.path.exists(path):
                    file_metadata_path = path
                    break

        if file_metadata_path and os.path.exists(file_metadata_path):
            self._load_file_metadata(file_metadata_path)

    def _load_file_metadata(self, metadata_path: "str") -> "None":
        """
        loads file metadata mapping from JSON.
        """
        try:
            with open(metadata_path, "r") as f:
                self.file_metadata = json.load(f)
            logger.info(
                f"RAG Service: Loaded metadata for "
                f"{len(self.file_metadata)} files from {metadata_path}"
            )
        except Exception as e:
            logger.warning(f"RAG Service: Could not load file metadata: {e}")

    def _load_source_url_map(self, pipelines: "list[Pipeline]") -> "dict[str, str]":
        """
        loads pipelines to map vector stores to source URLs.
        """
        for pipeline in pipelines:
            if not pipeline.enabled:
                continue

            if pipeline.source == SourceTypes.GITHUB:
                url = pipeline.source_config.url
                branch = pipeline.source_config.branch
                path = pipeline.source_config.path or ""

                # GitHub blob URL format: https://github.com/{org}/{repo}/blob/{branch}/{path}
                base_url = f"{url}/blob/{branch}/{path}".rstrip("/")

                category = (
                    pipeline.vector_store_name.split("-")[0]
                    if pipeline.vector_store_name
                    else ""
                )

                if category:
                    self.source_url_map[category] = base_url
                    logger.info(
                        f"RAG Service: Mapped category '{category}' "
                        f"to source URL: {base_url}"
                    )

        logger.info(
            f"RAG Service: Loaded source mappings for "
            f"{len(self.source_url_map)} categories"
        )
        return self.source_url_map

    def initialize(self) -> "bool":
        """
        initializes the Llama Stack client and load vector stores.
        """
        try:
            self.client = LlamaStackClient(base_url=self.llama_stack_url)
            self.client.models.list()
            logger.info("RAG Service: Llama Stack client initialized successfully")

            # Initialize OpenAI client for structured output
            # (points to LlamaStack's OpenAI-compatible endpoint)
            openai_base_url = f"{self.llama_stack_url}/v1"
            self.openai_client = OpenAI(
                base_url=openai_base_url,
                api_key="not-needed",  # LlamaStack doesn't require API key
            )
            logger.info(
                f"RAG Service: OpenAI client initialized (base_url={openai_base_url})"
            )

            return self.load_vector_stores()
        except Exception as e:
            logger.error(f"RAG Service: Failed to initialize: {e}")
        return False

    def map_to_category(self, category: "str", vs_name: "str", vs_id: "str") -> "None":
        """
        maps to categories based on naming conventions from ingestion-config.yaml
        """
        if category not in vs_name:
            return

        if category not in self.vector_store_map:
            self.vector_store_map[category] = []
        self.vector_store_map[category].append(vs_id)
        logger.debug(f"RAG Service: Mapped '{vs_name}' -> '{category}'")

    def check_vector_stores_exist(self, pipelines: "list[Pipeline]") -> "bool":
        """
        checks if all required vector stores exist based on enabled pipelines.
        """
        if not self.client:
            logger.error("RAG Service: Client not initialized")
            return False

        expected_stores = set()
        for pipeline in pipelines:
            if pipeline.enabled:
                expected_stores.add(pipeline.vector_store_name.lower())

        if not expected_stores:
            logger.warning("RAG Service: No enabled pipelines found")
            return False

        vector_stores = self.client.vector_stores.list() or []
        existing_stores = set()
        for vs in vector_stores:
            vs_name = vs.name.lower() if vs.name else vs.id.lower()
            existing_stores.add(vs_name)

        missing_stores = expected_stores - existing_stores

        if missing_stores:
            logger.info(f"RAG Service: Missing vector stores: {missing_stores}")
            return False

        logger.info(
            f"RAG Service: All {len(expected_stores)} required vector stores exist"
        )
        return True

    def load_vector_stores(self) -> "bool":
        """
        loads vector stores from Llama Stack and map them by category.

        raises:: NoVectorStoresFoundError if no vector stores are found.
        """
        if not self.client:
            logger.error("RAG Service: Client not initialized")
            return False

        vector_stores = self.client.vector_stores.list() or []
        vector_store_list = list(vector_stores)

        if not vector_stores or len(vector_store_list) == 0:
            err = "RAG Service: No vector stores found"
            logger.warning(err)
            raise NoVectorStoresFoundError(err)

        for vs in vector_stores:
            vs_name_lower = vs.name.lower() if vs.name else vs.id.lower()
            vs_id = vs.id
            self.all_vector_store_ids.append(vs_id)

            for category in PIPELINE_CATEGORIES:
                self.map_to_category(category, vs_name_lower, vs_id)

        logger.info(
            f"RAG Service: Loaded {len(self.all_vector_store_ids)} vector stores"
        )
        logger.info(
            f"RAG Service: Categories available: {list(self.vector_store_map.keys())}"
        )

        return len(self.all_vector_store_ids) > 0

    def get_vector_store_ids(self, category: "str | None" = None) -> "list[str]":
        """
        gets vector store IDs for a specific category or all stores.
        """
        if category:
            return self.vector_store_map.get(category, [])
        return self.all_vector_store_ids

    def get_file_search_tool(
        self, category: "str | None" = None
    ) -> "dict[str, Any] | None":
        """
        gets file_search tool configuration for OpenAI responses API.
        This tool format works with both OpenAI SDK and Llama Stack's
        OpenAI-compatible endpoint.
        """
        vector_store_ids = self.get_vector_store_ids(category)

        if not vector_store_ids:
            logger.warning(f"RAG Service: No vector stores for category '{category}'")
            return None

        return {"type": "file_search", "vector_store_ids": vector_store_ids}

    def get_source_url(self, category: "str", filename: "str") -> "str":
        """
        constructs the GitHub URL for a source file.
        """
        base_url = self.source_url_map.get(category, "")
        if base_url and filename:
            return f"{base_url}/{filename}"
        return ""

    def _get_file_id(self, result: "FileSearchToolCallResult") -> "str | None":
        """
        helper to extract file_id from a result object.
        """
        if hasattr(result, "file_id"):
            return result.file_id
        elif hasattr(result, "id"):
            return result.id
        elif hasattr(result, "filename"):
            return result.filename
        elif hasattr(result, "name"):
            return result.name
        elif isinstance(result, dict):
            return (
                result.get("file_id")
                or result.get("id")
                or result.get("filename")
                or result.get("name")
            )
        return None

    def extract_sources_from_response(
        self, rag_response: "ResponseObject", category: "str"
    ) -> "list[dict[str, str]]":
        """
        extracts source document references from a RAG response.
        Uses the file_metadata mapping to get original filenames and GitHub URLs.
        Returns a list of dicts with 'filename', 'url', and optionally 'snippet'.
        """
        sources = []
        seen_files = set()

        for output_item in rag_response.output:
            if not (
                hasattr(output_item, "type") and output_item.type == "file_search_call"
            ):
                continue

            results = getattr(output_item, "results", None)
            if not results:
                continue

            for result in results:
                snippet = None
                file_id = self._get_file_id(result)

                if hasattr(result, "text") and result.text:
                    text = str(result.text)
                    snippet = text[:200] + "..." if len(text) > 200 else text
                elif isinstance(result, dict) and result.get("text"):
                    text = str(result.get("text", ""))
                    snippet = text[:200] + "..." if len(text) > 200 else text
                else:
                    snippet = None

                if not (file_id and file_id not in seen_files):
                    continue

                seen_files.add(file_id)

                if file_id in self.file_metadata:
                    metadata = self.file_metadata[file_id]
                    sources.append(
                        {
                            "filename": metadata.get("original_filename", file_id),
                            "url": metadata.get("github_url", ""),
                            "snippet": snippet or "",
                        }
                    )
                    logger.debug(
                        f"Found source: {metadata.get('original_filename')} "
                        f"-> {metadata.get('github_url')}"
                    )
                else:
                    source_url = self.get_source_url(category, file_id)
                    sources.append(
                        {
                            "filename": file_id,
                            "url": source_url,
                            "snippet": snippet or "",
                        }
                    )

        if not sources:
            sources = self._get_files_from_vector_store(category)

        return sources

    def _get_files_from_vector_store(self, category: "str") -> "list[dict[str, str]]":
        """
        get all files from the vector store for a category.
        """
        if not self.client:
            return []

        sources = []
        seen_files = set()
        vector_store_ids = self.get_vector_store_ids(category)

        for vs_id in vector_store_ids:
            try:
                files = self.client.vector_stores.files.list(vector_store_id=vs_id)
            except Exception as e:
                logger.debug(f"Could not list files for vector store {vs_id}: {e}")
                continue

            if not files:
                continue

            for file_info in files:
                file_id = getattr(file_info, "id", None) or getattr(
                    file_info, "file_id", None
                )

                if file_id and file_id not in seen_files:
                    seen_files.add(file_id)

                    # Look up in file_metadata
                    if file_id in self.file_metadata:
                        metadata = self.file_metadata[file_id]
                        sources.append(
                            {
                                "filename": metadata.get("original_filename", file_id),
                                "url": metadata.get("github_url", ""),
                                "snippet": "",
                            }
                        )
                        # next iteration if found in metadata
                        continue

                    filename = getattr(file_info, "filename", None) or file_id
                    source_url = self.get_source_url(category, filename)
                    sources.append(
                        {
                            "filename": filename,
                            "url": source_url,
                            "snippet": "",
                        }
                    )

        return sources

    def query_vectors_directly(
        self, query: "str", category: "str | None" = None, max_chunks: "int" = 5
    ) -> "str":
        """
        an alternative RAG approach: Query vector stores directly and build context.
        This is the second approach shown in ingest_openai.py (lines 420-444).
        Returns context string that can be prepended to prompts.
        """
        if not self.client:
            return ""

        vector_store_ids = self.get_vector_store_ids(category)

        if not vector_store_ids:
            logger.warning(f"RAG Service: No vector stores for category '{category}'")
            return ""

        all_chunks = []
        for vector_db_id in vector_store_ids:
            try:
                query_results = self.client.vector_io.query(
                    vector_db_id=vector_db_id,
                    query=query,
                    params={"max_chunks": max_chunks},
                )
                all_chunks.extend(query_results.chunks)
            except Exception as e:
                logger.error(f"RAG Service: Error querying {vector_db_id}: {e}")

        if not all_chunks:
            return ""

        # Build context from chunks
        context = "\n\n".join([chunk.content for chunk in all_chunks])
        logger.info(f"RAG Service: Retrieved {len(all_chunks)} chunks for query")

        return context
