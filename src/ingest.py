import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
import yaml
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from github import Auth, Github
from github.ContentFile import ContentFile
from github.Repository import Repository
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.file import File

from src.constants import (
    DEFAULT_CHUNK_SIZE_IN_TOKENS,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_HTTP_REQUEST_TIMEOUT,
    DEFAULT_LLAMA_STACK_RETRY_DELAY,
    DEFAULT_LLAMA_STACK_WAITING_RETRIES,
)
from src.types import (
    RAW_PIPELINES_TYPE,
    Pipeline,
    SourceConfig,
    SourceTypes,
    VectorDBConfig,
)
from src.utils import clean_text, logger


class IngestionService:
    """
    Service for ingesting documents into vector databases.
    """

    def __init__(self, config_path: "str") -> "None":
        self.vector_db_ids = None

        with open(config_path, "r") as f:
            _config: "dict[str, Any]" = yaml.safe_load(f)

            # validate ingest configuration
            if not self._config_is_valid(_config):
                logger.error("Invalid ingestion config file")
                sys.exit(1)

        self.config_path = config_path
        # Llama Stack setup
        self.llama_stack_url: "str" = _config["llamastack"]["base_url"]
        self.client = self._initialize_llama_stack_client()
        self.vector_store_ids: "list[str]" = []

        # Vector DB setup
        _embedding_dimension = (
            _config["vector_db"].get("embedding_dimension")
            if _config["vector_db"].get("embedding_dimension")
            else DEFAULT_EMBEDDING_DIMENSION
        )
        _embedding_model = (
            _config["vector_db"].get("embedding_model")
            if _config["vector_db"].get("embedding_model")
            else DEFAULT_EMBEDDING_MODEL
        )
        _chunk_size_in_tokens = (
            _config["vector_db"].get("chunk_size_in_tokens")
            if _config["vector_db"].get("chunk_size_in_tokens")
            else DEFAULT_CHUNK_SIZE_IN_TOKENS
        )
        self.vector_db_config = VectorDBConfig(
            embedding_model=_embedding_model,
            embedding_dimension=_embedding_dimension,
            chunk_size_in_tokens=_chunk_size_in_tokens,
        )
        self.file_metadata = {}

        # Document converter setup
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self.chunker = HybridChunker()

        # GitHub client setup
        gh_token = os.getenv("GITHUB_TOKEN")
        self.gh_client = Github(auth=Auth.Token(gh_token)) if gh_token else Github()
        logger.debug("Github client initialized")

        # Pipelines setup
        self.pipelines: "list[Pipeline]" = self._parse_pipelines(_config["pipelines"])

    def _parse_pipelines(self, raw_pipelines: "RAW_PIPELINES_TYPE") -> "list[Pipeline]":
        """
        parses raw pipeline configurations into Pipeline dataclass instances.
        """
        logger.debug("Parsing pipeline configurations...")
        pipelines: "list[Pipeline]" = []

        for p_title in raw_pipelines:
            logger.debug(f"Parsing pipeline {p_title}...")
            _pipeline_config: "dict[str, Any]" = raw_pipelines[p_title]
            logger.debug(f"Pipeline {p_title} type: {_pipeline_config['source']}")

            if _pipeline_config["source"] == SourceTypes.GITHUB:
                config = _pipeline_config.get("config")
                if not isinstance(config, dict):
                    logger.error(
                        f"Pipeline '{p_title}' has invalid config: expected dict"
                    )
                    continue
                url_value = config.get("url", "")
                url_str = url_value if isinstance(url_value, str) else ""
                branch_value = config.get("branch", "main")
                branch_str = branch_value if isinstance(branch_value, str) else "main"
                path_value = config.get("path", "")
                path_str = path_value if isinstance(path_value, str) else ""
                source_config = SourceConfig(
                    url=url_str,
                    branch=branch_str,
                    path=path_str,
                    urls=None,
                )
            elif _pipeline_config["source"] == SourceTypes.URL:
                urls_value = _pipeline_config.get("urls", [])
                urls_list: "list[str]" = (
                    urls_value if isinstance(urls_value, list) else []
                )
                source_config = SourceConfig(
                    url="",
                    branch="",
                    path="",
                    urls=urls_list,
                )
            else:
                logger.error(
                    f"Unknown source type '{_pipeline_config['source']}' "
                    f"in pipeline '{_pipeline_config['name']}'"
                )
                continue

            enabled_value = _pipeline_config.get("enabled", False)
            enabled_bool = (
                bool(enabled_value)
                if not isinstance(enabled_value, bool)
                else enabled_value
            )

            source_value = _pipeline_config.get("source")
            if not isinstance(source_value, str):
                logger.error(f"Pipeline '{p_title}' has invalid source type")
                continue

            pipeline = Pipeline(
                name=str(_pipeline_config["name"]),
                enabled=enabled_bool,
                version=str(_pipeline_config["version"]),
                vector_store_name=f"{_pipeline_config['vector_store_name']}",
                source=source_value,
                source_config=source_config,
            )
            pipelines.append(pipeline)
            logger.debug(f"Pipeline {p_title} sucessfully parsed!")

        return pipelines

    def _config_is_valid(self, raw_config: "dict[str, Any]") -> "bool":
        """
        loads and validates configuration.
        """
        try:
            logger.debug("Validating ingestion config...")
            assert "llamastack" in raw_config
            assert "base_url" in raw_config["llamastack"]
            assert "vector_db" in raw_config
            assert "embedding_model" in raw_config["vector_db"]
            assert "pipelines" in raw_config
        except AssertionError as e:
            logger.error(f"Configuration validation error: {e}")
            return False

        logger.debug("Ingestion config validated successfully!")
        return True

    def _initialize_llama_stack_client(
        self,
        max_retries: "int" = DEFAULT_LLAMA_STACK_WAITING_RETRIES,
        retry_delay: "int" = DEFAULT_LLAMA_STACK_RETRY_DELAY,
    ) -> "LlamaStackClient":
        logger.debug(
            f"Connecting Llama Stack Client to Server at {self.llama_stack_url}..."
        )
        _initialized = False

        for attempt in range(max_retries):
            try:
                client = LlamaStackClient(base_url=self.llama_stack_url)
                client.models.list()
                logger.debug("Llama Stack Client connected successfully!")
                _initialized = True
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.info(
                        f"Attempt {attempt + 1}/{max_retries}: "
                        f"Llama Stack not ready yet. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to connect to Llama Stack "
                        f"after {max_retries} attempts: {e}"
                    )
                    _initialized = False
        if not _initialized:
            logger.error("Failed to connect to Llama Stack. Exiting.")
            sys.exit(1)

        return client

    def _create_local_file(
        self, content: "ContentFile", path: "str", download_dir: "str"
    ) -> "str":
        logger.debug(f"Creating local file for {content.path}...")
        file_content = content.decoded_content

        relative_path = content.path
        if path and relative_path.startswith(path):
            relative_path = relative_path[len(path) :].lstrip("/")

        local_file_path = os.path.join(download_dir, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        with open(local_file_path, "wb") as f:
            f.write(file_content)

        logger.debug(f"Local file created at {content.path}")
        return local_file_path

    def _fetch_github_dir_contents(
        self,
        repo: "Repository",
        contents: "list[ContentFile]",
        branch: "str",
        path: "str",
        download_dir: "str",
    ) -> "list[str]":
        """
        fetches recursively contents from GitHub
        """
        pdf_files = []
        for content in contents:
            if content.type == "dir":
                # recursive fetch of directory contents
                try:
                    sub_contents = repo.get_contents(content.path, ref=branch)
                    if isinstance(sub_contents, list):
                        contents_list: "list[ContentFile]" = []
                        for c in sub_contents:
                            if isinstance(c, ContentFile):
                                contents_list.append(c)
                            elif hasattr(c, "type") and hasattr(c, "name"):
                                contents_list.append(c)  # type: ignore[arg-type]
                        pdf_files.extend(
                            self._fetch_github_dir_contents(
                                repo, contents_list, branch, path, download_dir
                            )
                        )
                    elif isinstance(sub_contents, ContentFile):
                        # Single file, not a directory
                        if sub_contents.name.lower().endswith(".pdf"):
                            local_file_path = self._create_local_file(
                                sub_contents, path, download_dir
                            )
                            pdf_files.append(local_file_path)
                except Exception as e:
                    logger.error(f"Error accessing directory {content.path}: {e}")
            # handle pdf files
            elif content.name.lower().endswith(".pdf"):
                try:
                    local_file_path = self._create_local_file(
                        content, path, download_dir
                    )
                    pdf_files.append(local_file_path)
                except Exception as e:
                    logger.error(f"Error downloading {content.path}: {e}")

        return pdf_files

    def _get_github_repo(self, url: "str") -> "Repository | None":
        """
        fetches PyGithub repo from GitHub URL.

        URL format: https://github.com/owner/repo or https://github.com/owner/repo.git
        """
        logger.debug(f"Getting GitHub repository from URL: {url}")
        url_parts = url.rstrip("/").rstrip(".git").split("github.com/")
        if len(url_parts) < 2:
            return None

        repo_path = url_parts[1]
        repo_parts = repo_path.split("/")
        if len(repo_parts) < 2:
            return None

        owner = repo_parts[0]
        repo_name = repo_parts[1]

        repo_title = f"{owner}/{repo_name}"

        if not repo_title:
            logger.error(f"Invalid GitHub URL: {url}")
            return None

        # Initialize GitHub client
        try:
            repo = self.gh_client.get_repo(f"{repo_title}")
            logger.debug(f"Accessed repository: {repo_title}")
            return repo
        except Exception as e:
            logger.error(f"Failed to access repository {repo_title}: {e}")
            return None

    def fetch_from_github(
        self, url: "str", path: "str", branch: "str", temp_dir: "str"
    ) -> "list[str]":
        """
        fetches documents from a GitHub repository using PyGithub API.
        """
        logger.debug(f"Fetching from GitHub: {url} (branch: {branch}, path: {path})")
        pdf_files: "list[str]" = []

        repo = self._get_github_repo(url)
        if repo is None:
            logger.error(f"Could not access repository at {url}")
            return []

        # create download directory
        download_dir = os.path.join(temp_dir, "repo")
        os.makedirs(download_dir, exist_ok=True)

        try:
            logger.debug(f"Accessing contents at path: {path if path else '/'}")
            contents = repo.get_contents(path if path else "", ref=branch)

            if isinstance(contents, list):
                contents_list: "list[ContentFile]" = []
                for c in contents:
                    if isinstance(c, ContentFile):
                        contents_list.append(c)
                    elif hasattr(c, "type") and hasattr(c, "name"):
                        contents_list.append(c)  # type: ignore[arg-type]
                pdf_files = self._fetch_github_dir_contents(
                    repo, contents_list, branch, path, download_dir
                )
            elif contents.name.lower().endswith(".pdf"):
                local_file_path = self._create_local_file(contents, path, download_dir)
                pdf_files.append(local_file_path)
                logger.debug(f"Downloaded content: {contents.path}")

        except Exception as e:
            logger.error(f"Error fetching contents from path '{path}': {e}")
            return []

        logger.debug(f"Found {len(pdf_files)} PDF files")
        return pdf_files

    def fetch_from_url(self, urls: "list[str]", temp_dir: "str") -> "list[str]":
        """
        fetches documents from direct URLs.
        """
        pdf_files: "list[str]" = []
        logger.debug(f"Fetching {len(urls)} documents from URLs")

        download_dir = os.path.join(temp_dir, "url_files")
        os.makedirs(download_dir, exist_ok=True)

        for url in urls:
            filename = os.path.basename(url.split("?")[0])
            if not filename.lower().endswith(".pdf"):
                filename += ".pdf"

            file_path = os.path.join(download_dir, filename)

            logger.debug(f"Downloading document from url: {url}")
            try:
                response = requests.get(url, timeout=DEFAULT_HTTP_REQUEST_TIMEOUT)
                response.raise_for_status()

                with open(file_path, "wb") as f:
                    f.write(response.content)

                pdf_files.append(file_path)

            except Exception as e:
                logger.error(f"Error downloading {url}: {e}")

        logger.debug(f"Downloaded {len(pdf_files)} PDF files from URLs")
        return pdf_files

    def process_documents(
        self, pdf_files: "list[str]", github_base_url="", category=""
    ) -> "list[File]":
        """
        processes PDF files into chunks using docling.
        """
        logger.info(f"Processing {len(pdf_files)} documents with docling...")
        llama_documents: "list[File]" = []

        for file_path in pdf_files:
            try:
                original_filename = os.path.basename(file_path)
                logger.info(f"Processing document: {original_filename}")
                result = self.converter.convert(file_path)

                # clean text for special characters
                markdown_text = result.document.export_to_markdown()
                cleaned_text = clean_text(markdown_text)

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                ) as tmp_file:
                    tmp_file.write(cleaned_text)
                    tmp_file_path = tmp_file.name

                try:
                    file_create_response = self.client.files.create(
                        file=Path(tmp_file_path), purpose="assistants"
                    )
                    llama_documents.append(file_create_response)

                    file_id = file_create_response.id
                    github_url = (
                        f"{github_base_url}/{original_filename}"
                        if github_base_url
                        else ""
                    )

                    self.file_metadata[file_id] = {
                        "original_filename": original_filename,
                        "github_url": github_url,
                        "category": category,
                    }
                    logger.info(f"Mapped file_id '{file_id}' -> '{original_filename}'")
                finally:
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        logger.info(f"Total documents processed: {len(llama_documents)}")
        return llama_documents

    def create_vector_db(
        self, vector_store_name: "str", documents: "list[File]"
    ) -> "bool":
        """
        creates vector database and inserts documents.
        """
        if not documents:
            logger.warning(f"No documents to insert for {vector_store_name}")
            return False

        logger.info(f"Creating vector database: {vector_store_name}")

        vector_store = None
        try:
            vector_store = self.client.vector_stores.create(name=vector_store_name)
            self.vector_store_ids.append(vector_store.id)
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower():
                logger.info(
                    f"Vector DB '{vector_store_name}' already exists, continuing..."
                )

                vector_stores = self.client.vector_stores.list() or []
                for vs in vector_stores:
                    if vs.name == vector_store_name:
                        vector_store = vs
                        break

                if vector_store is None:
                    logger.error(
                        f"Could not find existing vector store '{vector_store_name}'"
                    )
                    return False
            else:
                logger.error(f"Failed to register vector DB '{vector_store_name}': {e}")
                return False

        try:
            logger.info(f"Inserting {len(documents)}  into vector store...")
            for doc in documents:
                file_ingest_response = self.client.vector_stores.files.create(
                    vector_store_id=vector_store.id,
                    file_id=doc.id,
                )
                logger.info(
                    f"✓ Successfully inserted documents into "
                    f"'{vector_store_name}' "
                    f"with resp '{file_ingest_response}'"
                )
            return True

        except Exception as e:
            logger.error(f"Error inserting documents into '{vector_store_name}': {e}")
            return False

    def process_pipeline(self, pipeline: "Pipeline") -> "bool":
        """
        processes a single pipeline.
        """
        logger.info(f"Processing pipeline: {pipeline.name}")

        if pipeline.enabled is False:
            logger.info(f"Pipeline '{pipeline.name}' is disabled, skipping")
            return True

        vector_store_name = pipeline.vector_store_name
        source = pipeline.source

        category = (
            vector_store_name.split("-")[0] if vector_store_name else pipeline.name
        )

        github_base_url = ""
        if source == SourceTypes.GITHUB:
            github_url = pipeline.source_config.url.rstrip(".git").rstrip("/")
            github_base_url = (
                f"{github_url}/blob/"
                f"{pipeline.source_config.branch}/"
                f"{pipeline.source_config.path}".rstrip("/")
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            if source == SourceTypes.GITHUB:
                path_value = pipeline.source_config.path
                if path_value is None:
                    logger.error(
                        f"Pipeline '{pipeline.name}' has None path for GitHub source"
                    )
                    return False
                pdf_files = self.fetch_from_github(
                    pipeline.source_config.url,
                    path_value,
                    pipeline.source_config.branch,
                    temp_dir,
                )
            elif source == SourceTypes.URL:
                urls_value = pipeline.source_config.urls
                if urls_value is None:
                    logger.error(
                        f"Pipeline '{pipeline.name}' has None urls for URL source"
                    )
                    return False
                pdf_files = self.fetch_from_url(urls_value, temp_dir)
            else:
                logger.error(f"Unknown source type: {source}")
                return False

            if not pdf_files:
                logger.warning(f"No PDF files found for pipeline '{pipeline.name}'")
                return False

            documents = self.process_documents(pdf_files, github_base_url, category)

            if not documents:
                logger.warning(f"No documents processed for pipeline '{pipeline.name}'")
                return False

            return self.create_vector_db(vector_store_name, documents)

    def save_file_metadata(
        self, output_path: "str" = "rag_file_metadata.json"
    ) -> "None":
        """
        saves file metadata mapping to JSON for use by RAG service.
        """
        if not self.file_metadata:
            logger.warning("No file metadata to save")
            return

        try:
            with open(output_path, "w") as f:
                json.dump(self.file_metadata, f, indent=2)
            logger.info(
                f"Saved file metadata for {len(self.file_metadata)} "
                f"files to {output_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save file metadata: {e}")

    def run(self) -> "None":
        """
        runs the ingestion service.
        """
        logger.info("Starting RAG Ingestion Service")
        logger.info(f"Configuration: {os.path.abspath(self.config_path)}")
        total = len(self.pipelines)
        successful, failed, skipped = 0, 0, 0

        for pipeline in self.pipelines:
            if pipeline.enabled is False:
                skipped += 1
                continue

            try:
                if self.process_pipeline(pipeline):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(
                    f"Unexpected error processing pipeline '{pipeline.name}': {e}"
                )
                failed += 1

        logger.debug(f"\n{'=' * 60}")
        logger.debug("Ingestion Summary")
        logger.debug(f"{'=' * 60}")
        logger.debug(f"Total pipelines: {total}")
        logger.debug(f"Successful: {successful}")
        logger.debug(f"Failed: {failed}")
        logger.debug(f"Skipped: {skipped}")
        logger.debug(f"{'=' * 60}\n")

        if successful == 0:
            logger.warning("all pipeline(s) failed. Check logs for details.")
            sys.exit(1)
        elif failed > 0:
            logger.warning(f"{failed} pipeline(s) failed. Check logs for details.")
        else:
            logger.info("All pipelines completed successfully!")

        # Save file metadata for RAG source references
        self.save_file_metadata()
