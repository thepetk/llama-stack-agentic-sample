import os
from unittest.mock import Mock, patch

import pytest

from src.exceptions import NoVectorStoresFoundError
from src.responses import RAGService
from src.types import Pipeline, SourceConfig, SourceTypes


class TestRAGServiceInit:
    """
    tests for RAGService initialization.
    """

    def test_init_with_default_params(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)

        assert service.llama_stack_url == "http://localhost:8321"
        assert service.client is None
        assert service.vector_store_map == {}
        assert service.all_vector_store_ids == []
        assert isinstance(service.file_metadata, dict)

    def test_init_with_custom_url(self):
        pipelines = []
        service = RAGService(pipelines=pipelines, llama_stack_url="http://custom:9000")

        assert service.llama_stack_url == "http://custom:9000"

    def test_init_with_ingestion_config(self, sample_config_file):
        pipelines = []
        service = RAGService(
            pipelines=pipelines, ingestion_config_path=sample_config_file
        )

        assert service.source_url_map == {}

    def test_init_with_file_metadata(self, sample_metadata_file):
        pipelines = []
        service = RAGService(
            pipelines=pipelines, file_metadata_path=sample_metadata_file
        )

        assert len(service.file_metadata) > 0
        assert "file-id-1" in service.file_metadata


class TestLoadFileMetadata:
    """
    tests for _load_file_metadata method.
    """

    def test_load_file_metadata_success(self, sample_metadata_file):
        pipelines = []
        service = RAGService(
            pipelines=pipelines, file_metadata_path=sample_metadata_file
        )

        assert "file-id-1" in service.file_metadata
        assert service.file_metadata["file-id-1"]["original_filename"] == "test.pdf"
        assert service.file_metadata["file-id-1"]["category"] == "legal"

    def test_load_file_metadata_with_nonexistent_file(self):
        pipelines = []
        service = RAGService(
            pipelines=pipelines, file_metadata_path="/nonexistent/path.json"
        )

        assert service.file_metadata == {}

    def test_load_file_metadata_with_invalid_json(self, temp_dir):
        invalid_json_path = os.path.join(temp_dir, "invalid.json")
        with open(invalid_json_path, "w") as f:
            f.write("not valid json")

        pipelines = []
        service = RAGService(pipelines=pipelines, file_metadata_path=invalid_json_path)

        assert service.file_metadata == {}


class TestLoadSourceUrlMap:
    """
    tests for _load_source_url_map method.
    """

    def test_load_source_url_map_with_github_pipeline(self):
        pipelines = [
            Pipeline(
                name="legal",
                enabled=True,
                version="v1.0",
                vector_store_name="legal-vector-db",
                source=SourceTypes.GITHUB,
                source_config=SourceConfig(
                    url="https://github.com/test/repo",
                    branch="main",
                    path="docs",
                    urls=None,
                ),
            )
        ]

        service = RAGService(pipelines=pipelines, ingestion_config_path="test.yaml")

        assert "legal" in service.source_url_map
        assert (
            "https://github.com/test/repo/blob/main/docs"
            in service.source_url_map["legal"]
        )

    def test_load_source_url_map_with_disabled_pipeline(self):
        pipelines = [
            Pipeline(
                name="legal",
                enabled=False,
                version="v1.0",
                vector_store_name="legal-vector-db",
                source=SourceTypes.GITHUB,
                source_config=SourceConfig(
                    url="https://github.com/test/repo",
                    branch="main",
                    path="",
                    urls=None,
                ),
            )
        ]

        service = RAGService(pipelines=pipelines, ingestion_config_path="test.yaml")

        assert "legal" not in service.source_url_map

    def test_load_source_url_map_with_url_pipeline(self):
        pipelines = [
            Pipeline(
                name="test",
                enabled=True,
                version="v1.0",
                vector_store_name="test-vector-db",
                source=SourceTypes.URL,
                source_config=SourceConfig(
                    url="", branch="", path="", urls=["https://example.com/test.pdf"]
                ),
            )
        ]

        service = RAGService(pipelines=pipelines, ingestion_config_path="test.yaml")

        assert service.source_url_map == {}


class TestInitialize:
    """
    tests for initialize method.
    """

    def test_initialize_success(self, mock_llama_stack_client):
        pipelines = []
        service = RAGService(pipelines=pipelines)

        with patch(
            "src.responses.LlamaStackClient", return_value=mock_llama_stack_client
        ):
            with patch("src.responses.OpenAI") as mock_openai:
                with patch.object(service, "load_vector_stores", return_value=True):
                    result = service.initialize()

                    assert result is True
                    assert service.client is not None
                    mock_openai.assert_called_once()

    def test_initialize_failure(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)

        with patch(
            "src.responses.LlamaStackClient",
            side_effect=Exception("Connection error"),
        ):
            result = service.initialize()

            assert result is False


class TestLoadVectorStores:
    """
    tests for load_vector_stores method.
    """

    def test_load_vector_stores_success(self, mock_llama_stack_client):
        mock_vs1 = Mock()
        mock_vs1.id = "legal-vs-id"
        mock_vs1.name = "legal-vector-db"

        mock_vs2 = Mock()
        mock_vs2.id = "hr-vs-id"
        mock_vs2.name = "hr-vector-db"

        mock_llama_stack_client.vector_stores.list.return_value = [mock_vs1, mock_vs2]

        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = mock_llama_stack_client

        result = service.load_vector_stores()

        assert result is True
        assert len(service.all_vector_store_ids) == 2
        assert "legal" in service.vector_store_map
        assert "hr" in service.vector_store_map

    def test_load_vector_stores_with_no_stores(self, mock_llama_stack_client):
        mock_llama_stack_client.vector_stores.list.return_value = []

        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = mock_llama_stack_client

        with pytest.raises(NoVectorStoresFoundError) as exc_info:
            service.load_vector_stores()

        assert "No vector stores found" in str(exc_info.value)

    def test_load_vector_stores_without_client(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = None

        result = service.load_vector_stores()

        assert result is False


class TestMapToCategory:
    """
    tests for map_to_category method.
    """

    def test_map_to_category_success(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)

        service.map_to_category("legal", "legal-vector-db", "legal-vs-id")

        assert "legal" in service.vector_store_map
        assert "legal-vs-id" in service.vector_store_map["legal"]

    def test_map_to_category_with_non_matching_name(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)

        service.map_to_category("legal", "hr-vector-db", "hr-vs-id")

        assert "legal" not in service.vector_store_map

    def test_map_to_category_multiple_stores(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)

        service.map_to_category("legal", "legal-vector-db-v1", "legal-vs-id-1")
        service.map_to_category("legal", "legal-vector-db-v2", "legal-vs-id-2")

        assert len(service.vector_store_map["legal"]) == 2


class TestGetVectorStoreIds:
    """
    tests for get_vector_store_ids method.
    """

    def test_get_vector_store_ids_for_category(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.vector_store_map = {"legal": ["legal-vs-1", "legal-vs-2"]}

        result = service.get_vector_store_ids("legal")

        assert result == ["legal-vs-1", "legal-vs-2"]

    def test_get_vector_store_ids_for_nonexistent_category(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.vector_store_map = {"legal": ["legal-vs-1"]}

        result = service.get_vector_store_ids("hr")

        assert result == []

    def test_get_all_vector_store_ids(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.all_vector_store_ids = ["vs-1", "vs-2", "vs-3"]

        result = service.get_vector_store_ids()

        assert result == ["vs-1", "vs-2", "vs-3"]


class TestGetFileSearchTool:
    """
    tests for get_file_search_tool method.
    """

    def test_get_file_search_tool_success(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.vector_store_map = {"legal": ["legal-vs-1"]}

        result = service.get_file_search_tool("legal")

        assert result is not None
        assert result["type"] == "file_search"
        assert result["vector_store_ids"] == ["legal-vs-1"]

    def test_get_file_search_tool_with_no_stores(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.vector_store_map = {}

        result = service.get_file_search_tool("legal")

        assert result is None


class TestGetSourceUrl:
    """
    tests for get_source_url method.
    """

    def test_get_source_url_success(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.source_url_map = {
            "legal": "https://github.com/test/repo/blob/main/docs"
        }

        result = service.get_source_url("legal", "test.pdf")

        assert result == "https://github.com/test/repo/blob/main/docs/test.pdf"

    def test_get_source_url_with_nonexistent_category(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.source_url_map = {}

        result = service.get_source_url("legal", "test.pdf")

        assert result == ""

    def test_get_source_url_with_empty_filename(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.source_url_map = {
            "legal": "https://github.com/test/repo/blob/main/docs"
        }

        result = service.get_source_url("legal", "")

        assert result == ""


class TestGetFileId:
    """
    tests for _get_file_id method.
    """

    def test_get_file_id_from_file_id_attribute(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)

        mock_result = Mock()
        mock_result.file_id = "test-file-id"

        result = service._get_file_id(mock_result)

        assert result == "test-file-id"

    def test_get_file_id_from_id_attribute(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)

        mock_result = Mock(spec=["id"])
        mock_result.id = "test-id"

        result = service._get_file_id(mock_result)

        assert result == "test-id"

    def test_get_file_id_from_dict(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)

        result_dict = {"file_id": "test-file-id"}

        result = service._get_file_id(result_dict)

        assert result == "test-file-id"

    def test_get_file_id_returns_none(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)

        mock_result = Mock(spec=[])

        result = service._get_file_id(mock_result)

        assert result is None


class TestExtractSourcesFromResponse:
    """
    tests for extract_sources_from_response method.
    """

    def test_extract_sources_with_file_search_call(self, sample_metadata_file):
        pipelines = []
        service = RAGService(
            pipelines=pipelines, file_metadata_path=sample_metadata_file
        )

        mock_result = Mock()
        mock_result.file_id = "file-id-1"
        mock_result.text = "Sample text from document"

        mock_output_item = Mock()
        mock_output_item.type = "file_search_call"
        mock_output_item.results = [mock_result]

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        sources = service.extract_sources_from_response(mock_response, "legal")

        assert len(sources) > 0
        assert sources[0]["filename"] == "test.pdf"
        assert "github.com" in sources[0]["url"]

    def test_extract_sources_with_no_results(self, sample_metadata_file):
        pipelines = []
        service = RAGService(
            pipelines=pipelines, file_metadata_path=sample_metadata_file
        )

        mock_output_item = Mock()
        mock_output_item.type = "file_search_call"
        mock_output_item.results = None

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        with patch.object(service, "_get_files_from_vector_store", return_value=[]):
            sources = service.extract_sources_from_response(mock_response, "legal")

            assert sources == []

    def test_extract_sources_with_unknown_file_id(self):
        """Test that unknown file_id uses friendly fallback instead of raw file_id."""
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.source_url_map = {"legal": "https://github.com/test/repo/blob/main/docs"}

        mock_result = Mock()
        mock_result.file_id = "unknown-file-id"

        mock_output_item = Mock()
        mock_output_item.type = "file_search_call"
        mock_output_item.results = [mock_result]

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        sources = service.extract_sources_from_response(mock_response, "legal")

        assert len(sources) > 0
        # Now uses friendly fallback instead of raw file_id
        assert sources[0]["filename"] == "Document from legal knowledge base"
        assert sources[0]["url"] == "https://github.com/test/repo/blob/main/docs"


class TestGetFilesFromVectorStore:
    """
    tests for _get_files_from_vector_store method.
    """

    def test_get_files_from_vector_store_success(
        self, mock_llama_stack_client, sample_metadata_file
    ):
        pipelines = []
        service = RAGService(
            pipelines=pipelines, file_metadata_path=sample_metadata_file
        )
        service.client = mock_llama_stack_client
        service.vector_store_map = {"legal": ["legal-vs-id"]}

        mock_file = Mock()
        mock_file.id = "file-id-1"
        mock_llama_stack_client.vector_stores.files.list.return_value = [mock_file]

        sources = service._get_files_from_vector_store("legal")

        assert len(sources) > 0
        assert sources[0]["filename"] == "test.pdf"

    def test_get_files_from_vector_store_without_client(self):
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = None

        sources = service._get_files_from_vector_store("legal")

        assert sources == []

    def test_get_files_from_vector_store_with_error(self, mock_llama_stack_client):
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = mock_llama_stack_client
        service.vector_store_map = {"legal": ["legal-vs-id"]}

        mock_llama_stack_client.vector_stores.files.list.side_effect = Exception(
            "Error"
        )

        sources = service._get_files_from_vector_store("legal")

        assert sources == []


class TestLoadSourceUrlMapGitSuffix:
    """
    tests for _load_source_url_map handling of .git suffix in URLs.
    """

    def test_strips_git_suffix_from_url(self):
        """Test that .git suffix is properly stripped from GitHub URLs."""
        pipelines = [
            Pipeline(
                name="legal",
                enabled=True,
                version="v1.0",
                vector_store_name="legal-vector-db",
                source=SourceTypes.GITHUB,
                source_config=SourceConfig(
                    url="https://github.com/test/repo.git",
                    branch="main",
                    path="docs",
                    urls=None,
                ),
            )
        ]

        service = RAGService(pipelines=pipelines)

        assert "legal" in service.source_url_map
        # Should NOT contain .git
        assert ".git" not in service.source_url_map["legal"]
        assert service.source_url_map["legal"] == "https://github.com/test/repo/blob/main/docs"

    def test_handles_url_without_git_suffix(self):
        """Test that URLs without .git suffix work correctly."""
        pipelines = [
            Pipeline(
                name="legal",
                enabled=True,
                version="v1.0",
                vector_store_name="legal-vector-db",
                source=SourceTypes.GITHUB,
                source_config=SourceConfig(
                    url="https://github.com/test/repo",
                    branch="main",
                    path="docs",
                    urls=None,
                ),
            )
        ]

        service = RAGService(pipelines=pipelines)

        assert service.source_url_map["legal"] == "https://github.com/test/repo/blob/main/docs"

    def test_handles_url_with_trailing_slash(self):
        """Test that trailing slashes are handled correctly."""
        pipelines = [
            Pipeline(
                name="legal",
                enabled=True,
                version="v1.0",
                vector_store_name="legal-vector-db",
                source=SourceTypes.GITHUB,
                source_config=SourceConfig(
                    url="https://github.com/test/repo.git/",
                    branch="main",
                    path="docs",
                    urls=None,
                ),
            )
        ]

        service = RAGService(pipelines=pipelines)

        assert ".git" not in service.source_url_map["legal"]
        assert service.source_url_map["legal"] == "https://github.com/test/repo/blob/main/docs"


class TestValidateAndRegenerateMetadata:
    """
    tests for _validate_and_regenerate_metadata method.
    """

    def test_skips_validation_without_client(self):
        """Test that validation is skipped when client is not initialized."""
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = None
        service.file_metadata = {"old-file-id": {"original_filename": "test.pdf"}}

        service._validate_and_regenerate_metadata()

        # Metadata should remain unchanged
        assert "old-file-id" in service.file_metadata

    def test_skips_regeneration_when_metadata_matches(self, mock_llama_stack_client):
        """Test that regeneration is skipped when metadata matches vector store files."""
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = mock_llama_stack_client
        service.vector_store_map = {"legal": ["legal-vs-id"]}

        # Setup: file in vector store matches metadata
        mock_file = Mock()
        mock_file.id = "file-id-1"
        mock_llama_stack_client.vector_stores.files.list.return_value = [mock_file]

        service.file_metadata = {
            "file-id-1": {
                "original_filename": "test.pdf",
                "github_url": "https://github.com/test/repo/blob/main/test.pdf",
                "category": "legal",
            }
        }

        service._validate_and_regenerate_metadata()

        # Metadata should remain unchanged (not regenerated)
        assert service.file_metadata["file-id-1"]["original_filename"] == "test.pdf"

    def test_triggers_regeneration_when_metadata_mismatch(self, mock_llama_stack_client):
        """Test that regeneration is triggered when metadata doesn't match."""
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = mock_llama_stack_client
        service.vector_store_map = {"legal": ["legal-vs-id"]}
        service.source_url_map = {"legal": "https://github.com/test/repo/blob/main/docs"}

        # Setup: file in vector store does NOT match metadata
        mock_file = Mock()
        mock_file.id = "new-file-id"  # Different from metadata
        mock_llama_stack_client.vector_stores.files.list.return_value = [mock_file]

        # Old metadata with different file_id
        service.file_metadata = {
            "old-file-id": {
                "original_filename": "old.pdf",
                "github_url": "https://github.com/test/repo/blob/main/old.pdf",
                "category": "legal",
            }
        }

        # Mock files.retrieve to return filename
        mock_file_info = Mock()
        mock_file_info.filename = "NewDocument.txt"
        mock_llama_stack_client.files.retrieve.return_value = mock_file_info

        service._validate_and_regenerate_metadata()

        # Metadata should be regenerated with new file_id
        assert "new-file-id" in service.file_metadata
        assert "old-file-id" not in service.file_metadata

    def test_handles_empty_vector_stores(self, mock_llama_stack_client):
        """Test handling when vector stores have no files."""
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = mock_llama_stack_client
        service.vector_store_map = {"legal": ["legal-vs-id"]}

        mock_llama_stack_client.vector_stores.files.list.return_value = []

        service.file_metadata = {"old-file-id": {"original_filename": "test.pdf"}}

        service._validate_and_regenerate_metadata()

        # Metadata should remain unchanged when no files in vector stores
        assert "old-file-id" in service.file_metadata


class TestRegenerateFileMetadata:
    """
    tests for _regenerate_file_metadata method.
    """

    def test_regenerates_metadata_with_valid_filename(self, mock_llama_stack_client):
        """Test regeneration with a valid filename from Llama Stack."""
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = mock_llama_stack_client
        service.source_url_map = {"legal": "https://github.com/test/repo/blob/main/docs"}

        # Mock files.retrieve to return a .txt filename (as ingested)
        mock_file_info = Mock()
        mock_file_info.filename = "TestDocument.txt"
        mock_llama_stack_client.files.retrieve.return_value = mock_file_info

        file_id_to_category = {"file-123": "legal"}

        service._regenerate_file_metadata(file_id_to_category)

        assert "file-123" in service.file_metadata
        # Should convert .txt back to .pdf
        assert service.file_metadata["file-123"]["original_filename"] == "TestDocument.pdf"
        assert service.file_metadata["file-123"]["github_url"] == (
            "https://github.com/test/repo/blob/main/docs/TestDocument.pdf"
        )
        assert service.file_metadata["file-123"]["category"] == "legal"

    def test_regenerates_metadata_with_path_in_filename(self, mock_llama_stack_client):
        """Test that full paths in filename are cleaned up."""
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = mock_llama_stack_client
        service.source_url_map = {"hr": "https://github.com/test/repo/blob/main/hr"}

        mock_file_info = Mock()
        mock_file_info.filename = "/tmp/downloads/HRPolicy.txt"
        mock_llama_stack_client.files.retrieve.return_value = mock_file_info

        file_id_to_category = {"file-456": "hr"}

        service._regenerate_file_metadata(file_id_to_category)

        assert service.file_metadata["file-456"]["original_filename"] == "HRPolicy.pdf"

    def test_fallback_for_temp_filename(self, mock_llama_stack_client):
        """Test fallback when filename looks like a random temp file."""
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = mock_llama_stack_client
        service.source_url_map = {"legal": "https://github.com/test/repo/blob/main/docs"}

        mock_file_info = Mock()
        mock_file_info.filename = "tmp12345678.txt"  # Random temp filename
        mock_llama_stack_client.files.retrieve.return_value = mock_file_info

        file_id_to_category = {"file-789": "legal"}

        service._regenerate_file_metadata(file_id_to_category)

        # Should use fallback
        assert service.file_metadata["file-789"]["original_filename"] == "Document from legal"
        assert service.file_metadata["file-789"]["github_url"] == (
            "https://github.com/test/repo/blob/main/docs"
        )

    def test_fallback_when_no_filename(self, mock_llama_stack_client):
        """Test fallback when file info has no filename."""
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = mock_llama_stack_client
        service.source_url_map = {"sales": "https://github.com/test/repo/blob/main/sales"}

        mock_file_info = Mock(spec=[])  # No filename attribute
        mock_llama_stack_client.files.retrieve.return_value = mock_file_info

        file_id_to_category = {"file-abc": "sales"}

        service._regenerate_file_metadata(file_id_to_category)

        assert service.file_metadata["file-abc"]["original_filename"] == "Document from sales"

    def test_fallback_when_retrieve_fails(self, mock_llama_stack_client):
        """Test fallback when files.retrieve raises an exception."""
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = mock_llama_stack_client
        service.source_url_map = {"procurement": "https://github.com/test/repo/blob/main/procurement"}

        mock_llama_stack_client.files.retrieve.side_effect = Exception("File not found")

        file_id_to_category = {"file-xyz": "procurement"}

        service._regenerate_file_metadata(file_id_to_category)

        # Should still create metadata with fallback
        assert "file-xyz" in service.file_metadata
        assert service.file_metadata["file-xyz"]["original_filename"] == "Document from procurement"

    def test_skips_regeneration_without_client(self):
        """Test that regeneration is skipped when client is None."""
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.client = None
        service.file_metadata = {}

        file_id_to_category = {"file-123": "legal"}

        service._regenerate_file_metadata(file_id_to_category)

        # No metadata should be added
        assert service.file_metadata == {}


class TestExtractSourcesFallback:
    """
    tests for extract_sources_from_response fallback behavior.
    """

    def test_fallback_uses_category_url_not_file_id(self):
        """Test that unknown file_id falls back to category URL, not invalid file URL."""
        pipelines = []
        service = RAGService(pipelines=pipelines)
        service.source_url_map = {"legal": "https://github.com/test/repo/blob/main/docs"}
        service.file_metadata = {}  # Empty - no metadata

        mock_result = Mock()
        mock_result.file_id = "unknown-file-id-12345"
        mock_result.text = "Some text"

        mock_output_item = Mock()
        mock_output_item.type = "file_search_call"
        mock_output_item.results = [mock_result]

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        sources = service.extract_sources_from_response(mock_response, "legal")

        assert len(sources) == 1
        # Should use friendly filename, not raw file_id
        assert sources[0]["filename"] == "Document from legal knowledge base"
        # Should link to category folder, not invalid file URL
        assert sources[0]["url"] == "https://github.com/test/repo/blob/main/docs"
        # Should NOT contain the file_id in the URL
        assert "unknown-file-id" not in sources[0]["url"]
