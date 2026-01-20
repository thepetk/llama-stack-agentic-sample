import json
from unittest.mock import Mock, patch

from src.utils import (
    check_llama_stack_availability,
    clean_text,
    extract_mcp_output,
    extract_rag_response_text,
    route_to_next_node,
    support_route_to_next_node,
)


class TestCleanText:
    """
    tests for clean_text function.
    """

    def test_clean_text_with_unicode_dashes(self):
        text = "test\u2013dash\u2014test"
        result = clean_text(text)
        assert result == "test-dash--test"

    def test_clean_text_with_unicode_quotes(self):
        text = "\u2018single\u2019 \u201cdouble\u201d"
        result = clean_text(text)
        assert result == "'single' \"double\""

    def test_clean_text_with_ellipsis(self):
        text = "test\u2026more"
        result = clean_text(text)
        assert result == "test...more"

    def test_clean_text_with_non_ascii_characters(self):
        text = "test café résumé"
        result = clean_text(text)
        # Non-ASCII characters should be removed
        assert "café" not in result
        assert "test" in result

    def test_clean_text_with_empty_string(self):
        result = clean_text("")
        assert result == ""

    def test_clean_text_with_regular_text(self):
        text = "Regular text with no special characters"
        result = clean_text(text)
        assert result == text

    def test_clean_text_with_all_replacements(self):
        text = "\u2013\u2014\u2018\u2019\u201c\u201d\u2026"
        result = clean_text(text)
        assert result == "---''\"\"..."


class TestRouteToNextNode:
    """
    tests for route_to_next_node function.
    """

    def test_route_to_legal_agent(self, sample_workflow_state):
        sample_workflow_state["decision"] = "legal"
        result = route_to_next_node(sample_workflow_state)
        assert result == "legal_agent"

    def test_route_to_techsupport_agent(self, sample_workflow_state):
        sample_workflow_state["decision"] = "techsupport"
        result = route_to_next_node(sample_workflow_state)
        assert result == "support_agent"

    def test_route_to_hr_agent(self, sample_workflow_state):
        sample_workflow_state["decision"] = "hr"
        result = route_to_next_node(sample_workflow_state)
        assert result == "hr_agent"

    def test_route_to_sales_agent(self, sample_workflow_state):
        sample_workflow_state["decision"] = "sales"
        result = route_to_next_node(sample_workflow_state)
        assert result == "sales_agent"

    def test_route_to_procurement_agent(self, sample_workflow_state):
        sample_workflow_state["decision"] = "procurement"
        result = route_to_next_node(sample_workflow_state)
        assert result == "procurement_agent"

    def test_route_to_end_with_unknown_decision(self, sample_workflow_state):
        sample_workflow_state["decision"] = "unknown"
        result = route_to_next_node(sample_workflow_state)
        assert result == "__end__"

    def test_route_to_end_with_unsafe_decision(self, sample_workflow_state):
        sample_workflow_state["decision"] = "unsafe"
        result = route_to_next_node(sample_workflow_state)
        assert result == "__end__"

    def test_route_to_end_with_empty_decision(self, sample_workflow_state):
        sample_workflow_state["decision"] = ""
        result = route_to_next_node(sample_workflow_state)
        assert result == "__end__"


class TestSupportRouteToNextNode:
    """
    tests for support_route_to_next_node function.
    """

    def test_route_to_pod_agent(self, sample_workflow_state):
        sample_workflow_state["decision"] = "pod"
        result = support_route_to_next_node(sample_workflow_state)
        assert result == "pod_agent"

    def test_route_to_git_agent(self, sample_workflow_state):
        sample_workflow_state["decision"] = "git"
        result = support_route_to_next_node(sample_workflow_state)
        assert result == "git_agent"

    def test_route_to_perf_agent(self, sample_workflow_state):
        sample_workflow_state["decision"] = "perf"
        result = support_route_to_next_node(sample_workflow_state)
        assert result == "perf_agent"

    def test_route_to_end_with_unknown_decision(self, sample_workflow_state):
        sample_workflow_state["decision"] = "unknown"
        result = support_route_to_next_node(sample_workflow_state)
        assert result == "__end__"

    def test_route_to_end_with_empty_decision(self, sample_workflow_state):
        sample_workflow_state["decision"] = ""
        result = support_route_to_next_node(sample_workflow_state)
        assert result == "__end__"


class TestExtractRagResponseText:
    """
    tests for extract_rag_response_text function.
    """

    def test_extract_text_from_text_type_output(self):
        mock_content = Mock()
        mock_content.text = "Test response text"

        mock_output_item = Mock()
        mock_output_item.type = "text"
        mock_output_item.content = [mock_content]

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        result = extract_rag_response_text(mock_response)
        assert result == "Test response text"

    def test_extract_text_from_message_type_output(self):
        mock_content = Mock()
        mock_content.text = "Message response"

        mock_output_item = Mock()
        mock_output_item.type = "message"
        mock_output_item.content = [mock_content]

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        result = extract_rag_response_text(mock_response)
        assert result == "Message response"

    def test_extract_text_with_direct_text_attribute(self):
        mock_output_item = Mock()
        mock_output_item.type = "text"
        mock_output_item.text = "Direct text"
        del mock_output_item.content  # Remove content attribute

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        result = extract_rag_response_text(mock_response)
        assert result == "Direct text"

    def test_extract_text_with_file_search_call(self):
        mock_output_item = Mock()
        mock_output_item.type = "file_search_call"
        mock_output_item.queries = ["test query"]

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        result = extract_rag_response_text(mock_response)
        assert result == ""

    def test_extract_text_with_multiple_outputs(self):
        mock_content1 = Mock()
        mock_content1.text = "First response"

        mock_content2 = Mock()
        mock_content2.text = "Second response"

        mock_output_item1 = Mock()
        mock_output_item1.type = "text"
        mock_output_item1.content = [mock_content1]

        mock_output_item2 = Mock()
        mock_output_item2.type = "message"
        mock_output_item2.content = [mock_content2]

        mock_response = Mock()
        mock_response.output = [mock_output_item1, mock_output_item2]

        result = extract_rag_response_text(mock_response)
        assert "First response" in result
        assert "Second response" in result

    def test_extract_text_with_no_type_attribute(self):
        mock_output_item = Mock(spec=[])

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        result = extract_rag_response_text(mock_response)
        assert result == ""

    def test_extract_text_with_empty_output(self):
        mock_response = Mock()
        mock_response.output = []

        result = extract_rag_response_text(mock_response)
        assert result == ""

    def test_extract_text_with_content_without_text(self):
        mock_content = Mock(spec=[])

        mock_output_item = Mock()
        mock_output_item.type = "text"
        mock_output_item.content = [mock_content]

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        result = extract_rag_response_text(mock_response)
        assert result == ""


class TestExtractMcpOutput:
    """
    tests for extract_mcp_output function.
    """

    def test_extract_mcp_output_from_mcp_call(self):
        mock_item = Mock()
        mock_item.__class__.__name__ = "McpCall"
        mock_item.output = "Test MCP output"

        mock_response = Mock()
        mock_response.output = [mock_item]

        result = extract_mcp_output(mock_response, agent_name="test_agent")
        assert result == "Test MCP output"

    def test_extract_url_from_mcp_call(self):
        output_json = json.dumps({"url": "https://github.com/test/issue/1"})

        mock_item = Mock()
        mock_item.__class__.__name__ = "McpCall"
        mock_item.output = output_json

        mock_response = Mock()
        mock_response.output = [mock_item]

        result = extract_mcp_output(
            mock_response, agent_name="git_agent", extract_url=True
        )
        assert result == "https://github.com/test/issue/1"

    def test_extract_url_with_invalid_json(self):
        mock_item = Mock()
        mock_item.__class__.__name__ = "McpCall"
        mock_item.output = "invalid json"

        mock_response = Mock()
        mock_response.output = [mock_item]

        result = extract_mcp_output(
            mock_response, agent_name="git_agent", extract_url=True
        )
        assert result == "invalid json"

    def test_extract_url_with_missing_url_field(self):
        output_json = json.dumps({"status": "success"})

        mock_item = Mock()
        mock_item.__class__.__name__ = "McpCall"
        mock_item.output = output_json

        mock_response = Mock()
        mock_response.output = [mock_item]

        result = extract_mcp_output(
            mock_response, agent_name="git_agent", extract_url=True
        )
        assert result == output_json

    def test_extract_mcp_output_from_response_message(self):
        mock_content = Mock()
        mock_content.text = "Response message text"

        mock_item = Mock()
        mock_item.__class__.__name__ = "ResponseOutputMessage"
        mock_item.content = [mock_content]

        mock_response = Mock()
        mock_response.output = [mock_item]

        result = extract_mcp_output(mock_response, agent_name="test_agent")
        assert result == ""

    def test_extract_mcp_output_with_unexpected_type(self):
        mock_item = Mock()
        mock_item.__class__.__name__ = "UnknownType"

        mock_response = Mock()
        mock_response.output = [mock_item]

        result = extract_mcp_output(mock_response, agent_name="test_agent")
        assert result == ""

    def test_extract_mcp_output_with_empty_output(self):
        mock_response = Mock()
        mock_response.output = []

        result = extract_mcp_output(mock_response, agent_name="test_agent")
        assert result == ""

    def test_extract_mcp_output_with_multiple_items(self):
        mock_item1 = Mock()
        mock_item1.__class__.__name__ = "McpCall"
        mock_item1.output = "First output"

        mock_item2 = Mock()
        mock_item2.__class__.__name__ = "McpCall"
        mock_item2.output = "Second output"

        mock_response = Mock()
        mock_response.output = [mock_item1, mock_item2]

        result = extract_mcp_output(mock_response, agent_name="test_agent")
        assert result == "First output"

    def test_extract_mcp_output_without_output_attribute(self):
        mock_item = Mock(spec=["__class__"])
        mock_item.__class__.__name__ = "McpCall"
        del mock_item.output

        mock_response = Mock()
        mock_response.output = [mock_item]

        result = extract_mcp_output(
            mock_response, agent_name="test_agent", extract_url=True
        )
        assert result == ""


class TestCheckLlamaStackAvailability:
    """
    tests for check_llama_stack_availability function.
    """

    def test_successful_connection_without_model_check(self):
        mock_client = Mock()
        mock_client.models.list.return_value = []

        with patch("src.utils.LlamaStackClient", return_value=mock_client):
            result = check_llama_stack_availability("http://localhost:8321")

        assert result["connected"] is True
        assert result["error_message"] == ""
        assert result["available_models"] == []
        assert result["missing_models"] == []

    def test_successful_connection_with_all_models_available(self):
        mock_model1 = Mock()
        mock_model1.identifier = "model-1"
        mock_model2 = Mock()
        mock_model2.identifier = "model-2"

        mock_client = Mock()
        mock_client.models.list.return_value = [mock_model1, mock_model2]

        with patch("src.utils.LlamaStackClient", return_value=mock_client):
            result = check_llama_stack_availability(
                "http://localhost:8321",
                required_models=["model-1", "model-2"],
            )

        assert result["connected"] is True
        assert result["error_message"] == ""
        assert result["available_models"] == ["model-1", "model-2"]
        assert result["missing_models"] == []

    def test_successful_connection_with_some_models_missing(self):
        mock_model = Mock()
        mock_model.identifier = "model-1"

        mock_client = Mock()
        mock_client.models.list.return_value = [mock_model]

        with patch("src.utils.LlamaStackClient", return_value=mock_client):
            result = check_llama_stack_availability(
                "http://localhost:8321",
                required_models=["model-1", "model-2", "model-3"],
            )

        assert result["connected"] is True
        assert result["error_message"] == ""
        assert result["available_models"] == ["model-1"]
        assert result["missing_models"] == ["model-2", "model-3"]

    def test_successful_connection_with_all_models_missing(self):
        mock_client = Mock()
        mock_client.models.list.return_value = []

        with patch("src.utils.LlamaStackClient", return_value=mock_client):
            result = check_llama_stack_availability(
                "http://localhost:8321",
                required_models=["model-1", "model-2"],
            )

        assert result["connected"] is True
        assert result["available_models"] == []
        assert result["missing_models"] == ["model-1", "model-2"]

    def test_connection_refused_error(self):
        with patch(
            "src.utils.LlamaStackClient",
            side_effect=Exception("Connection refused"),
        ):
            result = check_llama_stack_availability("http://localhost:8321")

        assert result["connected"] is False
        assert "Cannot connect to Llama Stack server" in result["error_message"]
        assert "http://localhost:8321" in result["error_message"]
        assert result["available_models"] == []
        assert result["missing_models"] == []

    def test_connect_error(self):
        with patch(
            "src.utils.LlamaStackClient",
            side_effect=Exception("ConnectError: Failed to establish connection"),
        ):
            result = check_llama_stack_availability("http://localhost:8321")

        assert result["connected"] is False
        assert "Cannot connect to Llama Stack server" in result["error_message"]

    def test_timeout_error(self):
        with patch(
            "src.utils.LlamaStackClient",
            side_effect=Exception("Request timeout exceeded"),
        ):
            result = check_llama_stack_availability("http://localhost:8321")

        assert result["connected"] is False
        assert "timed out" in result["error_message"]

    def test_generic_error(self):
        with patch(
            "src.utils.LlamaStackClient",
            side_effect=Exception("Some unexpected error"),
        ):
            result = check_llama_stack_availability("http://localhost:8321")

        assert result["connected"] is False
        assert "Llama Stack connection failed" in result["error_message"]
        assert "Some unexpected error" in result["error_message"]

    def test_model_with_id_attribute_instead_of_identifier(self):
        mock_model = Mock(spec=["id"])
        mock_model.id = "model-with-id"

        mock_client = Mock()
        mock_client.models.list.return_value = [mock_model]

        with patch("src.utils.LlamaStackClient", return_value=mock_client):
            result = check_llama_stack_availability(
                "http://localhost:8321",
                required_models=["model-with-id"],
            )

        assert result["connected"] is True
        assert result["available_models"] == ["model-with-id"]
        assert result["missing_models"] == []

    def test_empty_required_models_list(self):
        mock_client = Mock()
        mock_client.models.list.return_value = []

        with patch("src.utils.LlamaStackClient", return_value=mock_client):
            result = check_llama_stack_availability(
                "http://localhost:8321",
                required_models=[],
            )

        assert result["connected"] is True
        assert result["available_models"] == []
        assert result["missing_models"] == []

    def test_none_models_response(self):
        mock_client = Mock()
        mock_client.models.list.return_value = None

        with patch("src.utils.LlamaStackClient", return_value=mock_client):
            result = check_llama_stack_availability(
                "http://localhost:8321",
                required_models=["model-1"],
            )

        assert result["connected"] is True
        assert result["available_models"] == []
        assert result["missing_models"] == ["model-1"]
