from unittest.mock import Mock

import pytest

from src.exceptions import AgentRunMethodParameterError
from src.methods import (
    classification_agent,
    git_agent,
    perf_agent,
    pod_agent,
    support_classification_agent,
)


class TestClassificationAgent:
    """
    tests for classification_agent function.
    """

    def test_classification_agent_success(
        self, sample_workflow_state, mock_openai_client
    ):
        result = classification_agent(
            sample_workflow_state,
            mock_openai_client,
            "test-model",
            "guardrail-model",
        )

        assert "agent_timings" in result
        assert "Classification" in result["agent_timings"]
        assert result["active_agent"] == "Classification"
        assert result["decision"] == "legal"

    def test_classification_agent_with_unsafe_content(
        self, sample_workflow_state, mock_openai_client
    ):
        moderation_result = Mock()
        moderation_result.flagged = True
        moderation_result.categories = Mock(violence=Mock(name="violence"))
        moderation_result.categories.model_extra = {"violence": True}
        mock_moderation_response = Mock()
        mock_moderation_response.results = [moderation_result]
        mock_openai_client.moderations.create.return_value = mock_moderation_response

        result = classification_agent(
            sample_workflow_state,
            mock_openai_client,
            "test-model",
            "guardrail-model",
        )

        assert result["decision"] == "unsafe"
        assert result["workflow_complete"] is True

    def test_classification_agent_without_openai_client(self, sample_workflow_state):
        with pytest.raises(AgentRunMethodParameterError):
            classification_agent(
                sample_workflow_state, None, "test-model", "guardrail-model"
            )

    def test_classification_agent_without_topic_llm(
        self, sample_workflow_state, mock_openai_client
    ):
        with pytest.raises(AgentRunMethodParameterError):
            classification_agent(
                sample_workflow_state, mock_openai_client, None, "guardrail-model"
            )

    def test_classification_agent_without_guardrail_model(
        self, sample_workflow_state, mock_openai_client
    ):
        with pytest.raises(AgentRunMethodParameterError):
            classification_agent(
                sample_workflow_state, mock_openai_client, "test-model", None
            )

    def test_classification_agent_with_exception(
        self, sample_workflow_state, mock_openai_client
    ):
        mock_openai_client.beta.chat.completions.parse.side_effect = Exception(
            "Parse error"
        )

        result = classification_agent(
            sample_workflow_state,
            mock_openai_client,
            "test-model",
            "guardrail-model",
        )

        assert result["decision"] == "unknown"
        assert result["workflow_complete"] is True

    def test_classification_agent_timing_tracking(
        self, sample_workflow_state, mock_openai_client
    ):
        result = classification_agent(
            sample_workflow_state,
            mock_openai_client,
            "test-model",
            "guardrail-model",
        )

        assert "agent_timings" in result
        assert "Classification" in result["agent_timings"]
        assert isinstance(result["agent_timings"]["Classification"], float)
        assert result["agent_timings"]["Classification"] >= 0


class TestSupportClassificationAgent:
    """
    tests for support_classification_agent function.
    """

    def test_support_classification_agent_success(
        self, sample_workflow_state, mock_openai_client
    ):
        # Create a simple object instead of Mock to avoid Mock attribute issues
        class ParsedResult:
            def __init__(self):
                self.classification = "git"
                self.namespace = "test-namespace"
                self.performance = "true"

        mock_parsed = ParsedResult()

        mock_message = Mock()
        mock_message.parsed = mock_parsed

        mock_parsed_message = Mock()
        mock_parsed_message.message = mock_message

        mock_parsed_completion = Mock()
        mock_parsed_completion.choices = [mock_parsed_message]

        mock_openai_client.beta.chat.completions.parse.return_value = (
            mock_parsed_completion
        )

        result = support_classification_agent(
            sample_workflow_state, mock_openai_client, "test-model"
        )

        assert result["decision"] == "git"
        assert result["namespace"] == "test-namespace"
        assert "agent_timings" in result
        assert "Support Classification" in result["agent_timings"]

    def test_support_classification_agent_with_performance_issue(
        self, sample_workflow_state, mock_openai_client
    ):
        # Create a simple object instead of Mock to avoid Mock attribute issues
        class ParsedResult:
            def __init__(self):
                self.classification = "pod"
                self.namespace = "test-namespace"
                self.performance = "performance issue"

        mock_parsed = ParsedResult()

        mock_message = Mock()
        mock_message.parsed = mock_parsed

        mock_parsed_message = Mock()
        mock_parsed_message.message = mock_message

        mock_parsed_completion = Mock()
        mock_parsed_completion.choices = [mock_parsed_message]

        mock_openai_client.beta.chat.completions.parse.return_value = (
            mock_parsed_completion
        )

        result = support_classification_agent(
            sample_workflow_state, mock_openai_client, "test-model"
        )

        assert result["decision"] == "perf"

    def test_support_classification_agent_without_openai_client(
        self, sample_workflow_state
    ):
        with pytest.raises(AgentRunMethodParameterError):
            support_classification_agent(sample_workflow_state, None, "test-model")

    def test_support_classification_agent_without_topic_llm(
        self, sample_workflow_state, mock_openai_client
    ):
        with pytest.raises(AgentRunMethodParameterError):
            support_classification_agent(
                sample_workflow_state, mock_openai_client, None
            )

    def test_support_classification_agent_with_exception(
        self, sample_workflow_state, mock_openai_client
    ):
        mock_openai_client.beta.chat.completions.parse.side_effect = Exception(
            "Parse error"
        )

        result = support_classification_agent(
            sample_workflow_state, mock_openai_client, "test-model"
        )

        assert result["decision"] == "unknown"
        assert result["namespace"] == ""

    def test_support_classification_agent_timing_tracking(
        self, sample_workflow_state, mock_openai_client
    ):
        result = support_classification_agent(
            sample_workflow_state, mock_openai_client, "test-model"
        )

        assert "agent_timings" in result
        assert "Support Classification" in result["agent_timings"]
        assert isinstance(result["agent_timings"]["Support Classification"], float)


class TestGitAgent:
    """
    tests for git_agent function.
    """

    def test_git_agent_success(self, sample_workflow_state, mock_openai_client):
        mock_output_item = Mock()
        mock_output_item.__class__.__name__ = "McpCall"
        mock_output_item.output = '{"url": "https://github.com/test/issue/1"}'

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        mock_openai_client.responses.create.return_value = mock_response

        result = git_agent(
            sample_workflow_state,
            mock_openai_client,
            "test-token",
            "test-model",
            "https://github.com/test/repo",
        )

        assert "agent_timings" in result
        assert "Git" in result["agent_timings"]
        assert result["active_agent"] == "Git"
        assert result["github_issue"] == "https://github.com/test/issue/1"
        assert result["status_message"] == "✅ Git Agent finished: GitHub issue created"
        assert "✅ Git Agent finished: GitHub issue created" in result["status_history"]

    def test_git_agent_without_openai_client(self, sample_workflow_state):
        with pytest.raises(AgentRunMethodParameterError):
            git_agent(
                sample_workflow_state,
                None,
                "test-token",
                "test-model",
                "https://github.com/test/repo",
            )

    def test_git_agent_without_git_token(
        self, sample_workflow_state, mock_openai_client
    ):
        with pytest.raises(AgentRunMethodParameterError):
            git_agent(
                sample_workflow_state,
                mock_openai_client,
                None,
                "test-model",
                "https://github.com/test/repo",
            )

    def test_git_agent_without_tools_llm(
        self, sample_workflow_state, mock_openai_client
    ):
        with pytest.raises(AgentRunMethodParameterError):
            git_agent(
                sample_workflow_state,
                mock_openai_client,
                "test-token",
                None,
                "https://github.com/test/repo",
            )

    def test_git_agent_with_exception(self, sample_workflow_state, mock_openai_client):
        mock_openai_client.responses.create.side_effect = Exception("MCP error")

        result = git_agent(
            sample_workflow_state,
            mock_openai_client,
            "test-token",
            "test-model",
            "https://github.com/test/repo",
        )

        assert "agent_timings" in result
        assert "Git" in result["agent_timings"]
        assert result["github_issue"] == ""
        assert "GitHub MCP Server Unavailable" in result["status_message"]
        assert (
            "Git Agent finished: GitHub MCP Server Unavailable"
            in result["status_history"]
        )

    def test_git_agent_timing_tracking(self, sample_workflow_state, mock_openai_client):
        mock_output_item = Mock()
        mock_output_item.__class__.__name__ = "McpCall"
        mock_output_item.output = '{"url": "https://github.com/test/issue/1"}'

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        mock_openai_client.responses.create.return_value = mock_response

        result = git_agent(
            sample_workflow_state,
            mock_openai_client,
            "test-token",
            "test-model",
            "https://github.com/test/repo",
        )

        assert isinstance(result["agent_timings"]["Git"], float)
        assert result["agent_timings"]["Git"] >= 0


class TestPodAgent:
    """
    tests for pod_agent function.
    """

    def test_pod_agent_success(self, sample_workflow_state, mock_openai_client):
        mock_output_item = Mock()
        mock_output_item.__class__.__name__ = "McpCall"
        mock_output_item.output = "Pod status information"

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        mock_openai_client.responses.create.return_value = mock_response

        result = pod_agent(sample_workflow_state, mock_openai_client, "test-model")

        assert "agent_timings" in result
        assert "Pod" in result["agent_timings"]
        assert result["active_agent"] == "Pod"
        assert result["mcp_output"] == "Pod status information"

    def test_pod_agent_without_openai_client(self, sample_workflow_state):
        with pytest.raises(AgentRunMethodParameterError):
            pod_agent(sample_workflow_state, None, "test-model")

    def test_pod_agent_without_tools_llm(
        self, sample_workflow_state, mock_openai_client
    ):
        with pytest.raises(AgentRunMethodParameterError):
            pod_agent(sample_workflow_state, mock_openai_client, None)

    def test_pod_agent_with_exception(self, sample_workflow_state, mock_openai_client):
        mock_openai_client.responses.create.side_effect = Exception("MCP error")

        result = pod_agent(sample_workflow_state, mock_openai_client, "test-model")

        assert "agent_timings" in result
        assert "Pod" in result["agent_timings"]

    def test_pod_agent_timing_tracking(self, sample_workflow_state, mock_openai_client):
        mock_output_item = Mock()
        mock_output_item.__class__.__name__ = "McpCall"
        mock_output_item.output = "Pod status"

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        mock_openai_client.responses.create.return_value = mock_response

        result = pod_agent(sample_workflow_state, mock_openai_client, "test-model")

        assert isinstance(result["agent_timings"]["Pod"], float)
        assert result["agent_timings"]["Pod"] >= 0


class TestPerfAgent:
    """
    tests for perf_agent function.
    """

    def test_perf_agent_success(self, sample_workflow_state, mock_openai_client):
        mock_output_item = Mock()
        mock_output_item.__class__.__name__ = "McpCall"
        mock_output_item.output = "Performance metrics"

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        mock_openai_client.responses.create.return_value = mock_response

        result = perf_agent(sample_workflow_state, mock_openai_client, "test-model")

        assert "agent_timings" in result
        assert "Performance" in result["agent_timings"]
        assert result["active_agent"] == "Performance"
        assert result["mcp_output"] == "Performance metrics"

    def test_perf_agent_without_openai_client(self, sample_workflow_state):
        with pytest.raises(AgentRunMethodParameterError):
            perf_agent(sample_workflow_state, None, "test-model")

    def test_perf_agent_without_tools_llm(
        self, sample_workflow_state, mock_openai_client
    ):
        with pytest.raises(AgentRunMethodParameterError):
            perf_agent(sample_workflow_state, mock_openai_client, None)

    def test_perf_agent_with_exception(self, sample_workflow_state, mock_openai_client):
        mock_openai_client.responses.create.side_effect = Exception("MCP error")

        result = perf_agent(sample_workflow_state, mock_openai_client, "test-model")

        assert "agent_timings" in result
        assert "Performance" in result["agent_timings"]

    def test_perf_agent_timing_tracking(
        self, sample_workflow_state, mock_openai_client
    ):
        mock_output_item = Mock()
        mock_output_item.__class__.__name__ = "McpCall"
        mock_output_item.output = "Performance data"

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        mock_openai_client.responses.create.return_value = mock_response

        result = perf_agent(sample_workflow_state, mock_openai_client, "test-model")

        assert isinstance(result["agent_timings"]["Performance"], float)
        assert result["agent_timings"]["Performance"] >= 0


class TestAgentTimingConsistency:
    """
    tests to ensure all agents track timing consistently.
    """

    def test_all_agents_initialize_timing_dict(
        self, sample_workflow_state, mock_openai_client
    ):
        result = classification_agent(
            sample_workflow_state,
            mock_openai_client,
            "test-model",
            "guardrail-model",
        )
        assert "agent_timings" in result
        assert isinstance(result["agent_timings"], dict)

        sample_workflow_state["agent_timings"] = None

        result = support_classification_agent(
            sample_workflow_state, mock_openai_client, "test-model"
        )
        assert "agent_timings" in result
        assert isinstance(result["agent_timings"], dict)

    def test_all_agents_set_active_agent(
        self, sample_workflow_state, mock_openai_client
    ):
        result = classification_agent(
            sample_workflow_state,
            mock_openai_client,
            "test-model",
            "guardrail-model",
        )
        assert result["active_agent"] == "Classification"

        result = support_classification_agent(
            sample_workflow_state, mock_openai_client, "test-model"
        )
        assert result["active_agent"] == "Support Classification"

        mock_output_item = Mock()
        mock_output_item.__class__.__name__ = "McpCall"
        mock_output_item.output = '{"url": "https://github.com/test/issue/1"}'
        mock_response = Mock()
        mock_response.output = [mock_output_item]
        mock_openai_client.responses.create.return_value = mock_response

        result = git_agent(
            sample_workflow_state,
            mock_openai_client,
            "test-token",
            "test-model",
            "https://github.com/test/repo",
        )
        assert result["active_agent"] == "Git"

        result = pod_agent(sample_workflow_state, mock_openai_client, "test-model")
        assert result["active_agent"] == "Pod"

        result = perf_agent(sample_workflow_state, mock_openai_client, "test-model")
        assert result["active_agent"] == "Performance"
