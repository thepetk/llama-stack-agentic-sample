import json
import logging
import os
from typing import Any

from llama_stack_client import LlamaStackClient
from llama_stack_client.types import ResponseObject
from typing_extensions import Literal

from src.types import WorkflowState

log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

submission_states: "dict[str, WorkflowState]" = {}


def clean_text(text: "str") -> "str":
    """
    cleans text to handle encoding issues.
    """
    replacements = {
        "\u2013": "-",
        "\u2014": "--",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.encode("ascii", "ignore").decode("ascii")


def route_to_next_node(
    state: "WorkflowState",
) -> Literal[
    "legal_agent",
    "support_agent",
    "hr_agent",
    "sales_agent",
    "procurement_agent",
    "__end__",
]:
    if state["decision"] == "legal":
        return "legal_agent"
    elif state["decision"] == "techsupport":
        return "support_agent"
    elif state["decision"] == "hr":
        return "hr_agent"
    elif state["decision"] == "sales":
        return "sales_agent"
    elif state["decision"] == "procurement":
        return "procurement_agent"
    else:
        return "__end__"


def support_route_to_next_node(
    state: "WorkflowState",
) -> "Literal['pod_agent', 'perf_agent', 'git_agent', '__end__']":
    if state["decision"] == "pod":
        return "pod_agent"
    elif state["decision"] == "git":
        return "git_agent"
    elif state["decision"] == "perf":
        return "perf_agent"

    return "__end__"


def extract_rag_response_text(rag_response: "ResponseObject") -> "str":
    """
    extracts text content from RAG response output.
    """
    _res_text = ""
    for output_item in rag_response.output:
        if not hasattr(output_item, "type"):
            continue

        if output_item.type in ("text", "message"):
            if hasattr(output_item, "content") and isinstance(
                output_item.content, list
            ):
                for content in output_item.content:
                    if not hasattr(content, "text"):
                        continue

                    text_value = getattr(content, "text", "")
                    if text_value:
                        _res_text += str(text_value) + "\n"

            elif hasattr(output_item, "text"):
                text_value = getattr(output_item, "text", "")
                if text_value:
                    _res_text += str(text_value) + "\n"

        elif output_item.type == "file_search_call":
            queries = getattr(output_item, "queries", [])
            logger.debug(f"RAG file_search executed with queries: {queries}")

    return _res_text.strip()


def extract_mcp_output(
    response: "ResponseObject", agent_name: "str" = "agent", extract_url: "bool" = False
) -> "str":
    """
    extracts MCP call output from a response object.
    """
    mcp_output = ""

    for item in response.output:
        item_type = item.__class__.__name__

        if item_type not in ("McpCall", "ResponseOutputMessage"):
            logger.debug(f"{agent_name}: Unexpected output item type: {item_type}")
            continue

        if item_type == "McpCall":
            if extract_url:
                # case: git agent - extract URL from JSON output
                try:
                    output_value = getattr(item, "output", "")
                    if isinstance(output_value, str):
                        output_json = json.loads(output_value)
                        mcp_output = output_json.get("url", output_value)
                        logger.info(f"{agent_name}: GitHub issue created: {mcp_output}")
                    else:
                        mcp_output = str(output_value) if output_value else ""
                except (json.JSONDecodeError, AttributeError, TypeError) as e:
                    logger.warning(f"{agent_name}: Failed to parse MCP output: {e}")
                    mcp_output = str(getattr(item, "output", ""))
            else:
                # case: other agents - return raw output
                output_value = getattr(item, "output", "")
                mcp_output = str(output_value) if output_value else ""
                logger.info(f"{agent_name}: MCP call completed")
                logger.debug(f"{agent_name}: MCP output: {mcp_output}")

            break

        else:
            content_attr = getattr(item, "content", None)
            if (
                content_attr
                and isinstance(content_attr, list)
                and len(content_attr) > 0
            ):
                first_content = content_attr[0]
                text_value = getattr(first_content, "text", None)
                if text_value:
                    logger.debug(f"{agent_name} response message: {text_value}")

    return mcp_output


def check_llama_stack_availability(
    base_url: "str",
    required_models: "list[str] | None" = None,
) -> "dict[str, Any]":
    """
    checks llama-stack server connectivity and model availability.
    Makes sure that all required models are present on the server.
    """
    result: "dict[str, Any]" = {
        "connected": False,
        "error_message": "",
        "available_models": [],
        "missing_models": [],
    }

    try:
        client = LlamaStackClient(base_url=base_url)
        models_response = client.models.list()
        result["connected"] = True

        if not required_models:
            return result

        available_model_ids = set()
        if models_response:
            for model in models_response:
                model_id = getattr(model, "identifier", None) or getattr(
                    model, "id", None
                )
                if model_id:
                    available_model_ids.add(model_id)

        result["available_models"] = [
            m for m in required_models if m in available_model_ids
        ]
        result["missing_models"] = [
            m for m in required_models if m not in available_model_ids
        ]

    except Exception as e:
        error_str = str(e)
        if "Connection refused" in error_str or "ConnectError" in error_str:
            result["error_message"] = (
                f"Cannot connect to Llama Stack server at {base_url}. "
                "Please ensure the server is running."
            )
        elif "timeout" in error_str.lower():
            result["error_message"] = (
                f"Connection to Llama Stack at {base_url} timed out."
            )
        else:
            result["error_message"] = f"Llama Stack connection failed: {error_str}"

        logger.error(f"Llama Stack health check failed: {result['error_message']}")

    return result
