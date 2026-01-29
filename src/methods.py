import os
import time
from typing import Any, cast

from llama_stack_client.types import ResponseObject
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

from src.constants import DEFAULT_MCP_SERVER_URL
from src.exceptions import AgentRunMethodParameterError
from src.models import ClassificationModel, SupportClassificationModel
from src.types import WorkflowAgentPrompts, WorkflowState
from src.utils import extract_mcp_output, logger, submission_states

# MCP Server URL - read from environment or use default
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL)


def classification_agent(
    state: "WorkflowState",
    openai_client: "OpenAI | None",
    topic_llm: "str | None",
    guardrail_model: "str | None",
) -> "WorkflowState":
    agent_start_time = time.time()
    if "agent_timings" not in state or state["agent_timings"] is None:
        state["agent_timings"] = {}
    if "status_history" not in state or state["status_history"] is None:
        state["status_history"] = []

    state["active_agent"] = "Classification"
    status_msg = "🔍 Classification Agent processes the request..."
    state["status_message"] = status_msg
    state["status_history"].append(status_msg)

    submission_id = state.get("submission_id")
    if submission_id:
        submission_states[submission_id] = cast("WorkflowState", dict(state))
        logger.info(f"Updated status: {status_msg} for submission {submission_id}")

    # check if necessary variables exist
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in classification agent"
        )

    if topic_llm is None:
        raise AgentRunMethodParameterError(
            "topic_llm is required in classification agent"
        )

    if guardrail_model is None:
        raise AgentRunMethodParameterError(
            "guardrail_model is required in classification agent"
        )

    # checking if the input is safe
    safety_response = openai_client.moderations.create(
        model=guardrail_model, input=str(state["input"])
    )

    for moderation in safety_response.results:
        if not moderation.flagged:
            continue

        logger.info(
            f"Classification result: '{state['input']}' is flagged as '{moderation}'"
        )
        state["decision"] = "unsafe"
        state["data"] = state["input"]
        model_extra = moderation.categories.model_extra
        flagged_categories = [
            key
            for key, value in (model_extra.items() if model_extra else [])
            if value is True
        ]
        categories_str = ", ".join(flagged_categories)
        state["classification_message"] = (
            f"Classification result: '{state['input']}' "
            f"is flagged for: {categories_str}"
        )
        state["workflow_complete"] = True
        return state

    # Use OpenAI client for structured output with Pydantic models
    try:
        classification_prompt = WorkflowAgentPrompts.CLASSIFICATION_PROMPT.format(
            state_input=state["input"]
        )
        logger.debug(f"Classification prompt: {classification_prompt}")

        messages: "list[ChatCompletionUserMessageParam]" = [
            cast(
                ChatCompletionUserMessageParam,
                {
                    "role": "user",
                    "content": classification_prompt,
                },
            )
        ]
        completion = openai_client.beta.chat.completions.parse(
            model=topic_llm,
            messages=messages,
            response_format=ClassificationModel,
        )

        classification_result = completion.choices[0].message.parsed
        logger.debug(f"Raw classification result object: {classification_result}")

        # Validate classification
        if not classification_result or not hasattr(
            classification_result, "classification"
        ):
            logger.error("Failed to get structured response from the model.")
            state["decision"] = "unknown"
            state["data"] = state["input"]
            state["classification_message"] = "Unable to determine request type."
            state["workflow_complete"] = True
            return state

        if classification_result.classification not in (
            "legal",
            "techsupport",
            "support",
            "hr",
            "sales",
            "procurement",
            "unsafe",
            "unknown",
        ):
            logger.error(
                f"Invalid classification: {classification_result.classification}"
            )
            state["decision"] = "unknown"
            state["data"] = state["input"]
            state["classification_message"] = "Unable to determine request type."
            state["workflow_complete"] = True
            return state

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        state["decision"] = "unknown"
        state["data"] = state["input"]
        state["classification_message"] = f"Classification error: {str(e)[:100]}"
        state["workflow_complete"] = True
        return state
    logger.info(
        f"Classification result: {classification_result} for input '{state['input']}'"
    )

    state["decision"] = classification_result.classification
    state["data"] = state["input"]

    agent_end_time = time.time()
    state["agent_timings"]["Classification"] = agent_end_time - agent_start_time

    completion_msg = (
        "✅ Classification Agent finished:"
        f" routed to {classification_result.classification}"
    )
    state["status_history"].append(completion_msg)
    state["status_message"] = completion_msg

    submission_id = state.get("submission_id")
    if submission_id:
        submission_states[submission_id] = cast("WorkflowState", dict(state))
        logger.info(f"Updated status: {completion_msg} for submission {submission_id}")

    return state


def support_classification_agent(
    state: "WorkflowState",
    openai_client: "OpenAI | None",
    topic_llm: "str | None",
) -> "WorkflowState":
    agent_start_time = time.time()
    if "agent_timings" not in state or state["agent_timings"] is None:
        state["agent_timings"] = {}
    if "status_history" not in state or state["status_history"] is None:
        state["status_history"] = []

    state["active_agent"] = "Support Classification"
    status_msg = "🔍 Support Classification Agent processes the request..."
    state["status_message"] = status_msg
    state["status_history"].append(status_msg)

    submission_id = state.get("submission_id")
    if submission_id:
        submission_states[submission_id] = cast("WorkflowState", dict(state))
        logger.info(f"Updated status: {status_msg} for submission {submission_id}")

    # check if necessary variables exist
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in support classification agent"
        )

    if topic_llm is None:
        raise AgentRunMethodParameterError(
            "model is required in support classification agent"
        )

    try:
        messages: "list[ChatCompletionUserMessageParam]" = [
            cast(
                ChatCompletionUserMessageParam,
                {
                    "role": "user",
                    "content": WorkflowAgentPrompts.SUPPORT_CLASSIFICATION_PROMPT.format(  # noqa
                        state_input=state["input"]
                    ),
                },
            )
        ]
        completion = openai_client.beta.chat.completions.parse(
            model=topic_llm,
            messages=messages,
            response_format=SupportClassificationModel,
        )

        classification_result = completion.choices[0].message.parsed

        if not classification_result:
            logger.error(
                "Failed to get structured response from support classification."
            )
            state["decision"] = "unknown"
            state["namespace"] = ""
            return state

        logger.info(
            f"Support Classification result: {classification_result} "
            f"for input '{state['input']}'"
        )

    except Exception as e:
        logger.error(f"Support classification failed: {e}")
        state["decision"] = "unknown"
        state["namespace"] = ""
        return state
    state["namespace"] = classification_result.namespace
    if classification_result.performance == "performance issue":
        state["decision"] = "perf"
    else:
        state["decision"] = classification_result.classification
    state["data"] = state["input"]

    agent_end_time = time.time()
    state["agent_timings"]["Support Classification"] = agent_end_time - agent_start_time

    completion_msg = (
        f"✅ Support Classification Agent finished: routed to {state['decision']}"
    )
    state["status_history"].append(completion_msg)
    state["status_message"] = completion_msg

    submission_id = state.get("submission_id")
    if submission_id:
        submission_states[submission_id] = cast("WorkflowState", dict(state))
        logger.info(f"Updated status: {completion_msg} for submission {submission_id}")

    return state


def git_agent(
    state: "WorkflowState",
    openai_client: "OpenAI | None",
    git_token: "str | None",
    tools_llm: "str | None",
    github_url: "str | None",
) -> "WorkflowState":
    logger.debug(f"git Agent request for submission: {state['submission_id']}")

    agent_start_time = time.time()
    if "agent_timings" not in state or state["agent_timings"] is None:
        state["agent_timings"] = {}
    if "status_history" not in state or state["status_history"] is None:
        state["status_history"] = []

    state["active_agent"] = "Git"
    status_msg = "🔗 Git Agent processes the request..."
    state["status_message"] = status_msg
    state["status_history"].append(status_msg)

    submission_id = state.get("submission_id")
    if submission_id:
        submission_states[submission_id] = cast("WorkflowState", dict(state))
        logger.info(f"Updated status: {status_msg} for submission {submission_id}")

    # check if necessary variables exist
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in support git agent"
        )

    if not git_token:
        raise AgentRunMethodParameterError("git_token is required in git agent")

    if not tools_llm:
        raise AgentRunMethodParameterError("tools_llm is required in git agent")

    openai_mcp_tool: "dict[str, Any]" = {
        "type": "mcp",
        "server_label": "github",
        "server_url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": f"Bearer {git_token}"},
        "allowed_tools": ["issue_write", "add_issue_comment"],
    }

    github_mcp_available = True
    try:
        logger.info("git_agent GIT calling response api")

        resp = openai_client.responses.create(
            model=tools_llm,
            input=WorkflowAgentPrompts.GIT_UPSERT_PROMPT.format(
                github_url=github_url,
                sub_id=state["submission_id"],
                user_question=state["input"],
            ),
            tools=[openai_mcp_tool],  # type: ignore[arg-type]
        )
        logger.debug("git_agent response returned")

        resp_obj: "ResponseObject" = cast(ResponseObject, resp)
        state["github_issue"] = extract_mcp_output(
            resp_obj, agent_name="git_agent", extract_url=True
        )
        gh_issue = state.get("github_issue", "gh issue not set")
        bodies = [
            state["rag_sources"],
            state["classification_message"],
            state["mcp_output"],
        ]
    except Exception as e:
        github_mcp_available = False
        state["github_issue"] = ""
        logger.info(
            f"Git Agent unsuccessful MCP request "
            f"for submission {state['submission_id']} with error: '{e}'"
        )

    github_mcp_comment_failed = False
    if github_mcp_available:
        try:
            for body in bodies:
                # Convert body to string based on type
                if isinstance(body, str):
                    body_as_string = body
                elif isinstance(body, list):
                    body_as_string = ""
                    for item in body:
                        body_as_string += str(item)
                else:
                    body_as_string = str(body)

                # Truncate to first 65,500 characters as github comments have a
                # max length of 65,536 characters
                # and the github server errors out (in a not obvious way) when
                # we exceed that number;
                # some test k8s performance top output went well over 100K
                # characters
                body_as_string = body_as_string[:65500]

                logger.info(f"git_agent COMMENT of len {len(body_as_string)}")
                resp = openai_client.responses.create(
                    model=tools_llm,
                    input=WorkflowAgentPrompts.GIT_COMMENT_PROMPT.format(
                        github_url=github_url,
                        issue_id=gh_issue,
                        comment_body=body_as_string,
                    ),
                    tools=[openai_mcp_tool],  # type: ignore[arg-type]
                    timeout=240.0,  # the bigger comment calls can take
                    # especially long ... put a cap on it
                )
                logger.info(
                    f"git_agent COMMENT {body_as_string[:100]} response returned {resp}"
                )
                github_mcp_comment_failed = False

        except Exception as e:
            github_mcp_comment_failed = True
            logger.info(
                f"Git Agent unsuccessful MCP request for comment creation "
                f"for submission {state['submission_id']} with error: '{e}'"
            )

    agent_end_time = time.time()
    state["agent_timings"]["Git"] = agent_end_time - agent_start_time
    state["workflow_complete"] = True

    if github_mcp_available:
        completion_msg = "✅ Git Agent finished: GitHub issue created"
    else:
        completion_msg = "Git Agent finished: GitHub MCP Server Unavailable"
    if github_mcp_comment_failed:
        completion_msg = completion_msg + " but some comment creation failed"
    state["status_history"].append(completion_msg)
    state["status_message"] = completion_msg

    submission_id = state.get("submission_id")
    if submission_id:
        submission_states[submission_id] = cast("WorkflowState", dict(state))
        logger.info(f"Updated status: {completion_msg} for submission {submission_id}")

    return state


def pod_agent(
    state: "WorkflowState",
    openai_client: "OpenAI | None",
    tools_llm: "str | None",
) -> "WorkflowState":
    logger.info(f"K8S Agent request for submission: {state['submission_id']}")

    agent_start_time = time.time()
    if "agent_timings" not in state or state["agent_timings"] is None:
        state["agent_timings"] = {}
    if "status_history" not in state or state["status_history"] is None:
        state["status_history"] = []

    state["active_agent"] = "Pod"
    status_msg = "☸️ Pod Agent processes the request..."
    state["status_message"] = status_msg
    state["status_history"].append(status_msg)

    submission_id = state.get("submission_id")
    if submission_id:
        submission_states[submission_id] = cast("WorkflowState", dict(state))
        logger.info(f"Updated status: {status_msg} for submission {submission_id}")

    # check if necessary variables exist
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in support git agent"
        )

    if not tools_llm:
        raise AgentRunMethodParameterError("tools_llm is required in git agent")

    openai_mcp_tool: "dict[str, Any]" = {
        "type": "mcp",
        "server_label": "OpenShift / Kubernetes MCP Tools",
        "server_url": MCP_SERVER_URL,
        "require_approval": "never",
        "allowed_tools": ["pods_list_in_namespace"],
    }

    try:
        logger.debug(
            f"K8S Agent making MCP request for submission: {state['submission_id']}"
        )
        resp = openai_client.responses.create(
            model=tools_llm,
            input=WorkflowAgentPrompts.POD_PROMPT.format(namespace=state["namespace"]),
            tools=[openai_mcp_tool],  # type: ignore[arg-type]
        )
        logger.debug(
            f"K8S Agent successful return MCP request "
            f"for submission: {state['submission_id']}"
        )

        resp_obj: "ResponseObject" = cast(ResponseObject, resp)
        state["mcp_output"] = extract_mcp_output(resp_obj, agent_name="pod_agent")
    except Exception as e:
        state["mcp_output"] = "K8s MCP Server not available"
        logger.info(
            f"K8s Agent unsuccessful return MCP request "
            f"for submission {state['submission_id']} with error: '{e}'"
        )

    agent_end_time = time.time()
    state["agent_timings"]["Pod"] = agent_end_time - agent_start_time

    completion_msg = "✅ Pod Agent finished"
    state["status_history"].append(completion_msg)
    state["status_message"] = completion_msg

    submission_id = state.get("submission_id")
    if submission_id:
        submission_states[submission_id] = cast("WorkflowState", dict(state))
        logger.info(f"Updated status: {completion_msg} for submission {submission_id}")

    return state


def perf_agent(
    state: "WorkflowState",
    openai_client: "OpenAI | None",
    tools_llm: "str | None",
) -> "WorkflowState":
    logger.info(f"K8S perf Agent request for submission: {state['submission_id']}")

    agent_start_time = time.time()
    if "agent_timings" not in state or state["agent_timings"] is None:
        state["agent_timings"] = {}
    if "status_history" not in state or state["status_history"] is None:
        state["status_history"] = []

    state["active_agent"] = "Performance"
    status_msg = "⚡ Performance Agent processes the request..."
    state["status_message"] = status_msg
    state["status_history"].append(status_msg)

    submission_id = state.get("submission_id")
    if submission_id:
        submission_states[submission_id] = cast("WorkflowState", dict(state))
        logger.info(f"Updated status: {status_msg} for submission {submission_id}")

    # check if necessary variables exist
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in support git agent"
        )

    if not tools_llm:
        raise AgentRunMethodParameterError("tools_llm is required in git agent")

    openai_mcp_tool: "dict[str, Any]" = {
        "type": "mcp",
        "server_label": "OpenShift / Kubernetes MCP Tools",
        "server_url": MCP_SERVER_URL,
        "require_approval": "never",
        "allowed_tools": ["pods_top"],
    }

    try:
        logger.debug(
            f"K8S perf Agent making MCP request "
            f"for submission: {state['submission_id']}"
        )
        resp = openai_client.responses.create(
            model=tools_llm,
            input=WorkflowAgentPrompts.PERF_PROMPT.format(namespace=state["namespace"]),
            tools=[openai_mcp_tool],  # type: ignore[arg-type]
        )
        logger.debug(
            f"K8S perf Agent successful return MCP request "
            f"for submission: {state['submission_id']}"
        )
        resp_obj: "ResponseObject" = cast(ResponseObject, resp)
        state["mcp_output"] = extract_mcp_output(resp_obj, agent_name="perf_agent")
    except Exception as e:
        state["mcp_output"] = "K8s MCP server not available"
        logger.info(
            f"K8s perf Agent unsuccessful return MCP request "
            f"for submission {state['submission_id']} with error: '{e}'"
        )

    agent_end_time = time.time()
    state["agent_timings"]["Performance"] = agent_end_time - agent_start_time

    completion_msg = "✅ Performance Agent finished"
    state["status_history"].append(completion_msg)
    state["status_message"] = completion_msg

    submission_id = state.get("submission_id")
    if submission_id:
        submission_states[submission_id] = cast("WorkflowState", dict(state))
        logger.info(f"Updated status: {completion_msg} for submission {submission_id}")

    return state
