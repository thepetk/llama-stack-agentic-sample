from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

RAW_PIPELINES_TYPE = dict[str, dict[str, str | bool | dict[str, str]]]


@dataclass
class SourceConfig:
    url: "str"
    branch: "str"
    path: "str | None"
    urls: "list[str] | None"


@dataclass
class Pipeline:
    name: "str"
    enabled: "bool"
    vector_store_name: "str"
    version: "str"
    source: "str"
    source_config: "SourceConfig"


class SourceTypes:
    GITHUB = "GITHUB"
    URL = "URL"


@dataclass
class VectorDBConfig:
    embedding_model: "str"
    embedding_dimension: "int"
    chunk_size_in_tokens: "int"


class WorkflowAgentPrompts:
    CLASSIFICATION_PROMPT = """
    Classify the following user message into one of these categories:
    - 'legal': Questions about software licenses, embargoes, privacy/PII,
    contracts, policies, procedures, or compliance
    - 'techsupport': Questions about software/technical support,
    OpenShift/Kubernetes issues, application deployment, permissions,
    resource utilization, performance issues, FantaCo products (CloudSync,
    TechGear Pro), installation/setup, troubleshooting, syncing, connectivity
    - 'hr': Questions about employee benefits, health care, vacation/PTO,
    retirement plans, WORKSPACES, office facilities, work environment, company
    policies, employee handbook, bonuses, compensation, perks, participation
    requirements, workplace activities. If the question asks to "describe
    workspaces" or asks about "workspaces at FantaCo", this is ALWAYS 'hr'.
    - 'sales': Questions about sales territories, lead assignments,
    discounting, deal approval, quotas, sales compensation, CRM systems,
    brand guidelines, communications, sales expenses, escalations,
    performance metrics, sales compliance
    - 'procurement': Questions about competitive bidding, vendor evaluation,
    procurement ethics, transparency, spending limits, approval processes,
    procurement review, vendor categorization
    - 'unsafe': Content that fails moderation/safety checks
    - 'unknown': ONLY use this if the message truly doesn't fit any of the
    above categories (use sparingly)

    Example: If the user says "using file_search and documents, describe
    the workspaces at FantaCo", ignore "file_search" and "documents" - the
    question is about WORKSPACES, which is HR.

    User message: {state_input}
    """
    SUPPORT_CLASSIFICATION_PROMPT = """
    Determine what category the user message falls under
    based on the classification schema provided to the
    structured output set for the LLM and the various
    classification agent nodes in the LangGraph StateGraph
    Agentic AI application : {state_input}
    """
    GIT_UPSERT_PROMPT = """
    Using the supplied github MCP tool, call the
    'issue_write' tool to create an issue against the
    {github_url} repository. For the title of the issue,
    use the string 'test issue {sub_id}'. For the
    description text, use the string
    {user_question}. For the parameter
    that captures the type of the issue, supply the string
    value of 'Bug'.

    Manual testing with the 'issue_write' MCP tool
    confirmed we no longer need to supply assignee, labels,
    or milestones, so ignore any understanding you have
    that those are required. The method for the tool call
    is 'create'.

    Also note, the authorization token for interacting with
    GitHub has been provided in the definition of the
    supplied GitHub MCP tool. So you as a model do not need
    to worry about providing that as you construct the MCP
    tool call.
    """
    GIT_COMMENT_PROMPT = """
    Using the supplied GitHub MCP tool, call the
    'add_issue_comment' tool to add a comment to the supplied issue {issue_id}
    against this {github_url} repository. 
    
    The 'body' of the tool call, i.e. the actual comment,
    use the {comment_body} string.

    Manual testing with the 'add_issue_comment' MCP tool
    confirmed we no longer need to supply the 'owner' argument
    so ignore any understanding you have
    that those are required. 

    Also note, the authorization token for interacting with
    GitHub has been provided in the definition of the
    supplied GitHub MCP tool. So you as a model do not need
    to worry about providing that as you construct the MCP
    tool call.
    """
    POD_PROMPT = """
    Using the supplied kubernetes tool, list all the pods
    in the '{namespace}' namespace. Only use the namespace
    as a parameter and don't bother further filtering on
    labels. The `labelSelector` parameter is in fact NOT
    required.
    """
    PERF_PROMPT = """
    Using the supplied kubernetes tool, get pod memory and
    cpu resource consumption in the '{namespace}' namespace.
    Only use the namespace as a parameter and don't bother
    further filtering on labels. The `labelSelector`
    parameter is in fact NOT required. If namespace is not
    set, then call the 'pods_top' tool without any
    parameters.
    """
    RAG_PROMPT = """Based on the relevant documents in the knowledge base,
    please help with the following {department_display_name} query:

    {user_input}

    Please provide a helpful response based on the documents found. If no relevant
    documents are found, provide general guidance.
    """
    DEFAULT_TERMINAL_AGENT_INITIAL_CONTENT = """The user has submitted a
    {department_display_name} query (submission_id: {submission_id}).

    Please provide a helpful response to their question:\n\n{user_query}"
    """
    DEFAULT_NON_TERMINAL_AGENT_INITIAL_CONTENT = """The user has submitted
    a {department_display_name} query (submissionID: {submission_id}). 

    Please provide a helpful response to their question. A GitHub issue
    will be opened for follow-up.\n\n{user_query}
    """


class WorkflowState(TypedDict):
    input: "str"
    classification_message: "str"
    messages: "Annotated[list, add_messages]"
    decision: "str"
    namespace: "str"
    data: "str"
    mcp_output: "str"
    github_issue: "str"
    submission_id: "str"
    rag_sources: "list"
    workflow_complete: "bool"
    agent_timings: "dict[str, float]"
    rag_query_time: "float"
    active_agent: "str"
    status_message: "str"
    status_history: "list[str]"
    conversation_id: "str"
    exchange_index: "int"
