class AgentRunMethodParameterError(Exception):
    """
    Raised when there is an error with the parameters
    provided to an agent run method.
    """

    pass


class NoVectorStoresFoundError(Exception):
    """
    Raised when no vector stores are found in the system.
    """

    pass
