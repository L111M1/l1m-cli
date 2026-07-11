class L1mError(Exception):
    """Base error for user-facing CLI failures."""


class ConfigError(L1mError):
    pass


class ModelError(L1mError):
    pass


class WorkspaceError(L1mError):
    pass


class PromptError(L1mError):
    pass


class TaskError(L1mError):
    pass


class ToolError(L1mError):
    pass
