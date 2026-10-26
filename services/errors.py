from __future__ import annotations


class PluginError(Exception):
    """插件内部异常。message 必须是可以安全展示给用户的中文文本。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.user_message = message


class ValidationError(PluginError):
    pass


class NotFoundError(PluginError):
    pass


class PermissionDeniedError(PluginError):
    pass


class SafetyError(PluginError):
    pass


class ConfigError(PluginError):
    pass


class ConflictError(PluginError):
    pass


class BusyError(PluginError):
    pass


class ImageApiError(PluginError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        endpoint_name: str = "",
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.endpoint_name = endpoint_name
