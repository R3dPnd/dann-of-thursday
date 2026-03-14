"""
Custom exception classes
"""
from fastapi import HTTPException, status


class MCPAPIException(HTTPException):
    """Base exception for MCP API errors"""

    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


class ToolExecutionError(MCPAPIException):
    """Exception raised when tool execution fails"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution error: {detail}",
        )


class ResourceNotFoundError(MCPAPIException):
    """Exception raised when a resource is not found"""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} with identifier '{identifier}' not found",
        )


class ValidationError(MCPAPIException):
    """Exception raised when validation fails"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {detail}",
        )

