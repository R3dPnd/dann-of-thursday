"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class StatusEnum(str, Enum):
    """Status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    RUNNING = "running"


class MCPRequest(BaseModel):
    """MCP request model"""
    method: str = Field(..., description="MCP method name")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Method parameters")
    id: Optional[str] = Field(default=None, description="Request ID")


class MCPResponse(BaseModel):
    """MCP response model"""
    result: Optional[Dict[str, Any]] = Field(default=None, description="Response result")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error information")
    id: Optional[str] = Field(default=None, description="Request ID")


class ToolExecutionRequest(BaseModel):
    """Tool execution request model"""
    tool_name: str = Field(..., description="Name of the tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    timeout: Optional[int] = Field(default=30, description="Execution timeout in seconds")


class ToolExecutionResponse(BaseModel):
    """Tool execution response model"""
    tool_name: str
    status: StatusEnum
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

