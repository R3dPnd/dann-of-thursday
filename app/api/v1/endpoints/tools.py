"""
Tool execution endpoints
"""
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import ToolExecutionRequest, ToolExecutionResponse
from app.services.tool_service import ToolService
from app.core.exceptions import ToolExecutionError

router = APIRouter()


@router.post("/execute", response_model=ToolExecutionResponse, summary="Execute a tool")
async def execute_tool(request: ToolExecutionRequest):
    """
    Execute a security tool with specified parameters
    
    Args:
        request: Tool execution request with tool name and parameters
    
    Returns:
        ToolExecutionResponse: Tool execution result
    
    Raises:
        HTTPException: If tool execution fails
    """
    try:
        tool_service = ToolService()
        result = await tool_service.execute_tool(
            tool_name=request.tool_name,
            parameters=request.parameters,
            timeout=request.timeout,
        )
        return result
    except Exception as e:
        raise ToolExecutionError(detail=str(e))


@router.get("/list", summary="List available tools")
async def list_tools():
    """
    List all available security tools
    
    Returns:
        dict: List of available tools with descriptions
    """
    tool_service = ToolService()
    return {
        "tools": tool_service.get_available_tools(),
        "count": len(tool_service.get_available_tools()),
    }


@router.get("/{tool_name}/info", summary="Get tool information")
async def get_tool_info(tool_name: str):
    """
    Get detailed information about a specific tool
    
    Args:
        tool_name: Name of the tool
    
    Returns:
        dict: Tool information including parameters and description
    """
    tool_service = ToolService()
    info = tool_service.get_tool_info(tool_name)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )
    return info

