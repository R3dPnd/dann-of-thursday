"""
Model Context Protocol (MCP) endpoints
"""
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import MCPRequest, MCPResponse
from app.services.mcp_service import MCPService

router = APIRouter()


@router.post("/request", response_model=MCPResponse, summary="Execute MCP request")
async def execute_mcp_request(request: MCPRequest):
    """
    Execute a Model Context Protocol request
    
    Args:
        request: MCP request containing method and parameters
    
    Returns:
        MCPResponse: MCP response with result or error
    
    Raises:
        HTTPException: If request execution fails
    """
    try:
        mcp_service = MCPService()
        result = await mcp_service.execute_request(
            method=request.method,
            params=request.params or {},
        )
        return MCPResponse(result=result, id=request.id)
    except Exception as e:
        return MCPResponse(
            error={"code": -1, "message": str(e)},
            id=request.id,
        )


@router.get("/methods", summary="List available MCP methods")
async def list_mcp_methods():
    """
    List all available MCP methods
    
    Returns:
        dict: List of available methods with descriptions
    """
    mcp_service = MCPService()
    return {
        "methods": mcp_service.get_available_methods(),
        "count": len(mcp_service.get_available_methods()),
    }

