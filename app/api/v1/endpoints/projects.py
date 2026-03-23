"""GET /api/v1/projects — list of discovered Git projects."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("", summary="List available Git projects")
async def list_projects() -> JSONResponse:
    """Return all Git projects discovered by the Claude Code MCP server."""
    from src.mcp_servers.claude_code_server import _find_projects
    projects = _find_projects()
    return JSONResponse({"projects": projects, "count": len(projects)})


@router.post("/{project_name}/open", summary="Open project in Claude Code terminal")
async def open_project(project_name: str) -> JSONResponse:
    """Open a new Terminal window running Claude Code in the given project."""
    from src.mcp_servers.claude_code_server import open_claude_code
    result = open_claude_code(project_name)
    if result.startswith("Project") and "not found" in result:
        raise HTTPException(status_code=404, detail=result)
    return JSONResponse({"result": result})
