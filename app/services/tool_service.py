"""
Tool execution service
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import logging
from app.models.schemas import ToolExecutionResponse, StatusEnum

logger = logging.getLogger(__name__)


class ToolService:
    """Service for executing security tools"""

    def __init__(self):
        """Initialize tool service"""
        self.available_tools = {
            "nmap": {
                "name": "nmap",
                "description": "Network mapper - port scanning and service detection",
                "parameters": {
                    "target": {"type": "string", "required": True, "description": "Target host or IP"},
                    "ports": {"type": "string", "required": False, "description": "Port range (e.g., '1-1000')"},
                    "scan_type": {"type": "string", "required": False, "description": "Scan type (e.g., 'syn', 'tcp')"},
                },
            },
            "sqlmap": {
                "name": "sqlmap",
                "description": "Automatic SQL injection and database takeover tool",
                "parameters": {
                    "url": {"type": "string", "required": True, "description": "Target URL"},
                    "data": {"type": "string", "required": False, "description": "POST data"},
                    "cookie": {"type": "string", "required": False, "description": "Cookie string"},
                },
            },
        }

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: Optional[int] = 30,
    ) -> ToolExecutionResponse:
        """
        Execute a security tool
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool-specific parameters
            timeout: Execution timeout in seconds
        
        Returns:
            ToolExecutionResponse: Tool execution result
        
        Raises:
            ValueError: If tool is not found or parameters are invalid
        """
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        tool_info = self.available_tools[tool_name]
        start_time = datetime.utcnow()

        logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")

        try:
            # Validate required parameters
            self._validate_parameters(tool_info, parameters)

            # Execute tool (placeholder - implement actual tool execution)
            output = await self._run_tool(tool_name, parameters, timeout)

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return ToolExecutionResponse(
                tool_name=tool_name,
                status=StatusEnum.SUCCESS,
                output=output,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Tool execution failed: {str(e)}")

            return ToolExecutionResponse(
                tool_name=tool_name,
                status=StatusEnum.ERROR,
                error=str(e),
                execution_time=execution_time,
            )

    def _validate_parameters(
        self, tool_info: Dict[str, Any], parameters: Dict[str, Any]
    ):
        """Validate tool parameters"""
        required_params = [
            name
            for name, param_info in tool_info["parameters"].items()
            if param_info.get("required", False)
        ]

        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Required parameter '{param}' is missing")

    async def _run_tool(
        self, tool_name: str, parameters: Dict[str, Any], timeout: int
    ) -> str:
        """Run the tool as a subprocess and return its stdout."""
        cmd = self._build_command(tool_name, parameters)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=float(timeout)
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TimeoutError(f"{tool_name} timed out after {timeout}s")

        output = stdout.decode(errors="replace").strip()
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            if err:
                output = f"{output}\n{err}".strip()
        return output or f"{tool_name} produced no output."

    def _build_command(self, tool_name: str, parameters: Dict[str, Any]) -> list:
        """Build the subprocess argument list for a given tool."""
        if tool_name == "nmap":
            cmd = ["nmap"]
            scan_type = parameters.get("scan_type", "").lower()
            if scan_type in ("syn", "tcp", "udp", "ping"):
                flag_map = {"syn": "-sS", "tcp": "-sT", "udp": "-sU", "ping": "-sn"}
                cmd.append(flag_map[scan_type])
            ports = parameters.get("ports", "")
            if ports:
                cmd.extend(["-p", ports])
            cmd.append(parameters["target"])
            return cmd

        if tool_name == "sqlmap":
            cmd = ["sqlmap", "-u", parameters["url"], "--batch", "--output-dir=/tmp/sqlmap"]
            data = parameters.get("data", "")
            if data:
                cmd.extend(["--data", data])
            cookie = parameters.get("cookie", "")
            if cookie:
                cmd.extend(["--cookie", cookie])
            return cmd

        raise ValueError(f"No command builder for tool: {tool_name}")

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools
        
        Returns:
            list: List of available tools with basic information
        """
        return [
            {
                "name": tool_info["name"],
                "description": tool_info["description"],
            }
            for tool_info in self.available_tools.values()
        ]

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a tool
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            dict: Tool information or None if not found
        """
        return self.available_tools.get(tool_name)

