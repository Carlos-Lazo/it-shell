import argparse
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from semantic_search_engine.bootstrap import create_engine
from semantic_search_engine.mcp.tools import MCPToolService
from semantic_search_engine.shared.paths import default_config_path


def create_server(tools: MCPToolService) -> FastMCP:
    server = FastMCP("it-shell")

    @server.tool()
    def search_commands(query: str, top_k: int = 10) -> dict[str, Any]:
        """Find relevant CLI command candidates for a natural-language query."""
        return tools.search_commands(query, top_k)

    @server.tool()
    def plan_operation(query: str, top_k: int = 10) -> dict[str, Any]:
        """Build a non-executable plan with candidates, dependencies, and warnings."""
        return tools.plan_operation(query, top_k)

    @server.tool()
    def validate_command(command: str) -> dict[str, Any]:
        """Check a command string for known risks; this tool never executes it."""
        return tools.validate_command(command)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="it-shell MCP server; does not execute commands")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
    )
    args = parser.parse_args()
    server = create_server(MCPToolService(create_engine(args.config)))
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
