from mcp.server.fastmcp import FastMCP
from backend.app.mcp.browser_mcp import search_web
from backend.app.mcp.fs_mcp import list_uploads_files, read_upload_file
from backend.app.mcp.github_mcp import get_github_issues_prs, search_github_code

# Initialize Official FastMCP Server
mcp_server = FastMCP("Enterprise-Analyst-MCP-Server")


@mcp_server.tool()
def fs_list_files():
    """MCP Tool: List all uploaded raw files stored in the server's uploads folder."""
    return list_uploads_files()


@mcp_server.tool()
def fs_read_file(filename: str):
    """MCP Tool: Read raw text content of an uploaded file from server storage with path safety."""
    return read_upload_file(filename=filename)


@mcp_server.tool()
def browser_search(query: str):
    """MCP Tool: Execute live web search for real-time compliance news and standards."""
    return search_web(query=query)


@mcp_server.tool()
def github_code_search(query: str, repo_name: str):
    """MCP Tool: Search code files across any specified target GitHub repository ('owner/repo')."""
    return search_github_code(query=query, repo_name=repo_name)


@mcp_server.tool()
def github_issues_search(repo_name: str, state: str = "open"):
    """MCP Tool: Fetch open/closed issues and pull requests from any specified target GitHub repository."""
    return get_github_issues_prs(repo_name=repo_name, state=state)


if __name__ == "__main__":
    # Exposes JSON-RPC 2.0 protocol over stdio for Claude Desktop, Cursor, and external LLM clients!
    mcp_server.run()
