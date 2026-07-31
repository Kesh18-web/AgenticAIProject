from typing import Any, Dict, List, Optional
from backend.app.core.logging import logger
from backend.app.mcp.browser_mcp import search_web
from backend.app.mcp.fs_mcp import list_uploads_files, read_upload_file
from backend.app.mcp.github_mcp import get_github_issues_prs, search_github_code


class MCPToolRegistry:
    """Central Manager for Model Context Protocol (MCP) Tool Registrations and Execution."""

    def __init__(self):
        self.registered_tools = {
            "browser_search": "Execute live web search for real-time compliance news and standards",
            "fs_list_files": "List uploaded raw files in the server's uploads folder",
            "fs_read_file": "Read raw text content of an uploaded file from server storage",
            "github_code_search": "Search code files across any specified target GitHub repository ('owner/repo')",
            "github_issues_search": "Fetch open/closed issues and PRs from any specified target GitHub repository",
        }

    def execute_mcp_tools(
        self,
        tool_names: List[str],
        query: str,
        repo_name: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dispatch and execute requested MCP tools, aggregating structured results."""
        mcp_results: Dict[str, Any] = {"executed_tools": [], "data": {}}

        if not tool_names:
            return mcp_results

        for tool in tool_names:
            tool_clean = tool.strip().lower()
            try:
                if tool_clean == "browser_search":
                    res = search_web(query)
                    mcp_results["data"]["browser_search"] = res
                    mcp_results["executed_tools"].append("browser_search")

                elif tool_clean == "fs_list_files":
                    res = list_uploads_files()
                    mcp_results["data"]["fs_list_files"] = res
                    mcp_results["executed_tools"].append("fs_list_files")

                elif tool_clean == "fs_read_file":
                    target_file = filename or "SOC2_Security_Policy_2025.pdf"
                    res = read_upload_file(target_file)
                    mcp_results["data"]["fs_-read_file"] = res
                    mcp_results["executed_tools"].append("fs_read_file")

                elif tool_clean == "github_code_search":
                    res = search_github_code(query=query, repo_name=repo_name)
                    mcp_results["data"]["github_code_search"] = res
                    mcp_results["executed_tools"].append("github_code_search")

                elif tool_clean == "github_issues_search":
                    res = get_github_issues_prs(repo_name=repo_name)
                    mcp_results["data"]["github_issues_search"] = res
                    mcp_results["executed_tools"].append("github_issues_search")

                else:
                    logger.warning(f"[MCP Registry] Unknown tool requested: '{tool}'")
            except Exception as e:
                logger.error(f"[MCP Registry] Exception executing tool '{tool}': {e}")
                mcp_results["data"][tool] = {"error": f"Failed to execute tool '{tool}': {str(e)}"}

        logger.info(
            f"[MCP Registry] Executed {len(mcp_results['executed_tools'])} MCP tools: {mcp_results['executed_tools']}"
        )
        return mcp_results


# Global singleton instance
mcp_registry = MCPToolRegistry()
