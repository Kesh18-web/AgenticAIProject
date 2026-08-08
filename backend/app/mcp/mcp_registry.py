from typing import Any, Dict, List, Optional
from backend.app.core.logging import logger
from backend.app.mcp.browser_mcp import search_web
from backend.app.mcp.github_mcp import get_github_commits, get_github_issues_prs, search_github_code


class MCPToolRegistry:
    """Central Manager for Model Context Protocol (MCP) Tool Registrations and Execution."""

    def __init__(self):
        self.registered_tools = {
            "browser_search": "Execute live web search for real-time news, weather and external data",
            "github_code_search": "Search code files across any specified target GitHub repository ('owner/repo')",
            "github_issues_search": "Fetch open/closed issues and PRs from any specified target GitHub repository",
            "github_commits": "Fetch the most recent commits from any specified target GitHub repository",
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

                elif tool_clean == "github_code_search":
                    res = search_github_code(query=query, repo_name=repo_name)
                    mcp_results["data"]["github_code_search"] = res
                    mcp_results["executed_tools"].append("github_code_search")

                elif tool_clean == "github_issues_search":
                    res = get_github_issues_prs(repo_name=repo_name)
                    mcp_results["data"]["github_issues_search"] = res
                    mcp_results["executed_tools"].append("github_issues_search")

                elif tool_clean == "github_commits":
                    res = get_github_commits(repo_name=repo_name)
                    mcp_results["data"]["github_commits"] = res
                    mcp_results["executed_tools"].append("github_commits")

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
