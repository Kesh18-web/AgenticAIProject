import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional
from backend.app.core.logging import logger

def _get_headers() -> Dict[str, str]:
    """Build GitHub API headers, dynamically injecting auth token on every call."""
    headers = {
        "User-Agent": "Enterprise-AI-Analyst-Agent",
        "Accept": "application/vnd.github.v3+json",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _normalize_repo_name(repo_input: Optional[str]) -> Optional[str]:
    """Normalize any GitHub URL or slug to 'owner/repo' format.

    Handles all of:
      - 'https://github.com/owner/repo'
      - 'https://github.com/owner/repo.git'
      - 'owner/repo'
      - '  owner/repo  ' (whitespace)
    """
    if not repo_input or not repo_input.strip():
        return None

    cleaned = repo_input.strip().rstrip("/")

    # Extract owner/repo from any github.com URL
    match = re.search(r"github\.com[/:]([^/\s]+/[^/\s]+?)(?:\.git)?$", cleaned)
    if match:
        return match.group(1)

    # Already in owner/repo form
    if re.match(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$", cleaned):
        return cleaned

    return None


def get_github_commits(repo_name: Optional[str] = None, per_page: int = 10) -> Dict[str, Any]:
    """MCP Tool: Fetch the most recent commits from a GitHub repository ('owner/repo')."""
    repo_clean = _normalize_repo_name(repo_name)
    if not repo_clean:
        logger.warning(f"[MCP GitHub] Commits fetch invoked with invalid repo: '{repo_name}'")
        return {
            "status": "repo_not_specified",
            "message": f"Could not parse a valid 'owner/repo' from: '{repo_name}'. Please provide a valid GitHub repository URL or slug.",
        }

    try:
        logger.info(f"[MCP GitHub] Fetching recent commits for repo '{repo_clean}'")
        url = f"https://api.github.com/repos/{repo_clean}/commits?per_page={per_page}"
        req = urllib.request.Request(url, headers=_get_headers())

        with urllib.request.urlopen(req, timeout=10) as response:
            commits_raw = json.loads(response.read().decode("utf-8"))

        commit_results = []
        for c in commits_raw[:per_page]:
            commit_info = c.get("commit", {})
            commit_results.append({
                "sha": c.get("sha", "")[:8],
                "message": commit_info.get("message", "").split("\n")[0],  # First line only
                "author": commit_info.get("author", {}).get("name", "Unknown"),
                "date": commit_info.get("author", {}).get("date", ""),
                "url": c.get("html_url", ""),
            })

        logger.info(f"[MCP GitHub] Fetched {len(commit_results)} commits from '{repo_clean}'")
        return {
            "repository": repo_clean,
            "total_fetched": len(commit_results),
            "commits": commit_results,
        }

    except Exception as e:
        logger.error(f"[MCP GitHub] Error fetching commits for repo '{repo_clean}': {e}")
        auth_hint = " Add a GITHUB_TOKEN to .env for authenticated access (5000 req/hr)." if not _GITHUB_TOKEN else ""
        return {
            "repository": repo_clean,
            "total_fetched": 0,
            "commits": [],
            "error": f"GitHub API error: {e}.{auth_hint}",
        }


def search_github_code(query: str, repo_name: Optional[str] = None) -> Dict[str, Any]:
    """MCP Tool: Search code files across ANY specified target GitHub repository ('owner/repo')."""
    repo_clean = _normalize_repo_name(repo_name)
    if not repo_clean:
        logger.warning(f"[MCP GitHub] Code search invoked without valid repo: '{repo_name}'")
        return {
            "status": "repo_not_specified",
            "message": f"Could not parse a valid 'owner/repo' from: '{repo_name}'.",
        }

    try:
        logger.info(f"[MCP GitHub] Searching code in repo '{repo_clean}' for query: '{query}'")
        encoded_q = urllib.parse.quote(f"{query} repo:{repo_clean}")
        url = f"https://api.github.com/search/code?q={encoded_q}"

        req = urllib.request.Request(url, headers=_get_headers())
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        items = data.get("items", [])
        code_results = []
        for item in items[:5]:
            code_results.append({
                "name": item.get("name"),
                "path": item.get("path"),
                "html_url": item.get("html_url"),
                "repository": repo_clean,
            })

        logger.info(f"[MCP GitHub] Found {len(code_results)} code files in '{repo_clean}'")
        return {
            "repository": repo_clean,
            "query": query,
            "total_count": data.get("total_count", len(code_results)),
            "items": code_results,
        }

    except Exception as e:
        logger.error(f"[MCP GitHub] Error searching code in repo '{repo_clean}': {e}")
        return {
            "repository": repo_clean,
            "query": query,
            "items": [],
            "error": f"GitHub API error: {e}",
        }


def get_github_issues_prs(repo_name: Optional[str] = None, state: str = "open") -> Dict[str, Any]:
    """MCP Tool: Fetch open/closed issues and pull requests from ANY specified target GitHub repository."""
    repo_clean = _normalize_repo_name(repo_name)
    if not repo_clean:
        logger.warning(f"[MCP GitHub] Issues/PR search invoked without valid repo: '{repo_name}'")
        return {
            "status": "repo_not_specified",
            "message": f"Could not parse a valid 'owner/repo' from: '{repo_name}'.",
        }

    try:
        logger.info(f"[MCP GitHub] Fetching {state} issues/PRs for repo '{repo_clean}'")
        url = f"https://api.github.com/repos/{repo_clean}/issues?state={state}&per_page=5"

        req = urllib.request.Request(url, headers=_get_headers())
        with urllib.request.urlopen(req, timeout=10) as response:
            issues = json.loads(response.read().decode("utf-8"))

        issue_results = []
        for issue in issues[:5]:
            issue_results.append({
                "number": issue.get("number"),
                "title": issue.get("title"),
                "is_pull_request": "pull_request" in issue,
                "html_url": issue.get("html_url"),
                "state": issue.get("state"),
                "repository": repo_clean,
            })

        logger.info(f"[MCP GitHub] Fetched {len(issue_results)} issues/PRs from '{repo_clean}'")
        return {
            "repository": repo_clean,
            "state_filter": state,
            "total_count": len(issue_results),
            "issues_and_prs": issue_results,
        }

    except Exception as e:
        logger.error(f"[MCP GitHub] Error fetching issues for repo '{repo_clean}': {e}")
        auth_hint = " Add a GITHUB_TOKEN to .env for authenticated access." if not _GITHUB_TOKEN else ""
        return {
            "repository": repo_clean,
            "state_filter": state,
            "total_count": 0,
            "issues_and_prs": [],
            "error": f"GitHub API error: {e}.{auth_hint}",
        }
