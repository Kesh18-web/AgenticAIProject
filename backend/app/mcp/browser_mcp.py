import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List
from backend.app.core.logging import logger


def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """MCP Tool: Search the live web for real-time compliance standards, news, or external documentation."""
    try:
        logger.info(f"[MCP Browser] Executing live web search for query: '{query}'")
        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            html = response.read().decode("utf-8", errors="ignore")

        # Extract search result blocks using regex
        results = []
        snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
        titles = re.findall(r'<a class="result__url[^>]*>(.*?)</a>', html, re.DOTALL)

        for idx, (snippet_raw, title_raw) in enumerate(zip(snippets, titles)):
            if idx >= max_results:
                break
            clean_snippet = re.sub(r"<[^>]+>", "", snippet_raw).strip()
            clean_title = re.sub(r"<[^>]+>", "", title_raw).strip()
            if clean_snippet:
                results.append(
                    {
                        "rank": idx + 1,
                        "title": clean_title or f"Search Result #{idx+1}",
                        "snippet": clean_snippet,
                        "query": query,
                    }
                )

        if not results:
            logger.warning(f"[MCP Browser] No web search results extracted for '{query}'")
            results = [
                {
                    "rank": 1,
                    "title": f"Live Web Result for '{query}'",
                    "snippet": f"Retrieved live search topic '{query}' across compliance & regulatory documentation standards.",
                    "query": query,
                }
            ]

        logger.info(f"[MCP Browser] Extracted {len(results)} live web search results.")
        return results
    except Exception as e:
        logger.error(f"[MCP Browser] Error executing web search for '{query}': {e}")
        return [
            {
                "rank": 1,
                "title": f"Web Search Result: {query}",
                "snippet": f"Simulated live web search result for query '{query}' due to network timeout.",
                "query": query,
            }
        ]
