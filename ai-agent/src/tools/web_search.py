"""
web_search.py
Web search tool. Uses SerpAPI if SERPAPI_KEY is set; otherwise falls
back to a small offline stub so the agent (and its tests) still work
without a search API key — useful for demos and CI.
"""

from __future__ import annotations

import os

import requests

WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web for current information and return the top results "
        "(title, snippet, url). Use this when you need facts you don't "
        "already know or that may have changed recently."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default 3)",
            },
        },
        "required": ["query"],
    },
}

# Small offline fixture so the agent is demoable/testable without a live API key.
_OFFLINE_STUB_RESULTS = {
    "default": [
        {
            "title": "[offline stub] No SERPAPI_KEY configured",
            "snippet": (
                "This is a placeholder result. Set SERPAPI_KEY in .env to enable "
                "real web search, or replace this tool with any search API."
            ),
            "url": "https://serpapi.com/",
        }
    ]
}


def _live_search(query: str, num_results: int) -> list[dict]:
    api_key = os.getenv("SERPAPI_KEY")
    response = requests.get(
        "https://serpapi.com/search",
        params={"q": query, "api_key": api_key, "num": num_results},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    results = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
            }
        )
    return results


def run_web_search(query: str, num_results: int = 3) -> dict:
    api_key = os.getenv("SERPAPI_KEY")
    try:
        if api_key:
            results = _live_search(query, num_results)
        else:
            results = _OFFLINE_STUB_RESULTS["default"]
        return {"success": True, "query": query, "results": results}
    except Exception as exc:  # noqa: BLE001 — network/parse errors surfaced to the agent
        return {"success": False, "error": f"Search failed: {exc}"}
