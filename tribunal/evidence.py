"""The investigator's tool: looks up real facts on Wikipedia (free, no API key needed)."""

from urllib.parse import quote

import httpx

API = "https://en.wikipedia.org/w/api.php"
SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
# Wikimedia blocks clients that don't say who they are and where to find them.
HEADERS = {"User-Agent": "AgentTribunal/1.0 (https://hack.gdgriga.com/challenges/agent-tribunal) python-httpx"}


async def search_wikipedia(query: str) -> dict:
    """Searches English Wikipedia and returns the top 3 articles with a short summary.

    Args:
        query: a few keywords about one fact to check, e.g. "sleep deprivation memory".

    Returns:
        A dict with "results": a list of {title, summary, url}.
    """
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=8) as client:
            found = await client.get(API, params={
                "action": "query", "list": "search", "srsearch": query, "srlimit": 3, "format": "json",
            })
            titles = [hit["title"] for hit in found.json()["query"]["search"]]
            results = []
            for title in titles:
                page = (await client.get(SUMMARY + quote(title.replace(" ", "_")))).json()
                results.append({
                    "title": page.get("title", title),
                    "summary": (page.get("extract") or "")[:700],
                    "url": page.get("content_urls", {}).get("desktop", {}).get("page", ""),
                })
    except (httpx.HTTPError, KeyError, ValueError) as e:
        # No internet or Wikipedia hiccup: the investigator just reports nothing found.
        return {"results": [], "error": f"Wikipedia unavailable: {e}"}
    return {"results": results}
