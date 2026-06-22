from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse
from xml.etree import ElementTree

import requests

from config.settings import Config
from src.monitoring.logging import get_logger


class DuckDuckGoResultsParser(HTMLParser):
    """Extract result titles, links, and snippets from DuckDuckGo HTML."""

    def __init__(self):
        super().__init__()
        self.results = []
        self._current = None
        self._capture_title = False
        self._capture_snippet = False

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        classes = set(attrs.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._finish_result()
            self._current = {
                "title": "",
                "url": self._clean_url(attrs.get("href", "")),
                "snippet": "",
            }
            self._capture_title = True
        elif self._current and "result__snippet" in classes:
            self._capture_snippet = True

    def handle_data(self, data):
        if self._current and self._capture_title:
            self._current["title"] += data
        elif self._current and self._capture_snippet:
            self._current["snippet"] += data

    def handle_endtag(self, tag):
        if tag == "a" and self._capture_title:
            self._capture_title = False
        elif self._capture_snippet and tag in {"a", "div", "span"}:
            self._capture_snippet = False

    def close(self):
        self._finish_result()
        super().close()

    def _finish_result(self):
        if not self._current:
            return
        result = {
            key: unescape(value).strip()
            for key, value in self._current.items()
        }
        if result["title"] and result["url"]:
            self.results.append(result)
        self._current = None
        self._capture_title = False
        self._capture_snippet = False

    @staticmethod
    def _clean_url(url: str) -> str:
        parsed = urlparse(url)
        redirect_target = parse_qs(parsed.query).get("uddg")
        return unquote(redirect_target[0]) if redirect_target else url


class WebSearchTool:
    def __init__(
        self,
        search_url: str = Config.WEB_SEARCH_URL,
        fallback_url: str = Config.WEB_SEARCH_FALLBACK_URL,
        timeout: int = Config.WEB_SEARCH_TIMEOUT_SECONDS,
        session=None,
    ):
        self.search_url = search_url
        self.fallback_url = fallback_url
        self.timeout = timeout
        self.session = session or requests.Session()
        self.logger = get_logger(__name__)

    def search(self, query: str, limit: int = 5) -> dict:
        query = query.strip()
        if not query:
            return self._unavailable(query, "Search query cannot be empty")

        try:
            response = self.session.get(
                self.search_url,
                params={"q": query},
                timeout=self.timeout,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; PropLens/1.0)"
                    )
                },
            )
            response.raise_for_status()
        except requests.RequestException as error:
            self.logger.warning("Web search failed: %s", error)
            results = self._search_bing_rss(query, limit)
            if results:
                return {
                    "status": "success",
                    "message": "Live web search completed",
                    "query": query,
                    "results": results,
                }
            return self._unavailable(
                query,
                "Live web search is temporarily unavailable",
            )

        results = self._parse_duckduckgo(response.text, limit)
        if not results:
            results = self._search_bing_rss(query, limit)
        if not results:
            return self._unavailable(query, "No web results were found")

        return {
            "status": "success",
            "message": "Live web search completed",
            "query": query,
            "results": results,
        }

    @staticmethod
    def _parse_duckduckgo(content: str, limit: int) -> list[dict[str, str]]:
        parser = DuckDuckGoResultsParser()
        parser.feed(content)
        parser.close()
        return parser.results[:limit]

    def _search_bing_rss(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, str]]:
        try:
            response = self.session.get(
                self.fallback_url,
                params={"q": query, "format": "rss"},
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; PropLens/1.0)"},
            )
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
        except (requests.RequestException, ElementTree.ParseError) as error:
            self.logger.warning("Fallback web search failed: %s", error)
            return []

        results = []
        for item in root.findall("./channel/item")[:limit]:
            title = (item.findtext("title") or "").strip()
            url = self._clean_bing_url(
                (item.findtext("link") or "").strip()
            )
            snippet = unescape(
                (item.findtext("description") or "").strip()
            )
            if title and url:
                results.append(
                    {"title": title, "url": url, "snippet": snippet}
                )
        return results

    @staticmethod
    def _clean_bing_url(url: str) -> str:
        target = parse_qs(urlparse(url).query).get("url")
        return unquote(target[0]) if target else url

    @staticmethod
    def _unavailable(query: str, message: str) -> dict:
        return {
            "status": "unavailable",
            "message": message,
            "query": query,
            "results": [],
        }
