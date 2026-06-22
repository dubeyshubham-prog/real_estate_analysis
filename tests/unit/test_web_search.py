import requests

from src.agent.web_search_tool import WebSearchTool


SEARCH_HTML = """
<div class="result">
  <a class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnews">
    Gurgaon Property News
  </a>
  <a class="result__snippet">Latest verified market update.</a>
</div>
"""


class FakeResponse:
    text = SEARCH_HTML

    def raise_for_status(self) -> None:
        return None


class SuccessfulSession:
    def get(self, *args, **kwargs) -> FakeResponse:
        return FakeResponse()


class FailingSession:
    def get(self, *args, **kwargs):
        raise requests.ConnectionError("offline")


def test_web_search_returns_structured_live_results() -> None:
    result = WebSearchTool(session=SuccessfulSession()).search(
        "latest Gurgaon property news"
    )

    assert result["status"] == "success"
    assert result["results"][0] == {
        "title": "Gurgaon Property News",
        "url": "https://example.com/news",
        "snippet": "Latest verified market update.",
    }


def test_web_search_reports_network_failure_cleanly() -> None:
    result = WebSearchTool(session=FailingSession()).search("latest news")

    assert result["status"] == "unavailable"
    assert result["results"] == []
