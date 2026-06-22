from src.agent.agent_planner import AgentPlanner
from src.agent.agent_tools import AgentTools
from src.agent.query_parser import QueryParser


def test_global_market_query_uses_web_search() -> None:
    parsed_query = QueryParser().parse(
        "Tell me about the world real estate market"
    )

    plan = AgentPlanner().create_plan("analysis", parsed_query)

    assert parsed_query["market_scope"] == "external"
    assert plan == ["web_search"]


def test_gurgaon_market_query_uses_local_analysis() -> None:
    parsed_query = QueryParser().parse(
        "Analyze the Gurgaon real estate market"
    )

    plan = AgentPlanner().create_plan("analysis", parsed_query)

    assert parsed_query["market_scope"] == "gurgaon"
    assert plan == ["analysis"]


class SearchRecorder:
    def __init__(self):
        self.query = None

    def search(self, query):
        self.query = query
        return {"status": "success", "results": [], "query": query}


def test_hinglish_world_query_is_normalized_for_web_search() -> None:
    search = SearchRecorder()
    tools = AgentTools(web_search=search)
    parsed_query = QueryParser().parse(
        "world ke real estate ke baare me batao"
    )

    result = tools.get_web_search_results(parsed_query)

    assert search.query == "global real estate market latest news"
    assert result["original_query"] == parsed_query["original_query"]


def test_general_explanation_does_not_use_pdf_rag() -> None:
    parsed_query = QueryParser().parse("What is real estate?")

    plan = AgentPlanner().create_plan("general", parsed_query)

    assert plan == ["web_search"]
    assert "rag" not in plan


def test_rule_router_identifies_investment_recommendation() -> None:
    from src.agent.llm_router import LLMRouter

    result = LLMRouter().route(
        "I want a 3 BHK flat under 2 Cr for investment"
    )

    assert result == "recommendation"
