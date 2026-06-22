from src.agent.query_parser import QueryParser
from src.agent.agent_tools import AgentTools
from src.agent.response_generator import (
    AgentResponseGenerator
)
from src.agent.llm_router import LLMRouter
from src.agent.agent_planner import AgentPlanner
from src.agent.plan_executor import PlanExecutor
from src.agent.agent_reasoner import AgentReasoner


class EstateAgent:

    def __init__(self):

        self.parser = QueryParser()
        self.tools = AgentTools()

        self.response_generator = (
            AgentResponseGenerator()
        )

        self.llm_router = LLMRouter()
        self.planner = AgentPlanner()
        self.executor = PlanExecutor(
            self.tools
        )
        self.reasoner = AgentReasoner()

    def select_tool(
            self,
            query
    ):

        query = query.lower()

        if "price" in query:
            return "price"

        if "recommend" in query:
            return "recommendation"

        if "market" in query:
            return "analysis"

        return "price"

    def process_query(
            self,
            query
    ):

        parsed_query = self.parser.parse(
            query
        )

        try:

            selected_tool = (
                self.llm_router.route(query)
            )

        except Exception:

            selected_tool = (
                parsed_query["intent"]
            )

        plan = self.planner.create_plan(
            selected_tool,
            parsed_query
        )

        execution_results = (
            self.executor.execute(
                plan,
                parsed_query
            )
        )

        reasoning_result = (
            self.reasoner.reason(
                execution_results
            )
        )

        if "web_search" in execution_results:
            tool_result = execution_results["web_search"]
            results = tool_result.get("results", [])
            if results:
                formatted_results = "\n\n".join(
                    (
                        f"{index}. {item['title']}\n"
                        f"{item.get('snippet', '')}\n"
                        f"{item['url']}"
                    ).strip()
                    for index, item in enumerate(results, start=1)
                )
                final_response = (
                    "Here are the latest web results:\n\n"
                    f"{formatted_results}"
                )
            else:
                final_response = tool_result["message"]

            return {
                "selected_tool": selected_tool,
                "plan": plan,
                "parsed_query": parsed_query,
                "tool_result": tool_result,
                "final_response": final_response,
                "execution_results": execution_results,
                "reasoning_result": reasoning_result
            }

        tool_result = None
        final_response = None

        if reasoning_result["action"] == "ask_user":

            final_response = (
                reasoning_result["message"]
            )

        else:

            if (
                    "recommendation" in execution_results
                    and execution_results["recommendation"].get(
                "recommendation_type"
            ) == "investment_guidance"
            ):

                tool_result = execution_results["recommendation"]

                final_response = (
                    self.response_generator
                    .generate_investment_guidance_response(
                        tool_result
                    )
                )

            elif "recommendation" in execution_results:

                tool_result = execution_results["recommendation"]

                final_response = (
                    self.response_generator
                    .generate_recommendation_response(
                        tool_result
                    )
                )

            elif "analysis" in execution_results:

                tool_result = execution_results["analysis"]

                final_response = (
                    self.response_generator
                    .generate_market_analysis_response(
                        tool_result
                    )
                )

            elif "price" in execution_results:

                tool_result = execution_results["price"]

                final_response = (
                    tool_result["message"]
                )

        return {
            "selected_tool": selected_tool,
            "plan": plan,
            "parsed_query": parsed_query,
            "tool_result": tool_result,
            "final_response": final_response,
            "execution_results": execution_results,
            "reasoning_result": reasoning_result
        }
