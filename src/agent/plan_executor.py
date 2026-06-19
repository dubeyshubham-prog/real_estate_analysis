class PlanExecutor:

    def __init__(
            self,
            tools
    ):
        self.tools = tools

    def execute(
            self,
            plan,
            parsed_query
    ):

        results = {}

        for step in plan:

            if step == "analysis":

                results["analysis"] = (
                    self.tools.get_market_analysis(
                        parsed_query
                    )
                )

            elif step == "recommendation":

                results["recommendation"] = (
                    self.tools.get_recommendations(
                        parsed_query
                    )
                )

            elif step == "price":

                results["price"] = (
                    self.tools.get_price_prediction(
                        parsed_query
                    )
                )

            elif step == "web_search":
                results[
                    "web_search"
                ] = (
                    self.tools
                    .get_web_search_results(
                        parsed_query
                    )
                )

            elif step == "rag":

                results["rag"] = (
                    self.tools.get_rag_answer(
                        parsed_query
                    )
                )

        return results