class AgentPlanner:

    def create_plan(
            self,
            selected_tool,
            parsed_query
    ):
        query = (
            parsed_query[
                "original_query"
            ].lower()
        )

        # The built-in analytics dataset covers Gurgaon only. Broader market
        # questions must use live sources instead of misusing local data.
        if parsed_query.get("market_scope") == "external":
            return [
                "web_search"
            ]

        if any(
                word in query
                for word in [
                    "latest",
                    "recent",
                    "today",
                    "news",
                    "current"
                ]
        ):
            return [
                "web_search"
            ]

        # Explicit market-analysis intent takes priority over generic
        # question phrases such as "what is".
        if selected_tool == "analysis":

            return [
                "analysis"
            ]

        # PDF retrieval is a separate product module. General knowledge
        # questions use live sources and never depend on uploaded documents.
        if any(
            phrase in query
            for phrase in [
                "guide",
                "explain",
                "what is",
                "which area",
                "document",
                "knowledge",
            ]
        ):
            return ["web_search"]

        purpose = parsed_query.get("purpose")
        budget = parsed_query.get("budget")
        property_name = parsed_query.get("property_name")
        bhk = parsed_query.get("bhk")

        if selected_tool == "price":

            return [
                "price"
            ]

        if property_name is not None:

            return [
                "recommendation"
            ]

        if purpose == "investment" or budget is not None or bhk is not None:

            return [
                "analysis",
                "recommendation"
            ]

        if selected_tool == "recommendation":

            return [
                "recommendation"
            ]

        if selected_tool == "vision":

            return [
                "vision"
            ]

        return [
            "general"
        ]
