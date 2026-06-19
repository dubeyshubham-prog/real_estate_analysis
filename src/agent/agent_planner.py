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

        if any(
                word in query
                for word in [
                    "guide",
                    "document",
                    "knowledge",
                    "explain",
                    "what is",
                    "which area"
                ]
        ):
            return [
                "rag"
            ]

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

        if selected_tool == "analysis":

            return [
                "analysis"
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