class AgentReasoner:

    def reason(
            self,
            execution_results
    ):

        if "recommendation" in execution_results:

            recommendation_result = (
                execution_results["recommendation"]
            )

            if (
                    recommendation_result.get("status")
                    == "missing_input"
            ):

                return {
                    "action": "ask_user",
                    "message": (
                        "I can help you with property recommendations, "
                        "but I need one more detail. Please provide a property name "
                        "you like, so I can find similar properties."
                    ),
                    "reason": "Recommendation tool needs property_name",
                    "execution_results": execution_results
                }

        return {
            "action": "respond",
            "message": "Tool execution completed successfully.",
            "execution_results": execution_results
        }