from config.settings import Config


class LLMRouter:

    def __init__(self):

        self.allowed_intents = [
            "price",
            "recommendation",
            "analysis",
            "vision",
            "general"
        ]

    def build_prompt(
            self,
            query
    ):

        return f"""
You are an intent router for a real estate AI product.

Choose exactly one intent from:
price
recommendation
analysis
vision
general

Rules:

1. Use "analysis" when the user asks about:
- average price
- market price
- market trend
- expensive sectors
- affordable sectors
- Gurgaon market
- sector comparison
- overall market insights
The internal analysis dataset covers Gurgaon only. Global, international,
country-level, or other-city market questions must use "general"; the
planner will send them to live web search.

2. Use "price" only when the user wants to predict the price of a specific property.

3. Use "recommendation" when the user asks for similar properties or property suggestions.

4. Use "vision" when the user asks about images, rooms, interiors, or visual similarity.

5. PDF and uploaded-document questions belong to the separate RAG module,
not the AI assistant. Use "general" for general knowledge questions.

6. Use "general" for anything else.

Examples:

Query: What is the average price in Gurgaon?
Intent: analysis

Query: Tell me about the global real estate market
Intent: general

Query: What is real estate?
Intent: general

Query: Explain real estate investment
Intent: general

Query: Predict price of a 3 BHK flat
Intent: price

Query: Recommend properties similar to DLF The Arbour
Intent: recommendation

Query: Find similar images to this room
Intent: vision

User Query:
{query}

Return only the intent name.
"""

    def route(
            self,
            query
    ):
        if not Config.USE_OLLAMA_ROUTER:
            return self._route_with_rules(query)

        from ollama import chat

        response = chat(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "user",
                    "content": self.build_prompt(query)
                }
            ]
        )

        intent = (
            response["message"]["content"]
            .strip()
            .lower()
        )

        if intent not in self.allowed_intents:
            return "general"

        return intent

    @staticmethod
    def _route_with_rules(query):
        """Route reliably when a local Ollama server is unavailable."""
        normalized = query.lower()
        if any(word in normalized for word in ["image", "room", "visual"]):
            return "vision"
        if any(
            word in normalized
            for word in [
                "recommend",
                "similar property",
                "suggest",
                "investment",
                "invest",
            ]
        ):
            return "recommendation"
        if any(
            phrase in normalized
            for phrase in [
                "average price",
                "market",
                "trend",
                "expensive sector",
                "affordable sector",
            ]
        ):
            return "analysis"
        if any(word in normalized for word in ["predict", "valuation"]):
            return "price"
        return "general"
