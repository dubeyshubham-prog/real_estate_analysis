from ollama import chat


class LLMRouter:

    def __init__(self):

        self.allowed_intents = [
            "price",
            "recommendation",
            "analysis",
            "vision",
            "rag",
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
rag
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

2. Use "price" only when the user wants to predict the price of a specific property.

3. Use "recommendation" when the user asks for similar properties or property suggestions.

4. Use "vision" when the user asks about images, rooms, interiors, or visual similarity.

5. Use "rag" when the user asks:
- what is
- explain
- guide
- document
- knowledge base
- uploaded document
- PDF related questions

6. Use "general" for anything else.

Examples:

Query: What is the average price in Gurgaon?
Intent: analysis

Query: What is real estate?
Intent: rag

Query: Explain real estate investment
Intent: rag

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