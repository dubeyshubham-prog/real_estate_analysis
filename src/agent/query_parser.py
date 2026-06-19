import re

class QueryParser:
    def extract_bhk(self, query):

        match = re.search(
            r"(\d+)\s*bhk",
            query.lower()
        )

        if match:
            return int(match.group(1))

        return None

    def extract_budget(self, query):
        query = query.lower()

        match = re.search(
            r"under\s*(\d+)\s*cr",
            query
        )

        if match:
            return float(match.group(1))

        return None

    def extract_property_type(self, query):

        query = query.lower()

        if "flat" in query or "apartment" in query:
            return "flat"

        if "house" in query or "villa" in query:
            return "house"

        return None

    def extract_purpose(self, query):

        query = query.lower()

        if "investment" in query:
            return "investment"

        if "family" in query:
            return "family"

        if "self use" in query:
            return "self_use"

        return None

    def extract_budget(self, query):
        query = query.lower()

        match = re.search(
            r"under\s*(\d+(\.\d+)?)\s*cr",
            query
        )

        if match:
            return float(match.group(1))

        return None

    def extract_intent(
            self,
            query
    ):

        query = query.lower()

        recommendation_keywords = [
            "recommend",
            "suggest",
            "best property",
            "investment"
        ]

        analysis_keywords = [
            "market",
            "analysis",
            "trend"
        ]

        price_keywords = [
            "price",
            "predict",
            "cost"
        ]

        for keyword in recommendation_keywords:

            if keyword in query:
                return "recommendation"

        for keyword in analysis_keywords:

            if keyword in query:
                return "analysis"

        for keyword in price_keywords:

            if keyword in query:
                return "price"

        return "price"

    def extract_property_name(self, query):

        query_lower = query.lower()

        trigger_phrases = [
            "similar to",
            "like",
            "based on"
        ]

        for phrase in trigger_phrases:

            if phrase in query_lower:

                property_name = query_lower.split(phrase)[-1]

                property_name = (
                    property_name
                    .replace("property", "")
                    .replace("project", "")
                    .strip()
                    .title()
                )

                return property_name

        return None

    def parse(self, query):

        return {
            "original_query": query,
            "intent": self.extract_intent(query),
            "bhk": self.extract_bhk(query),
            "budget": self.extract_budget(query),
            "property_type": self.extract_property_type(query),
            "purpose": self.extract_purpose(query),
            "property_name": self.extract_property_name(query)
        }