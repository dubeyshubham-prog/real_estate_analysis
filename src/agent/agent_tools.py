from src.utils.analyser import PropertyAnalyzer
from src.utils.data_loader import DataLoader
from app.services.prediction_service import (
    PredictionService
)
from app.services.recommendation_service import RecommendationService
from src.agent.web_search_tool import WebSearchTool
from src.rag.rag_services import RAGService


class AgentTools:

    def __init__(self):
        self.data_loader = DataLoader()
        self.analyzer = PropertyAnalyzer()
        # self.recommendation_service = RecommendationService()
        self.web_search = WebSearchTool()
        self.rag = RAGService()

    def get_price_prediction(
            self,
            parsed_query
    ):
        return {
            "status": "redirect",
            "message": (
                "For accurate price prediction, please use the dedicated "
                "Price Prediction module where all required property details "
                "can be entered."
            ),
            "redirect_url": "/predict",
            "input": parsed_query
        }

    def get_recommendations(
            self,
            parsed_query
    ):

        recommendation_service = (
            RecommendationService()
        )

        property_name = (
            parsed_query.get(
                "property_name"
            )
        )

        if property_name is not None:
            recommendations = (
                recommendation_service
                .get_recommendations(
                    property_name=property_name,
                    top_n=5
                )
            )

            return {
                "status": "success",
                "recommendation_type": "similar_property",
                "message": "Similar property recommendation completed",
                "property_name": property_name,
                "recommendations": recommendations,
                "input": parsed_query
            }

        budget = parsed_query.get("budget")
        bhk = parsed_query.get("bhk")
        purpose = parsed_query.get("purpose")

        if budget is not None or bhk is not None or purpose is not None:
            df = self.data_loader.load_gurgaon_property_analysis_data()

            summary = self.analyzer.market_summary(df)

            return {
                "status": "success",
                "recommendation_type": "investment_guidance",
                "message": "Investment recommendation guidance generated",
                "budget": budget,
                "bhk": bhk,
                "purpose": purpose,
                "summary": summary,
                "input": parsed_query
            }

        return {
            "status": "missing_input",
            "message": (
                "Please provide either a property name for similar recommendations "
                "or your budget/BHK/purpose for investment guidance."
            ),
            "input": parsed_query
        }

    def get_market_analysis(
            self,
            parsed_query
    ):
        df = self.data_loader.load_gurgaon_property_analysis_data()

        summary = self.analyzer.market_summary(df)

        return {
            "status": "success",
            "message": "Market analysis completed",
            "summary": summary,
            "input": parsed_query
        }

    def get_visual_matches(
            self,
            image_path
    ):

        return {
            "status": "success",
            "message": "Deep Vision tool called",
            "image_path": image_path
        }

    def get_web_search_results(
            self,
            parsed_query
    ):

        query = parsed_query[
            "original_query"
        ]

        return self.web_search.search(
            query
        )

    def get_rag_answer(
            self,
            parsed_query
    ):

        return self.rag.ask(
            parsed_query["original_query"]
        )