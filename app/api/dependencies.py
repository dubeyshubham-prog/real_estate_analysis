from functools import lru_cache

from app.services.agent_services import AgentService
from app.services.analysis_service import AnalysisService
from app.services.prediction_service import PredictionService
from app.services.recommendation_service import RecommendationService
from app.services.vision_services import VisionService
from src.rag.rag_services import RAGService


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    return PredictionService()


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    return RecommendationService()


@lru_cache(maxsize=1)
def get_analysis_service() -> AnalysisService:
    return AnalysisService()


@lru_cache(maxsize=1)
def get_vision_service() -> VisionService:
    return VisionService()


@lru_cache(maxsize=1)
def get_agent_service() -> AgentService:
    return AgentService()


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return RAGService()
