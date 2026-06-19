from src.rag.rag_services import RAGService
from src.agent.estate_agent import EstateAgent


def test_rag_service():

    rag = RAGService()

    docs = [
        "Gurgaon real estate market is strong for investment due to infrastructure growth.",
        "Dwarka Expressway is considered an emerging real estate investment corridor.",
        "Affordable sectors may be suitable for buyers with limited budget."
    ]

    rag.add_documents(docs)

    result = rag.ask(
        "Which area is good for investment?"
    )

    print(result)

    assert result["status"] == "success"
    assert len(result["context"]) > 0

def test_agent_rag():

    agent = EstateAgent()

    result = agent.process_query(
        "Which area is good for investment?"
    )

    print(result)

    assert "rag" in result["execution_results"]
    assert result["tool_result"]["status"] == "success"

from pathlib import Path
from src.rag.rag_services import RAGService


def test_pdf_ingestion():

    rag = RAGService()

    current_dir = Path(__file__).parent

    pdf_path = (
        current_dir
        / "test_data"
        / "sample.pdf"
    )

    result = rag.ingest_pdf(
        str(pdf_path)
    )

    print(result)

    assert result["status"] == "success"