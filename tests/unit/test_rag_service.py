from pathlib import Path
from uuid import uuid4

from src.rag.rag_services import RAGService


def test_rag_indexes_and_retrieves_without_network(tmp_path) -> None:
    rag = RAGService(
        database_path=tmp_path / "chroma",
        collection_name=f"test_{uuid4().hex}",
    )
    first = rag.add_documents(
        [
            "Dwarka Expressway is an emerging investment corridor.",
            "Golf Course Road contains premium residential properties.",
        ],
        source="first.pdf",
    )
    second = rag.add_documents(
        ["Sector 95 offers comparatively affordable housing."],
        source="second.pdf",
    )

    result = rag.ask("Which location is an emerging investment corridor?")

    assert first["chunks"] == 2
    assert second["chunks"] == 1
    assert result["status"] == "success"
    assert "Dwarka Expressway" in result["context"]


def test_empty_rag_asks_for_a_pdf(tmp_path) -> None:
    rag = RAGService(
        database_path=tmp_path / "empty-chroma",
        collection_name=f"test_{uuid4().hex}",
    )

    result = rag.ask("What does the document say?")

    assert result["status"] == "empty"
    assert "Upload a PDF" in result["answer"]


def test_pdf_ingestion_uses_isolated_database(tmp_path) -> None:
    rag = RAGService(
        database_path=tmp_path / "pdf-chroma",
        collection_name=f"test_{uuid4().hex}",
    )
    pdf_path = Path(__file__).parents[1] / "test_data" / "sample.pdf"

    result = rag.ingest_pdf(pdf_path)

    assert result["status"] == "success"
    assert result["chunks"] > 0
