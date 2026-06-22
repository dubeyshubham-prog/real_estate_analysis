from pathlib import Path
from uuid import uuid4

import chromadb
from sklearn.feature_extraction.text import HashingVectorizer

from config.settings import Config
from src.common.exceptions import DataValidationError
from src.rag.pdf_loader import PDFLoader
from src.rag.text_chunker import TextChunker


class LocalTextEmbedder:
    """Create deterministic text embeddings without a network dependency."""

    def __init__(self, dimensions: int = 384):
        self.vectorizer = HashingVectorizer(
            n_features=dimensions,
            alternate_sign=False,
            norm="l2",
            stop_words="english",
        )

    def encode(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        is_single_text = isinstance(texts, str)
        values = [texts] if is_single_text else texts
        matrix = self.vectorizer.transform(values).toarray()
        embeddings = matrix.astype(float).tolist()
        return embeddings[0] if is_single_text else embeddings


class RAGService:
    def __init__(
        self,
        database_path=Config.CHROMA_DB_DIR,
        collection_name=Config.RAG_COLLECTION_NAME,
        result_limit=Config.RAG_RESULT_LIMIT,
    ):
        self.client = chromadb.PersistentClient(
            path=str(database_path),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
        )
        self.embedding_model = LocalTextEmbedder()
        self.result_limit = result_limit
        self.pdf_loader = PDFLoader()
        self.chunker = TextChunker()

    def add_documents(self, documents, source: str = "manual"):
        cleaned_documents = [
            document.strip()
            for document in documents
            if document and document.strip()
        ]
        if not cleaned_documents:
            raise DataValidationError("No readable document text was found")

        document_group = uuid4().hex
        ids = [
            f"{document_group}_{index}"
            for index in range(len(cleaned_documents))
        ]
        embeddings = self.embedding_model.encode(cleaned_documents)
        metadata = [
            {"source": source, "chunk": index}
            for index in range(len(cleaned_documents))
        ]
        self.collection.add(
            ids=ids,
            documents=cleaned_documents,
            embeddings=embeddings,
            metadatas=metadata,
        )

        return {
            "status": "success",
            "message": "Documents added to RAG knowledge base",
            "chunks": len(cleaned_documents),
        }

    def ask(self, query):
        query = query.strip()
        if not query:
            raise DataValidationError("Question cannot be empty")
        if self.collection.count() == 0:
            return {
                "status": "empty",
                "query": query,
                "context": "",
                "answer": (
                    "The knowledge base is empty. Upload a PDF before "
                    "asking document questions."
                ),
            }

        query_embedding = self.embedding_model.encode(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(self.result_limit, self.collection.count()),
        )
        documents = (results.get("documents") or [[]])[0]
        context = "\n".join(documents)

        return {
            "status": "success",
            "query": query,
            "context": context,
            "answer": (
                "Based on the real estate knowledge base:\n\n"
                f"{context}"
            )
        }

    def ingest_pdf(self, pdf_path):
        path = Path(pdf_path)
        text = self.pdf_loader.load_pdf(path)
        chunks = self.chunker.chunk_text(text)
        result = self.add_documents(chunks, source=path.name)
        return {
            "status": "success",
            "chunks": result["chunks"],
            "source": path.name,
        }
