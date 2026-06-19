import chromadb
from sentence_transformers import SentenceTransformer
from src.rag.pdf_loader import (
    PDFLoader
)

from src.rag.text_chunker import (
    TextChunker
)

class RAGService:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="data/chroma_db"
        )

        self.collection = self.client.get_or_create_collection(
            name="real_estate_knowledge"
        )

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.pdf_loader = PDFLoader()
        self.chunker = TextChunker()

    def add_documents(
            self,
            documents
    ):

        ids = []
        embeddings = []

        for index, doc in enumerate(documents):

            ids.append(
                f"doc_{index}"
            )

            embeddings.append(
                self.embedding_model
                .encode(doc)
                .tolist()
            )

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings
        )

        return {
            "status": "success",
            "message": "Documents added to RAG knowledge base"
        }

    def ask(
            self,
            query
    ):

        query_embedding = (
            self.embedding_model
            .encode(query)
            .tolist()
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        documents = results["documents"][0]

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

    def ingest_pdf(
            self,
            pdf_path
    ):
        text = (
            self.pdf_loader
            .load_pdf(pdf_path)
        )

        chunks = (
            self.chunker
            .chunk_text(text)
        )

        self.add_documents(
            chunks
        )

        return {
            "status": "success",
            "chunks": len(chunks)
        }