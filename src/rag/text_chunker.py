class TextChunker:
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 800,
        overlap: int = 120,
    ) -> list[str]:
        """Split text into overlapping chunks to preserve nearby context."""
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Invalid chunk size or overlap")

        normalized_text = " ".join(text.split())
        chunks = []
        step = chunk_size - overlap
        for start in range(0, len(normalized_text), step):
            chunk = normalized_text[start:start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if start + chunk_size >= len(normalized_text):
                break
        return chunks
