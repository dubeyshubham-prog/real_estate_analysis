from pypdf import PdfReader

from src.common.exceptions import DataValidationError


class PDFLoader:
    def load_pdf(self, pdf_path) -> str:
        try:
            reader = PdfReader(pdf_path)
            page_text = [
                text.strip()
                for page in reader.pages
                if (text := page.extract_text())
            ]
        except Exception as error:
            raise DataValidationError(
                "The uploaded PDF could not be read"
            ) from error

        text = "\n\n".join(page_text).strip()
        if not text:
            raise DataValidationError(
                "The PDF contains no extractable text"
            )
        return text
