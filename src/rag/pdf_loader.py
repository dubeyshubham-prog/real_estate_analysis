from pypdf import PdfReader


class PDFLoader:

    def load_pdf(
            self,
            pdf_path
    ):

        reader = PdfReader(
            pdf_path
        )

        text = ""

        for page in reader.pages:

            page_text = (
                page.extract_text()
            )

            if page_text:
                text += page_text

        return text