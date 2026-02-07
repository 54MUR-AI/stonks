import requests
import PyPDF2
from io import BytesIO
from src.models import ScrapedContent
from src.scrapers.base import BaseScraper


class PDFScraper(BaseScraper):
    async def scrape(self, url: str) -> ScrapedContent:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_content = []
        for page in pdf_reader.pages:
            text_content.append(page.extract_text())
        
        full_text = "\n\n".join(text_content)
        
        metadata = pdf_reader.metadata if pdf_reader.metadata else {}
        title = metadata.get('/Title', None)
        author = metadata.get('/Author', None)
        
        return ScrapedContent(
            url=url,
            title=title,
            content=full_text,
            content_type="pdf",
            author=author,
            metadata={
                "page_count": len(pdf_reader.pages),
                "pdf_metadata": {k: str(v) for k, v in metadata.items()} if metadata else {}
            }
        )
    
    def can_handle(self, url: str) -> bool:
        return url.lower().endswith('.pdf')
