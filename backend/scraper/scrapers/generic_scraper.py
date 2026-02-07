import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from src.models import ScrapedContent
from src.scrapers.base import BaseScraper


class GenericScraper(BaseScraper):
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def scrape(self, url: str) -> ScrapedContent:
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        title = None
        if soup.title:
            title = soup.title.string
        elif soup.find('h1'):
            title = soup.find('h1').get_text()
        
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if main_content:
            content = md(str(main_content))
        else:
            content = md(str(soup))
        
        return ScrapedContent(
            url=url,
            title=title,
            content=content,
            content_type="generic",
            metadata={"scraper": "generic"}
        )
    
    def can_handle(self, url: str) -> bool:
        return True
