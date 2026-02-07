import requests
from newspaper import Article
from readability import Document
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from src.models import ScrapedContent
from src.scrapers.base import BaseScraper
from datetime import datetime
from typing import Optional


class ArticleScraper(BaseScraper):
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def scrape(self, url: str) -> ScrapedContent:
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            content = article.text
            title = article.title
            author = ", ".join(article.authors) if article.authors else None
            publish_date = article.publish_date
            
            if not content or len(content) < 100:
                content, title = self._fallback_scrape(url)
            
            return ScrapedContent(
                url=url,
                title=title,
                content=content,
                content_type="article",
                author=author,
                publish_date=publish_date,
                metadata={
                    "top_image": article.top_image,
                    "images": article.images,
                    "keywords": article.keywords,
                }
            )
        except Exception as e:
            content, title = self._fallback_scrape(url)
            return ScrapedContent(
                url=url,
                title=title,
                content=content,
                content_type="article",
                metadata={"scraper_error": str(e)}
            )
    
    def _fallback_scrape(self, url: str) -> tuple[str, Optional[str]]:
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        
        doc = Document(response.text)
        title = doc.title()
        
        soup = BeautifulSoup(doc.summary(), 'lxml')
        content = md(str(soup))
        
        return content, title
    
    def can_handle(self, url: str) -> bool:
        article_domains = [
            'medium.com', 'substack.com', 'blog', 'article', 'news',
            'post', 'wordpress', 'blogspot'
        ]
        return any(domain in url.lower() for domain in article_domains)
