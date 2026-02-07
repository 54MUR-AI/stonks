from src.scrapers import ArticleScraper, VideoScraper, PDFScraper, GenericScraper
from src.summarizers import (
    OpenAISummarizer,
    AnthropicSummarizer,
    OllamaSummarizer,
    HuggingFaceSummarizer,
    SimpleSummarizer,
)
from src.models import ScrapedContent, Summary
from src.config import settings
from typing import Optional


class ScraperManager:
    def __init__(self):
        self.scrapers = [
            VideoScraper(),
            PDFScraper(),
            ArticleScraper(),
            GenericScraper(),
        ]
        
        if settings.summarizer_provider == "openai":
            self.summarizer = OpenAISummarizer()
        elif settings.summarizer_provider == "anthropic":
            self.summarizer = AnthropicSummarizer()
        elif settings.summarizer_provider == "ollama":
            self.summarizer = OllamaSummarizer()
        elif settings.summarizer_provider == "huggingface":
            self.summarizer = HuggingFaceSummarizer()
        elif settings.summarizer_provider == "simple":
            self.summarizer = SimpleSummarizer()
        else:
            raise ValueError(f"Unknown summarizer provider: {settings.summarizer_provider}")
    
    async def scrape(self, url: str) -> ScrapedContent:
        for scraper in self.scrapers:
            if scraper.can_handle(url):
                return await scraper.scrape(url)
        
        raise ValueError(f"No scraper found for URL: {url}")
    
    async def scrape_and_summarize(self, url: str) -> tuple[ScrapedContent, Summary]:
        content = await self.scrape(url)
        summary = await self.summarizer.summarize(content)
        return content, summary
