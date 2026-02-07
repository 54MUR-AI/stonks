from abc import ABC, abstractmethod
from src.models import ScrapedContent


class BaseScraper(ABC):
    @abstractmethod
    async def scrape(self, url: str) -> ScrapedContent:
        pass
    
    @abstractmethod
    def can_handle(self, url: str) -> bool:
        pass
