from abc import ABC, abstractmethod
from src.models import ScrapedContent, Summary


class BaseSummarizer(ABC):
    @abstractmethod
    async def summarize(self, content: ScrapedContent) -> Summary:
        pass
