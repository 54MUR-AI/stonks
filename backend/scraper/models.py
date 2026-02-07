from pydantic import BaseModel, HttpUrl
from typing import Optional, Literal
from datetime import datetime


class ScrapedContent(BaseModel):
    url: str
    title: Optional[str] = None
    content: str
    content_type: Literal["article", "video", "pdf", "social", "generic"]
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    metadata: dict = {}
    scraped_at: datetime = datetime.now()


class Summary(BaseModel):
    original_url: str
    title: Optional[str] = None
    summary: str
    key_points: list[str] = []
    content_type: str
    word_count: int
    generated_at: datetime = datetime.now()
    model_used: str
