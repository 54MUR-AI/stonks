from src.models import ScrapedContent, Summary
from src.summarizers.base import BaseSummarizer
from src.config import settings
import re


class SimpleSummarizer(BaseSummarizer):
    async def summarize(self, content: ScrapedContent) -> Summary:
        text = content.content
        sentences = self._extract_sentences(text)
        
        top_sentences = sentences[:10]
        summary_text = " ".join(top_sentences)
        
        key_points = self._extract_key_points(sentences)
        
        return Summary(
            original_url=content.url,
            title=content.title,
            summary=summary_text,
            key_points=key_points,
            content_type=content.content_type,
            word_count=len(summary_text.split()),
            model_used="simple-extractive"
        )
    
    def _extract_sentences(self, text: str) -> list[str]:
        text = re.sub(r'\s+', ' ', text)
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return sentences[:50]
    
    def _extract_key_points(self, sentences: list[str]) -> list[str]:
        key_points = []
        
        for sentence in sentences[:15]:
            if any(keyword in sentence.lower() for keyword in 
                   ['important', 'key', 'main', 'significant', 'critical', 'essential']):
                key_points.append(sentence)
                if len(key_points) >= 5:
                    break
        
        if len(key_points) < 3:
            key_points = sentences[:5]
        
        return key_points
