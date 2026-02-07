from anthropic import Anthropic
from src.models import ScrapedContent, Summary
from src.summarizers.base import BaseSummarizer
from src.config import settings


class AnthropicSummarizer(BaseSummarizer):
    def __init__(self):
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.summarizer_model
    
    async def summarize(self, content: ScrapedContent) -> Summary:
        prompt = self._build_prompt(content)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        summary_text = response.content[0].text
        key_points = self._extract_key_points(summary_text)
        
        return Summary(
            original_url=content.url,
            title=content.title,
            summary=summary_text,
            key_points=key_points,
            content_type=content.content_type,
            word_count=len(summary_text.split()),
            model_used=self.model
        )
    
    def _build_prompt(self, content: ScrapedContent) -> str:
        max_content_length = 12000
        truncated_content = content.content[:max_content_length]
        
        prompt = f"""Please summarize the following {content.content_type} content.

Title: {content.title or 'N/A'}
URL: {content.url}

Content:
{truncated_content}

Provide:
1. A concise summary (max {settings.max_summary_length} words)
2. Key points (as a bulleted list starting with "KEY POINTS:")

Format your response with the summary first, then the key points section."""
        
        return prompt
    
    def _extract_key_points(self, summary_text: str) -> list[str]:
        key_points = []
        if "KEY POINTS:" in summary_text:
            parts = summary_text.split("KEY POINTS:")
            if len(parts) > 1:
                points_section = parts[1].strip()
                lines = points_section.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                        key_points.append(line.lstrip('-•* ').strip())
        return key_points
