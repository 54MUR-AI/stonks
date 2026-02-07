import requests
from src.models import ScrapedContent, Summary
from src.summarizers.base import BaseSummarizer
from src.config import settings


class HuggingFaceSummarizer(BaseSummarizer):
    def __init__(self):
        self.api_key = settings.huggingface_api_key
        self.model = settings.summarizer_model or "sshleifer/distilbart-cnn-12-6"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"
    
    async def summarize(self, content: ScrapedContent) -> Summary:
        max_input_length = 1024
        truncated_content = content.content[:max_input_length]
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json={
                    "inputs": truncated_content,
                    "parameters": {
                        "max_length": settings.max_summary_length,
                        "min_length": 50,
                        "do_sample": False
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                summary_text = result[0].get("summary_text", "")
            elif isinstance(result, dict):
                summary_text = result.get("summary_text", str(result))
            else:
                summary_text = str(result)
            
            key_points = self._extract_sentences(summary_text)
            
            return Summary(
                original_url=content.url,
                title=content.title,
                summary=summary_text,
                key_points=key_points[:5],
                content_type=content.content_type,
                word_count=len(summary_text.split()),
                model_used=self.model
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503:
                raise Exception(
                    "Hugging Face model is loading. Please wait a minute and try again. "
                    "Free tier models need to warm up."
                )
            elif e.response.status_code == 401:
                raise Exception(
                    "Invalid Hugging Face API key. Get a free key at: https://huggingface.co/settings/tokens"
                )
            else:
                raise Exception(f"Hugging Face API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Hugging Face summarization failed: {str(e)}")
    
    def _extract_sentences(self, text: str) -> list[str]:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences
