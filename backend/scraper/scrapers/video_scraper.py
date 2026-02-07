from youtube_transcript_api import YouTubeTranscriptApi
from src.models import ScrapedContent
from src.scrapers.base import BaseScraper
import re
from typing import Optional


class VideoScraper(BaseScraper):
    async def scrape(self, url: str) -> ScrapedContent:
        video_id = self._extract_video_id(url)
        
        if not video_id:
            raise ValueError(f"Could not extract video ID from URL: {url}")
        
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([entry['text'] for entry in transcript_list])
            
            title = self._get_video_title(video_id)
            
            return ScrapedContent(
                url=url,
                title=title,
                content=transcript_text,
                content_type="video",
                metadata={
                    "video_id": video_id,
                    "platform": "youtube",
                    "transcript_language": "en"
                }
            )
        except Exception as e:
            raise Exception(f"Failed to fetch transcript: {str(e)}")
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
            r'youtube\.com\/embed\/([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _get_video_title(self, video_id: str) -> Optional[str]:
        try:
            import requests
            from bs4 import BeautifulSoup
            
            url = f"https://www.youtube.com/watch?v={video_id}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('meta', property='og:title')
            return title['content'] if title else None
        except:
            return None
    
    def can_handle(self, url: str) -> bool:
        return 'youtube.com' in url or 'youtu.be' in url
