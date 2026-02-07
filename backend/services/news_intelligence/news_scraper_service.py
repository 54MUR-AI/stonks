import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scraper"))

from typing import List, Optional, Dict
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc
import re

from scraper.scraper_manager import ScraperManager
from scraper.models import ScrapedContent, Summary
from backend.models import NewsArticle, NewsSentiment, SentimentType
from backend.services.news_intelligence.sentiment_analyzer import SentimentAnalyzer


class NewsScraperService:
    def __init__(self, db: Session):
        self.db = db
        self.scraper_manager = ScraperManager()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    async def scrape_and_store_news(
        self, 
        url: str, 
        symbols: Optional[List[str]] = None,
        analyze_sentiment: bool = True
    ) -> NewsArticle:
        existing = self.db.query(NewsArticle).filter(NewsArticle.url == url).first()
        if existing:
            return existing
        
        content, summary = await self.scraper_manager.scrape_and_summarize(url)
        
        if symbols is None:
            symbols = self._extract_symbols_from_content(content.content)
        
        news_article = NewsArticle(
            url=content.url,
            title=content.title or summary.title,
            content=content.content,
            content_type=content.content_type,
            author=content.author,
            publish_date=content.publish_date,
            summary=summary.summary,
            key_points=summary.key_points,
            metadata=content.metadata,
            symbols=symbols,
            scraped_at=datetime.utcnow()
        )
        
        self.db.add(news_article)
        self.db.flush()
        
        if analyze_sentiment and symbols:
            await self._analyze_and_store_sentiment(news_article, symbols)
        
        self.db.commit()
        self.db.refresh(news_article)
        
        return news_article
    
    async def _analyze_and_store_sentiment(
        self, 
        article: NewsArticle, 
        symbols: List[str]
    ):
        text = f"{article.title or ''} {article.summary or article.content[:1000]}"
        
        for symbol in symbols:
            sentiment_result = await self.sentiment_analyzer.analyze_sentiment(
                text, 
                symbol
            )
            
            sentiment = NewsSentiment(
                article_id=article.id,
                symbol=symbol,
                sentiment_type=sentiment_result["sentiment"],
                sentiment_score=sentiment_result["score"],
                confidence=sentiment_result.get("confidence"),
                model_used=sentiment_result.get("model", "vader"),
                analyzed_at=datetime.utcnow()
            )
            
            self.db.add(sentiment)
    
    def _extract_symbols_from_content(self, content: str) -> List[str]:
        pattern = r'\$([A-Z]{1,5})\b'
        symbols = re.findall(pattern, content)
        
        common_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 
                         'JPM', 'BAC', 'WMT', 'V', 'MA', 'DIS', 'NFLX', 'AMD']
        for ticker in common_tickers:
            if ticker in content.upper():
                symbols.append(ticker)
        
        return list(set(symbols))[:10]
    
    def get_news_by_symbol(
        self, 
        symbol: str, 
        limit: int = 50,
        offset: int = 0
    ) -> List[NewsArticle]:
        return (
            self.db.query(NewsArticle)
            .filter(NewsArticle.symbols.contains([symbol]))
            .order_by(desc(NewsArticle.publish_date))
            .offset(offset)
            .limit(limit)
            .all()
        )
    
    def get_sentiment_by_symbol(
        self, 
        symbol: str, 
        limit: int = 100
    ) -> List[NewsSentiment]:
        return (
            self.db.query(NewsSentiment)
            .filter(NewsSentiment.symbol == symbol)
            .order_by(desc(NewsSentiment.analyzed_at))
            .limit(limit)
            .all()
        )
    
    def get_sentiment_summary(self, symbol: str) -> Dict:
        sentiments = self.get_sentiment_by_symbol(symbol, limit=100)
        
        if not sentiments:
            return {
                "symbol": symbol,
                "total_articles": 0,
                "sentiment_breakdown": {},
                "average_score": 0.0
            }
        
        sentiment_counts = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
        
        total_score = 0.0
        
        for sentiment in sentiments:
            sentiment_counts[sentiment.sentiment_type] += 1
            total_score += sentiment.sentiment_score
        
        return {
            "symbol": symbol,
            "total_articles": len(sentiments),
            "sentiment_breakdown": sentiment_counts,
            "average_score": total_score / len(sentiments),
            "sentiment_percentages": {
                k: (v / len(sentiments)) * 100 
                for k, v in sentiment_counts.items()
            }
        }
    
    async def batch_scrape_news(
        self, 
        urls: List[str], 
        symbols: Optional[List[str]] = None
    ) -> List[NewsArticle]:
        articles = []
        
        for url in urls:
            try:
                article = await self.scrape_and_store_news(url, symbols)
                articles.append(article)
            except Exception as e:
                print(f"Error scraping {url}: {str(e)}")
                continue
        
        return articles
    
    def get_latest_news(self, limit: int = 20) -> List[NewsArticle]:
        return (
            self.db.query(NewsArticle)
            .order_by(desc(NewsArticle.scraped_at))
            .limit(limit)
            .all()
        )
