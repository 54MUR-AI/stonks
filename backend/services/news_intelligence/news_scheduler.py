import asyncio
from typing import List, Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import logging

from backend.database import SessionLocal
from backend.services.news_intelligence.news_scraper_service import NewsScraperService

logger = logging.getLogger(__name__)


class NewsScheduler:
    def __init__(self):
        self.news_sources = {
            "financial_news": [
                "https://www.bloomberg.com/markets",
                "https://www.reuters.com/markets",
                "https://www.wsj.com/market-data",
                "https://www.ft.com/markets",
            ],
            "tech_news": [
                "https://techcrunch.com",
                "https://www.theverge.com",
            ],
            "crypto_news": [
                "https://cointelegraph.com",
                "https://decrypt.co",
            ]
        }
        
        self.watchlist_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
            "JPM", "BAC", "V", "MA", "WMT", "DIS", "NFLX", "AMD"
        ]
    
    async def scrape_news_for_symbols(
        self, 
        symbols: List[str],
        db: Session
    ) -> Dict:
        service = NewsScraperService(db)
        
        results = {
            "success": 0,
            "failed": 0,
            "articles": []
        }
        
        search_urls = []
        for symbol in symbols:
            search_urls.extend([
                f"https://finance.yahoo.com/quote/{symbol}/news",
                f"https://seekingalpha.com/symbol/{symbol}/news",
            ])
        
        for url in search_urls:
            try:
                article = await service.scrape_and_store_news(
                    url=url,
                    symbols=symbols,
                    analyze_sentiment=True
                )
                results["success"] += 1
                results["articles"].append(article.id)
                logger.info(f"Successfully scraped: {url}")
            except Exception as e:
                results["failed"] += 1
                logger.error(f"Failed to scrape {url}: {str(e)}")
        
        return results
    
    async def run_scheduled_scraping(self):
        logger.info("Starting scheduled news scraping...")
        
        db = SessionLocal()
        try:
            results = await self.scrape_news_for_symbols(
                self.watchlist_symbols,
                db
            )
            
            logger.info(
                f"Scheduled scraping completed. "
                f"Success: {results['success']}, Failed: {results['failed']}"
            )
            
            return results
        finally:
            db.close()
    
    async def start_scheduler(self, interval_hours: int = 1):
        logger.info(f"News scheduler started. Running every {interval_hours} hour(s)")
        
        while True:
            try:
                await self.run_scheduled_scraping()
            except Exception as e:
                logger.error(f"Error in scheduled scraping: {str(e)}")
            
            await asyncio.sleep(interval_hours * 3600)
    
    def add_news_source(self, category: str, url: str):
        if category not in self.news_sources:
            self.news_sources[category] = []
        
        if url not in self.news_sources[category]:
            self.news_sources[category].append(url)
            logger.info(f"Added news source: {url} to category {category}")
    
    def add_watchlist_symbol(self, symbol: str):
        symbol = symbol.upper()
        if symbol not in self.watchlist_symbols:
            self.watchlist_symbols.append(symbol)
            logger.info(f"Added {symbol} to watchlist")
    
    def remove_watchlist_symbol(self, symbol: str):
        symbol = symbol.upper()
        if symbol in self.watchlist_symbols:
            self.watchlist_symbols.remove(symbol)
            logger.info(f"Removed {symbol} from watchlist")


news_scheduler = NewsScheduler()
