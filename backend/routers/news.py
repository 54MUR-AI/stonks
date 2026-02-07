from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, HttpUrl
from datetime import datetime

from backend.database import SessionLocal
from backend.models import User, NewsArticle, NewsSentiment
from backend.auth_service import get_current_active_user
from backend.services.news_intelligence.news_scraper_service import NewsScraperService


router = APIRouter(prefix="/api/v1/news", tags=["news"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class ScrapeNewsRequest(BaseModel):
    url: HttpUrl
    symbols: Optional[List[str]] = None
    analyze_sentiment: bool = True


class BatchScrapeRequest(BaseModel):
    urls: List[HttpUrl]
    symbols: Optional[List[str]] = None


class NewsArticleResponse(BaseModel):
    id: int
    url: str
    title: Optional[str]
    content_type: str
    author: Optional[str]
    publish_date: Optional[datetime]
    summary: Optional[str]
    key_points: Optional[List[str]]
    symbols: Optional[List[str]]
    scraped_at: datetime
    
    class Config:
        from_attributes = True


class SentimentResponse(BaseModel):
    id: int
    symbol: str
    sentiment_type: str
    sentiment_score: float
    confidence: Optional[float]
    analyzed_at: datetime
    model_used: Optional[str]
    
    class Config:
        from_attributes = True


class SentimentSummaryResponse(BaseModel):
    symbol: str
    total_articles: int
    sentiment_breakdown: dict
    average_score: float
    sentiment_percentages: Optional[dict] = None


@router.post("/scrape", response_model=NewsArticleResponse)
async def scrape_news(
    request: ScrapeNewsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    service = NewsScraperService(db)
    
    try:
        article = await service.scrape_and_store_news(
            url=str(request.url),
            symbols=request.symbols,
            analyze_sentiment=request.analyze_sentiment
        )
        return article
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scrape news: {str(e)}")


@router.post("/scrape/batch", response_model=List[NewsArticleResponse])
async def batch_scrape_news(
    request: BatchScrapeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    service = NewsScraperService(db)
    
    try:
        urls = [str(url) for url in request.urls]
        articles = await service.batch_scrape_news(urls, request.symbols)
        return articles
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to batch scrape: {str(e)}")


@router.get("/{symbol}", response_model=List[NewsArticleResponse])
async def get_news_by_symbol(
    symbol: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    service = NewsScraperService(db)
    articles = service.get_news_by_symbol(symbol.upper(), limit, offset)
    return articles


@router.get("/sentiment/{symbol}", response_model=List[SentimentResponse])
async def get_sentiment_by_symbol(
    symbol: str,
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    service = NewsScraperService(db)
    sentiments = service.get_sentiment_by_symbol(symbol.upper(), limit)
    return sentiments


@router.get("/sentiment/{symbol}/summary", response_model=SentimentSummaryResponse)
async def get_sentiment_summary(
    symbol: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    service = NewsScraperService(db)
    summary = service.get_sentiment_summary(symbol.upper())
    return summary


@router.get("/latest", response_model=List[NewsArticleResponse])
async def get_latest_news(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    service = NewsScraperService(db)
    articles = service.get_latest_news(limit)
    return articles


@router.get("/article/{article_id}", response_model=NewsArticleResponse)
async def get_article_by_id(
    article_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    article = db.query(NewsArticle).filter(NewsArticle.id == article_id).first()
    
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    return article
