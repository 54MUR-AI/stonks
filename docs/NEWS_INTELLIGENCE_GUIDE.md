# News Intelligence Service - Integration Guide

## Overview

The News Intelligence Service transforms STONKS into a comprehensive investment intelligence platform by integrating web scraping, AI-powered summarization, and sentiment analysis capabilities.

## Architecture

### Components

1. **Web Scraper Module** (`backend/scraper/`)
   - Article scraper (news sites, blogs)
   - Video scraper (YouTube transcripts)
   - PDF scraper (research reports, filings)
   - Generic scraper (fallback)

2. **News Intelligence Service** (`backend/services/news_intelligence/`)
   - `news_scraper_service.py` - Core scraping and storage logic
   - `sentiment_analyzer.py` - Financial sentiment analysis
   - `news_scheduler.py` - Automated news monitoring

3. **Database Models** (`backend/models.py`)
   - `NewsArticle` - Stores scraped news content
   - `NewsSentiment` - Stores sentiment analysis results
   - `ResearchDocument` - Stores research papers and filings

4. **API Endpoints** (`backend/routers/news.py`)
   - RESTful API for news operations
   - Authenticated access required

## Database Schema

### NewsArticle Table
```sql
- id: Primary key
- url: Unique URL (indexed)
- title: Article title
- content: Full text content
- content_type: article/video/pdf/social/generic
- author: Author name
- publish_date: Publication date
- scraped_at: Timestamp when scraped
- summary: AI-generated summary
- key_points: JSON array of key points
- metadata: JSON object with additional data
- symbols: JSON array of stock symbols
```

### NewsSentiment Table
```sql
- id: Primary key
- article_id: Foreign key to NewsArticle
- symbol: Stock symbol (indexed)
- sentiment_type: positive/negative/neutral
- sentiment_score: Float (-1.0 to 1.0)
- confidence: Confidence score
- analyzed_at: Analysis timestamp
- model_used: Sentiment model identifier
```

## API Usage

### Authentication
All endpoints require JWT authentication:
```python
headers = {
    "Authorization": f"Bearer {access_token}"
}
```

### Scrape Single Article
```bash
curl -X POST "http://localhost:8000/api/v1/news/scrape" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://finance.yahoo.com/news/apple-earnings",
    "symbols": ["AAPL"],
    "analyze_sentiment": true
  }'
```

### Get News for Symbol
```bash
curl "http://localhost:8000/api/v1/news/AAPL?limit=50" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Get Sentiment Summary
```bash
curl "http://localhost:8000/api/v1/news/sentiment/AAPL/summary" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Response:
```json
{
  "symbol": "AAPL",
  "total_articles": 45,
  "sentiment_breakdown": {
    "positive": 28,
    "negative": 7,
    "neutral": 10
  },
  "average_score": 0.35,
  "sentiment_percentages": {
    "positive": 62.2,
    "negative": 15.6,
    "neutral": 22.2
  }
}
```

## Sentiment Analysis

### Financial Lexicon
The sentiment analyzer uses a custom financial lexicon with 40+ terms:

**Positive Terms:**
- bullish (+3.0), rally (+2.5), surge (+2.5)
- earnings beat (+3.0), upgrade (+2.5)
- strong buy (+3.0), buy (+2.0)
- innovation (+2.0), breakthrough (+2.5)

**Negative Terms:**
- bearish (-3.0), crash (-3.5), plunge (-3.0)
- earnings miss (-3.0), downgrade (-2.5)
- strong sell (-3.0), sell (-2.0)
- bankruptcy (-4.0), fraud (-4.0)

### Scoring System
- **Compound Score**: -1.0 (very negative) to +1.0 (very positive)
- **Classification**:
  - Positive: score >= 0.05
  - Negative: score <= -0.05
  - Neutral: -0.05 < score < 0.05

### Context Extraction
When analyzing sentiment for a specific symbol, the analyzer:
1. Extracts sentences mentioning the symbol
2. Focuses analysis on relevant context
3. Falls back to full text if no mentions found

## Automated News Monitoring

### Scheduler Configuration
```python
from backend.services.news_intelligence.news_scheduler import news_scheduler

# Add watchlist symbols
news_scheduler.add_watchlist_symbol("AAPL")
news_scheduler.add_watchlist_symbol("MSFT")

# Add custom news sources
news_scheduler.add_news_source(
    "tech_news",
    "https://techcrunch.com/tag/apple"
)

# Start scheduler (runs every hour)
await news_scheduler.start_scheduler(interval_hours=1)
```

### Default Watchlist
The scheduler monitors these symbols by default:
- Tech: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, AMD
- Finance: JPM, BAC, V, MA
- Retail: WMT, DIS
- Streaming: NFLX

## Integration Examples

### Example 1: Daily News Digest
```python
from backend.services.news_intelligence.news_scraper_service import NewsScraperService
from backend.database import SessionLocal

async def generate_daily_digest(symbols: list[str]):
    db = SessionLocal()
    service = NewsScraperService(db)
    
    digest = {}
    for symbol in symbols:
        # Get latest news
        articles = service.get_news_by_symbol(symbol, limit=10)
        
        # Get sentiment summary
        sentiment = service.get_sentiment_summary(symbol)
        
        digest[symbol] = {
            "article_count": len(articles),
            "sentiment": sentiment,
            "top_headlines": [a.title for a in articles[:5]]
        }
    
    return digest
```

### Example 2: Sentiment-Based Alerts
```python
async def check_sentiment_alerts(symbol: str, threshold: float = 0.3):
    db = SessionLocal()
    service = NewsScraperService(db)
    
    summary = service.get_sentiment_summary(symbol)
    
    if abs(summary["average_score"]) > threshold:
        sentiment_type = "positive" if summary["average_score"] > 0 else "negative"
        
        return {
            "alert": True,
            "symbol": symbol,
            "sentiment": sentiment_type,
            "score": summary["average_score"],
            "article_count": summary["total_articles"]
        }
    
    return {"alert": False}
```

### Example 3: Research Document Analysis
```python
async def analyze_research_report(pdf_url: str, symbols: list[str]):
    db = SessionLocal()
    service = NewsScraperService(db)
    
    # Scrape PDF
    article = await service.scrape_and_store_news(
        url=pdf_url,
        symbols=symbols,
        analyze_sentiment=True
    )
    
    return {
        "title": article.title,
        "summary": article.summary,
        "key_points": article.key_points,
        "sentiment_by_symbol": {
            s.symbol: {
                "type": s.sentiment_type,
                "score": s.sentiment_score
            }
            for s in article.sentiments
        }
    }
```

## Environment Configuration

Add to `.env`:
```env
# Summarizer Configuration
SUMMARIZER_PROVIDER=openai  # openai, anthropic, ollama, huggingface, simple
SUMMARIZER_MODEL=gpt-4-turbo-preview
MAX_SUMMARY_LENGTH=500

# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HUGGINGFACE_API_KEY=hf_...

# Ollama Configuration (if using local models)
OLLAMA_BASE_URL=http://localhost:11434
```

## Database Migration

Run migrations to create new tables:
```bash
cd backend
python -c "from database import Base, engine; Base.metadata.create_all(bind=engine)"
```

## Testing

### Unit Tests
```python
import pytest
from backend.services.news_intelligence.sentiment_analyzer import SentimentAnalyzer

@pytest.mark.asyncio
async def test_sentiment_analysis():
    analyzer = SentimentAnalyzer()
    
    result = await analyzer.analyze_sentiment(
        "Apple stock surges after earnings beat expectations",
        "AAPL"
    )
    
    assert result["sentiment"] == "positive"
    assert result["score"] > 0.5
```

### Integration Tests
```python
@pytest.mark.asyncio
async def test_news_scraping(db_session):
    service = NewsScraperService(db_session)
    
    article = await service.scrape_and_store_news(
        url="https://example.com/test-article",
        symbols=["TEST"],
        analyze_sentiment=True
    )
    
    assert article.id is not None
    assert len(article.sentiments) > 0
```

## Performance Considerations

### Caching
- Scraped articles are stored in database to avoid re-scraping
- URL uniqueness constraint prevents duplicates

### Rate Limiting
- Implement rate limiting for external scraping
- Use delays between batch requests
- Respect robots.txt

### Optimization
- Use async/await for concurrent scraping
- Batch process multiple URLs
- Index frequently queried fields (url, symbol, scraped_at)

## Troubleshooting

### Common Issues

**Issue: Scraping fails with 403 error**
- Solution: Check User-Agent headers, use rotating proxies

**Issue: Sentiment analysis returns neutral for all articles**
- Solution: Verify financial lexicon is loaded, check text preprocessing

**Issue: YouTube transcript not available**
- Solution: Video may not have captions, try alternative sources

**Issue: PDF parsing errors**
- Solution: Ensure PDF is text-based (not scanned images)

## Future Enhancements

1. **SEC Filings Parser**
   - Automated 10-K, 10-Q, 8-K parsing
   - Risk factor extraction
   - Management discussion analysis

2. **Social Media Integration**
   - Reddit sentiment tracking
   - Twitter/X trend analysis
   - StockTwits integration

3. **Advanced NLP**
   - Named entity recognition
   - Event extraction
   - Relationship mapping

4. **Real-time Alerts**
   - WebSocket streaming for breaking news
   - Sentiment shift notifications
   - Unusual activity detection

## Support

For issues or questions:
- GitHub Issues: https://github.com/54MUR-AI/stonks/issues
- Documentation: https://github.com/54MUR-AI/stonks/docs

## License

MIT License - See LICENSE file for details
