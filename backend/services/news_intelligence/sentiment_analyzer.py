from typing import Dict, Optional
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        self.financial_lexicon = {
            'bullish': 3.0,
            'bearish': -3.0,
            'rally': 2.5,
            'crash': -3.5,
            'surge': 2.5,
            'plunge': -3.0,
            'soar': 2.5,
            'tumble': -2.5,
            'gain': 2.0,
            'loss': -2.0,
            'profit': 2.5,
            'revenue': 1.5,
            'earnings beat': 3.0,
            'earnings miss': -3.0,
            'upgrade': 2.5,
            'downgrade': -2.5,
            'outperform': 2.0,
            'underperform': -2.0,
            'buy': 2.0,
            'sell': -2.0,
            'hold': 0.0,
            'strong buy': 3.0,
            'strong sell': -3.0,
            'acquisition': 1.5,
            'merger': 1.5,
            'bankruptcy': -4.0,
            'lawsuit': -2.0,
            'innovation': 2.0,
            'breakthrough': 2.5,
            'scandal': -3.0,
            'fraud': -4.0,
            'growth': 2.0,
            'decline': -2.0,
            'expansion': 2.0,
            'layoff': -2.5,
            'hiring': 1.5,
            'dividend': 1.5,
            'buyback': 1.5,
            'debt': -1.5,
            'profit margin': 1.5,
            'market share': 1.5,
            'competition': -1.0,
            'regulation': -1.5,
            'approval': 2.0,
            'rejection': -2.5,
        }
        
        for word, score in self.financial_lexicon.items():
            self.vader.lexicon[word] = score
    
    async def analyze_sentiment(
        self, 
        text: str, 
        symbol: Optional[str] = None
    ) -> Dict:
        if symbol:
            text = self._extract_symbol_context(text, symbol)
        
        scores = self.vader.polarity_scores(text)
        
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        confidence = abs(compound)
        
        return {
            "sentiment": sentiment,
            "score": compound,
            "confidence": confidence,
            "positive": scores['pos'],
            "negative": scores['neg'],
            "neutral": scores['neu'],
            "model": "vader_financial"
        }
    
    def _extract_symbol_context(self, text: str, symbol: str) -> str:
        sentences = re.split(r'[.!?]+', text)
        
        relevant_sentences = []
        for sentence in sentences:
            if symbol.upper() in sentence.upper() or f"${symbol}" in sentence:
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            return ' '.join(relevant_sentences)
        
        return text
    
    async def batch_analyze(
        self, 
        texts: list[str], 
        symbol: Optional[str] = None
    ) -> list[Dict]:
        results = []
        for text in texts:
            result = await self.analyze_sentiment(text, symbol)
            results.append(result)
        return results
    
    def get_sentiment_trend(
        self, 
        sentiment_scores: list[float]
    ) -> Dict:
        if not sentiment_scores:
            return {
                "trend": "neutral",
                "average": 0.0,
                "volatility": 0.0
            }
        
        avg = sum(sentiment_scores) / len(sentiment_scores)
        
        if len(sentiment_scores) > 1:
            variance = sum((x - avg) ** 2 for x in sentiment_scores) / len(sentiment_scores)
            volatility = variance ** 0.5
        else:
            volatility = 0.0
        
        if avg >= 0.1:
            trend = "improving"
        elif avg <= -0.1:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "average": avg,
            "volatility": volatility,
            "count": len(sentiment_scores)
        }
