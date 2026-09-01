"""
Sentiment NLP & RSS News Harvesting Subsystem.

Provides:
- RSSNewsCrawler: Real-time and historical financial RSS feed crawler (Google News, Yahoo Finance, Casablanca).
- ArticleDeduplicator: SHA-256 URL and SimHash/Token-Jaccard duplicate removal.
- SQLiteNewsStore: ACID-compliant news article and sentiment score database.
- NLPSentimentEngine: Multi-Tier NLP Sentiment Engine (FinBERT + Loughran-McDonald Lexicon).
- SentimentAggregator: Point-in-Time exponential time-decay sentiment feature aggregator.
"""

from src.sentiment.aggregator import SentimentAggregator
from src.sentiment.deduplicator import (
    ArticleDeduplicator,
    SQLiteNewsStore,
    canonicalize_url,
    compute_jaccard_similarity,
    compute_simhash,
    compute_url_hash,
    simhash_hamming_distance,
)
from src.sentiment.nlp_engine import (
    FinancialLexiconScorer,
    NLPSentimentEngine,
    SentimentResult,
    TransformerSentimentScorer,
)
from src.sentiment.rss_crawler import (
    BVC_COMPANY_MAP,
    US_COMPANY_MAP,
    NewsArticle,
    RSSNewsCrawler,
    clean_html_text,
    parse_rfc822_or_iso_date,
)

__all__ = [
    "NewsArticle",
    "RSSNewsCrawler",
    "US_COMPANY_MAP",
    "BVC_COMPANY_MAP",
    "clean_html_text",
    "parse_rfc822_or_iso_date",
    "ArticleDeduplicator",
    "SQLiteNewsStore",
    "canonicalize_url",
    "compute_url_hash",
    "compute_simhash",
    "simhash_hamming_distance",
    "compute_jaccard_similarity",
    "NLPSentimentEngine",
    "SentimentResult",
    "FinancialLexiconScorer",
    "TransformerSentimentScorer",
    "SentimentAggregator",
]
