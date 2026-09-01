"""
Comprehensive Unit & Integration Test Suite for Sentiment NLP Pipeline (Milestone M2).
Covers RSS Harvester, Deduplicator, NLP Engine (Tier 1 & 2), SQLite Store, and Sentiment Aggregator.
"""

import sys
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.sentiment.aggregator import SentimentAggregator
from src.sentiment.deduplicator import (
    ArticleDeduplicator,
    SQLiteNewsStore,
    canonicalize_url,
    compute_jaccard_similarity,
    compute_simhash,
    compute_url_hash,
    simhash_hamming_distance,
    tokenize_text,
)
from src.sentiment.nlp_engine import (
    FinancialLexiconScorer,
    NLPSentimentEngine,
    SentimentResult,
)
from src.sentiment.rss_crawler import (
    BVC_COMPANY_MAP,
    US_COMPANY_MAP,
    NewsArticle,
    RSSNewsCrawler,
    clean_html_text,
    parse_rfc822_or_iso_date,
)


# ---------------------------------------------------------------------------
# 1. Deduplication & URL Canonicalization Tests
# ---------------------------------------------------------------------------

def test_url_canonicalization():
    raw_url = "https://www.finance.yahoo.com/news/nvidia-ai-chip-surge-12345.html?utm_source=twitter&utm_medium=social&ref=homepage#comments"
    canonical = canonicalize_url(raw_url)

    assert "utm_source" not in canonical
    assert "utm_medium" not in canonical
    assert "ref=" not in canonical
    assert "#comments" not in canonical
    assert canonical.startswith("https://finance.yahoo.com/news/nvidia-ai-chip-surge-12345.html")

    # Hash determinism
    hash1 = compute_url_hash(raw_url)
    hash2 = compute_url_hash("https://finance.yahoo.com/news/nvidia-ai-chip-surge-12345.html")
    assert hash1 == hash2


def test_simhash_and_jaccard_fuzzy_deduplication():
    text1 = "NVIDIA Reports Record Q4 Revenue of $22 Billion as Data Center AI Surges"
    text2 = "NVIDIA Reports Record Q4 Revenues of $22B as Data Center AI Demand Surges"
    text3 = "Federal Reserve Holds Interest Rates Steady at 5.25% in Unanimous Decision"

    hash1 = compute_simhash(text1)
    hash2 = compute_simhash(text2)
    hash3 = compute_simhash(text3)

    dist_1_2 = simhash_hamming_distance(hash1, hash2)
    dist_1_3 = simhash_hamming_distance(hash1, hash3)

    # Near-duplicates should have smaller Hamming distance than completely different text
    assert dist_1_2 <= 14
    assert dist_1_3 > 20

    jaccard_1_2 = compute_jaccard_similarity(text1, text2)
    jaccard_1_3 = compute_jaccard_similarity(text1, text3)

    assert jaccard_1_2 >= 0.50
    assert jaccard_1_3 <= 0.10


def test_article_deduplicator_filter():
    dedup = ArticleDeduplicator(time_window_hours=24.0, simhash_max_distance=14, jaccard_threshold=0.50)
    now = datetime(2026, 8, 31, 14, 0, tzinfo=timezone.utc)

    art1 = NewsArticle.create(
        symbol="NVDA",
        market="US",
        title="NVIDIA Unveils Next-Generation Blackwell Ultra GPU",
        summary="NVIDIA announced its newest Blackwell Ultra architecture with 4x training speedup.",
        source="Reuters",
        url="https://reuters.com/technology/nvidia-blackwell-ultra-101",
        published_at_utc=now,
    )

    art2_syndicated = NewsArticle.create(
        symbol="NVDA",
        market="US",
        title="NVIDIA Unveils Next-Gen Blackwell Ultra GPUs",
        summary="NVIDIA announced its newest Blackwell Ultra architecture with 4x training speedup.",
        source="YahooFinance",
        url="https://finance.yahoo.com/news/nvidia-unveils-blackwell-ultra-202?utm_source=feed",
        published_at_utc=now + timedelta(minutes=15),
    )

    art3_different = NewsArticle.create(
        symbol="NVDA",
        market="US",
        title="DOJ Subpoenas NVIDIA in Artificial Intelligence Antitrust Probe",
        summary="Department of Justice expands regulatory inquiry into AI chip sales practices.",
        source="Bloomberg",
        url="https://bloomberg.com/news/doj-nvidia-probe-303",
        published_at_utc=now + timedelta(hours=2),
    )

    filtered = dedup.filter_duplicates([art1, art2_syndicated, art3_different])
    assert len(filtered) == 2
    assert filtered[0].title == art1.title
    assert filtered[1].title == art3_different.title


# ---------------------------------------------------------------------------
# 2. RSS News Crawler & XML Parsing Tests
# ---------------------------------------------------------------------------

SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Yahoo Finance: NVDA Headlines</title>
        <link>https://finance.yahoo.com/quote/NVDA</link>
        <description>Latest financial news for NVDA</description>
        <item>
            <title>NVIDIA beats Q2 earnings estimates on record AI chip sales</title>
            <link>https://finance.yahoo.com/news/nvidia-beats-q2-earnings-120000.html?ref=rss</link>
            <description>&lt;p&gt;NVIDIA reported quarterly earnings that exceeded analyst consensus, with revenue surging 122%.&lt;/p&gt;</description>
            <pubDate>Mon, 31 Aug 2026 12:00:00 GMT</pubDate>
            <source>Yahoo Finance</source>
        </item>
        <item>
            <title>Tech stocks rally as semiconductor guidance raised</title>
            <link>https://finance.yahoo.com/news/tech-stocks-rally-130000.html</link>
            <description>Chipmakers gained following positive forecast from industry leaders.</description>
            <pubDate>Mon, 31 Aug 2026 13:00:00 +0000</pubDate>
            <source>Reuters</source>
        </item>
    </channel>
</rss>
"""

SAMPLE_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
    <title>Google News - ATW.BVC</title>
    <entry>
        <title>Attijariwafa Bank annonce une hausse de 15% de son résultat net</title>
        <link href="https://news.google.com/articles/attijariwafa-hausse-benefice-456"/>
        <summary>Le groupe bancaire marocain enregistre des performances solides au premier semestre.</summary>
        <published>2026-08-31T10:30:00Z</published>
    </entry>
</feed>
"""


def test_rss_xml_parsing():
    crawler = RSSNewsCrawler()
    articles = crawler.parse_rss_xml(SAMPLE_RSS_XML, symbol="NVDA", market="US")

    assert len(articles) == 2
    assert articles[0].symbol == "NVDA"
    assert articles[0].market == "US"
    assert "beats Q2 earnings estimates" in articles[0].title
    assert "revenue surging 122%" in articles[0].summary
    assert articles[0].published_at_utc == datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    assert articles[0].article_id is not None

    assert articles[1].source == "Reuters"
    assert articles[1].published_at_utc == datetime(2026, 8, 31, 13, 0, tzinfo=timezone.utc)


def test_atom_xml_parsing():
    crawler = RSSNewsCrawler()
    articles = crawler.parse_rss_xml(SAMPLE_ATOM_XML, symbol="ATW.BVC", market="BVC", source_name="GoogleNews")

    assert len(articles) == 1
    assert articles[0].symbol == "ATW.BVC"
    assert articles[0].market == "BVC"
    assert "Attijariwafa Bank" in articles[0].title
    assert articles[0].published_at_utc == datetime(2026, 8, 31, 10, 30, tzinfo=timezone.utc)


def test_ticker_query_formulation():
    crawler = RSSNewsCrawler()
    us_url = crawler.build_google_news_url("NVDA", "US")
    assert "NVDA" in us_url or "NVIDIA" in us_url
    assert "hl=en-US" in us_url

    bvc_url = crawler.build_google_news_url("ATW.BVC", "BVC")
    unquoted = urllib.parse.unquote(bvc_url)
    assert "Attijariwafa" in unquoted or "ATW" in unquoted
    assert "hl=fr" in bvc_url


def test_crawler_custom_fetcher_and_universe_fetch():
    def mock_fetcher(url: str) -> str:
        if "NVDA" in url or "s=NVDA" in url:
            return SAMPLE_RSS_XML
        elif "ATW" in url or "BVC" in url:
            return SAMPLE_ATOM_XML
        return ""

    crawler = RSSNewsCrawler(custom_fetcher=mock_fetcher)
    univ_news = crawler.fetch_universe_news(symbols=["NVDA", "ATW.BVC"], max_workers=2)

    assert "NVDA" in univ_news
    assert len(univ_news["NVDA"]) >= 2
    assert "ATW.BVC" in univ_news
    assert len(univ_news["ATW.BVC"]) >= 1


# ---------------------------------------------------------------------------
# 3. Multi-Tier NLP Sentiment Engine Tests
# ---------------------------------------------------------------------------

def test_financial_lexicon_positive_sentiment():
    scorer = FinancialLexiconScorer()
    text = "NVIDIA beats earnings estimates as data center profit surges to all-time high with raised guidance."
    score, conf, label, probs = scorer.score_text(text, title="Record earnings beat")

    assert score >= 0.40
    assert conf >= 0.50
    assert label == "POSITIVE"
    assert probs["positive"] > probs["negative"]
    assert -1.0 <= score <= 1.0
    assert 0.0 <= conf <= 1.0


def test_financial_lexicon_negative_sentiment():
    scorer = FinancialLexiconScorer()
    text = "Intel slashes dividend and lowers guidance amid severe profit plunge and SEC accounting fraud investigation."
    score, conf, label, probs = scorer.score_text(text, title="Devastating guidance cut and lawsuit")

    assert score <= -0.40
    assert conf >= 0.50
    assert label == "NEGATIVE"
    assert probs["negative"] > probs["positive"]
    assert -1.0 <= score <= 1.0


def test_financial_lexicon_french_moroccan():
    scorer = FinancialLexiconScorer()

    # Positive Moroccan news
    text_pos = "Attijariwafa Bank annonce une forte progression du résultat net et une hausse du dividende pour les actionnaires."
    s_pos, c_pos, l_pos, p_pos = scorer.score_text(text_pos, title="Bénéfice net en hausse")
    assert s_pos > 0.20
    assert l_pos == "POSITIVE"

    # Negative Moroccan news
    text_neg = "Maroc Telecom subit une baisse du chiffre d'affaires et une lourde sanction avec repli des bénéfices."
    s_neg, c_neg, l_neg, p_neg = scorer.score_text(text_neg, title="Avertissement sur résultats et litige")
    assert s_neg < -0.20
    assert l_neg == "NEGATIVE"


def test_financial_lexicon_negation_and_intensifiers():
    scorer = FinancialLexiconScorer()

    # Negation
    s_pos, _, _, _ = scorer.score_text("Company achieves strong growth and high profit")
    s_neg_negated, _, _, _ = scorer.score_text("Company did not achieve growth and was not profitable")
    assert s_pos > 0.3
    assert s_neg_negated < 0.0

    # Intensifier
    s_standard, _, _, _ = scorer.score_text("Revenue gained")
    s_boosted, _, _, _ = scorer.score_text("Revenue significantly surged massively")
    assert s_boosted > s_standard


def test_nlp_engine_article_scoring():
    engine = NLPSentimentEngine(prefer_transformer=False)  # Deterministic test with Tier 1
    art = NewsArticle.create(
        symbol="AAPL",
        market="US",
        title="Apple reports record iPhone sales and raises dividend payout",
        summary="Quarterly revenue beat expectations driven by strong demand in services and hardware.",
        source="Bloomberg",
        url="https://bloomberg.com/apple-earnings-beat",
        published_at_utc=datetime(2026, 8, 31, 14, 0, tzinfo=timezone.utc),
    )

    res = engine.score_article(art)
    assert isinstance(res, SentimentResult)
    assert res.symbol == "AAPL"
    assert res.sentiment_score > 0.30
    assert res.sentiment_label == "POSITIVE"
    assert res.inference_latency_ms >= 0.0


def test_nlp_engine_batch_scoring():
    engine = NLPSentimentEngine(prefer_transformer=False)
    articles = [
        NewsArticle.create("AAPL", "US", "Apple beats revenue expectations", "Strong profit growth", "Source1", "https://url1.com", datetime.now(timezone.utc)),
        NewsArticle.create("TSLA", "US", "Tesla delivers record vehicles exceeding consensus", "Surging deliveries", "Source2", "https://url2.com", datetime.now(timezone.utc)),
        NewsArticle.create("INTC", "US", "Intel reports unexpected loss and slashes guidance", "Restructuring and layoffs", "Source3", "https://url3.com", datetime.now(timezone.utc)),
    ]

    results = engine.score_batch(articles)
    assert len(results) == 3
    assert results[0].sentiment_score > 0.15
    assert results[1].sentiment_score > 0.15
    assert results[2].sentiment_score < -0.15


# ---------------------------------------------------------------------------
# 4. SQLite Storage Persistence Tests
# ---------------------------------------------------------------------------

def test_sqlite_news_store_persistence(tmp_path):
    db_file = str(tmp_path / "test_news.db")
    store = SQLiteNewsStore(db_file)

    now = datetime(2026, 8, 31, 15, 0, tzinfo=timezone.utc)
    art1 = NewsArticle.create(
        symbol="NVDA",
        market="US",
        title="NVIDIA beats revenue estimates",
        summary="Strong AI demand.",
        source="Yahoo",
        url="https://yahoo.com/news/101",
        published_at_utc=now,
    )
    art2 = NewsArticle.create(
        symbol="NVDA",
        market="US",
        title="NVIDIA beats revenue estimates",
        summary="Duplicate test.",
        source="Yahoo",
        url="https://yahoo.com/news/101",  # Duplicate URL
        published_at_utc=now,
    )

    inserted_count = store.store_articles([art1, art2])
    assert inserted_count == 1  # Only 1 unique inserted

    articles_fetched = store.get_articles(symbol="NVDA")
    assert len(articles_fetched) == 1
    assert articles_fetched[0]["title"] == art1.title

    # Test Sentiment Score Persistence
    sent_res = SentimentResult(
        article_id=art1.article_id,
        symbol="NVDA",
        published_at_utc=now,
        sentiment_score=0.85,
        sentiment_confidence=0.90,
        sentiment_label="POSITIVE",
        probabilities={"positive": 0.85, "neutral": 0.10, "negative": 0.05},
        model_name="test_model",
    )
    store.store_sentiment_results([sent_res])

    scores_fetched = store.get_sentiment_scores(symbol="NVDA")
    assert len(scores_fetched) == 1
    assert scores_fetched[0]["sentiment_score"] == 0.85
    assert scores_fetched[0]["sentiment_label"] == "POSITIVE"


# ---------------------------------------------------------------------------
# 5. Exponential Decay Sentiment Aggregator & Point-in-Time Tests
# ---------------------------------------------------------------------------

def test_sentiment_aggregator_point_in_time_and_features():
    aggregator = SentimentAggregator()

    # Timeline of articles for NVDA
    t0 = datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 8, 31, 14, 0, tzinfo=timezone.utc)
    future_t = datetime(2026, 8, 31, 16, 0, tzinfo=timezone.utc)

    res0 = SentimentResult("id0", "NVDA", t0, 0.40, 0.70, "POSITIVE")
    res1 = SentimentResult("id1", "NVDA", t1, 0.80, 0.90, "POSITIVE")
    res2 = SentimentResult("id2", "NVDA", t2, -0.60, 0.85, "NEGATIVE")
    res_future = SentimentResult("id_fut", "NVDA", future_t, 0.99, 0.99, "POSITIVE")

    aggregator.add_sentiment_results([res0, res1, res2, res_future])

    # Target timestamps for 30-minute intraday evaluation bars
    eval_timestamps = pd.date_range(
        "2026-08-31 09:30:00",
        "2026-08-31 15:30:00",
        freq="30min",
        tz="UTC",
    )

    df_features = aggregator.get_sentiment_features(
        symbol="NVDA",
        target_timestamps=eval_timestamps,
        half_life_hours=4.0,
    )

    # 1. Check all required columns exist
    expected_cols = [
        "sentiment_score_ema",
        "sentiment_confidence_ema",
        "news_volume_1h",
        "news_volume_24h",
        "sentiment_dispersion",
        "sentiment_momentum",
        "bull_bear_ratio",
    ]
    for col in expected_cols:
        assert col in df_features.columns

    # 2. Strict Point-in-Time Causality Check:
    # At 09:30 UTC, no articles have been published yet -> sentiment must be 0.0
    bar_0930 = df_features.loc[pd.Timestamp("2026-08-31 09:30:00+00:00")]
    assert bar_0930["sentiment_score_ema"] == 0.0
    assert bar_0930["news_volume_1h"] == 0
    assert bar_0930["news_volume_24h"] == 0

    # At 12:30 UTC, only t0 (10:00) and t1 (12:00) have occurred; t2 (14:00) and future_t (16:00) must NOT leak
    bar_1230 = df_features.loc[pd.Timestamp("2026-08-31 12:30:00+00:00")]
    assert bar_1230["sentiment_score_ema"] > 0.50  # Strong positive
    assert bar_1230["news_volume_1h"] == 1  # only t1 is in last 1 hour
    assert bar_1230["news_volume_24h"] == 2  # t0 and t1 in last 24h

    # At 15:00 UTC, t2 (negative at 14:00) has arrived -> sentiment should pull down and dispersion increase
    bar_1500 = df_features.loc[pd.Timestamp("2026-08-31 15:00:00+00:00")]
    assert bar_1500["sentiment_dispersion"] > 0.10  # Conflict between positive and negative
    assert bar_1500["news_volume_1h"] == 1  # t2 at 14:00 is within 1h (14:00-15:00)

    # Future article at 16:00 must NOT affect any evaluation bar up to 15:30
    bar_1530 = df_features.loc[pd.Timestamp("2026-08-31 15:30:00+00:00")]
    assert bar_1530["sentiment_score_ema"] < 0.90  # Does not reflect the 0.99 future score


def test_sentiment_aggregator_half_life_decay():
    aggregator = SentimentAggregator()
    t_pub = datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc)
    res = SentimentResult("art1", "AAPL", t_pub, 1.0, 1.0, "POSITIVE")
    aggregator.add_sentiment_results([res])

    # Evaluate right at publication (10:00) and exactly 4 hours later (14:00)
    eval_ts = pd.to_datetime(["2026-08-31 10:00:00", "2026-08-31 14:00:00"], utc=True)
    df = aggregator.get_sentiment_features("AAPL", eval_ts, half_life_hours=4.0)

    # Single article score is 1.0 at both, but volume 1h is 1 at 10:00 and 0 at 14:00
    assert df.loc[eval_ts[0], "sentiment_score_ema"] == 1.0
    assert df.loc[eval_ts[0], "news_volume_1h"] == 1
    assert df.loc[eval_ts[1], "news_volume_1h"] == 0
    assert df.loc[eval_ts[1], "news_volume_24h"] == 1


def test_fuse_with_price_bars():
    aggregator = SentimentAggregator()
    t_pub = datetime(2026, 8, 31, 10, 15, tzinfo=timezone.utc)
    res = SentimentResult("art1", "TSLA", t_pub, 0.75, 0.85, "POSITIVE")
    aggregator.add_sentiment_results([res])

    dates = pd.date_range("2026-08-31 10:00:00", "2026-08-31 11:00:00", freq="15min", tz="UTC")
    ohlcv_df = pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 101.5, 103.0],
        "high": [101.5, 102.5, 103.0, 102.5, 104.0],
        "low": [99.5, 100.5, 101.0, 100.5, 102.0],
        "close": [101.0, 102.0, 101.5, 102.5, 103.5],
        "volume": [10000, 15000, 20000, 12000, 18000],
    }, index=dates)

    fused = aggregator.fuse_with_price_bars(ohlcv_df, symbol="TSLA", half_life_hours=4.0)

    assert "sentiment_score_ema" in fused.columns
    assert "news_volume_1h" in fused.columns
    assert len(fused) == len(ohlcv_df)

    # First bar at 10:00 has no news -> sentiment = 0.0
    assert fused.iloc[0]["sentiment_score_ema"] == 0.0
    # Second bar at 10:15 has the news -> sentiment = 0.75
    assert fused.iloc[1]["sentiment_score_ema"] == 0.75
    assert not fused.isna().any().any()  # No NaNs allowed


# ---------------------------------------------------------------------------
# 6. End-to-End Pipeline Integration Test
# ---------------------------------------------------------------------------

def test_full_sentiment_pipeline_end_to_end(tmp_path):
    db_path = str(tmp_path / "news_pipeline.db")
    store = SQLiteNewsStore(db_path)
    dedup = ArticleDeduplicator()
    nlp_engine = NLPSentimentEngine(prefer_transformer=False)
    aggregator = SentimentAggregator(news_store=store, nlp_engine=nlp_engine)

    # 1. Ingest via mock RSS feed
    mock_feed = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
        <channel>
            <title>Attijariwafa Bank News</title>
            <item>
                <title>Attijariwafa Bank publie des resultats solides avec benefice record</title>
                <link>https://leboursier.ma/attijariwafa-resultats-record?utm_source=twitter</link>
                <description>Le groupe bancaire depasse les previsions grace a la dynamique de ses activites.</description>
                <pubDate>Mon, 31 Aug 2026 09:00:00 GMT</pubDate>
            </item>
            <item>
                <title>Attijariwafa Bank publie des resultats solides avec benefice record</title>
                <link>https://medias24.com/attijariwafa-resultats-record?ref=rss</link>
                <description>Le groupe bancaire depasse les previsions grace a la dynamique de ses activites.</description>
                <pubDate>Mon, 31 Aug 2026 09:15:00 GMT</pubDate>
            </item>
        </channel>
    </rss>
    """
    crawler = RSSNewsCrawler(custom_fetcher=lambda url: mock_feed)
    raw_articles = crawler.fetch_symbol_news("ATW.BVC", market="BVC")
    assert len(raw_articles) == 2

    # 2. Deduplicate
    unique_articles = dedup.filter_duplicates(raw_articles)
    assert len(unique_articles) == 1  # Syndicated duplicate eliminated

    # 3. NLP Scoring & Storage
    sentiment_results = aggregator.add_articles(unique_articles)
    assert len(sentiment_results) == 1
    assert sentiment_results[0].sentiment_score > 0.20
    assert sentiment_results[0].sentiment_label == "POSITIVE"

    # 4. Feature Aggregation
    bars = pd.date_range("2026-08-31 08:30:00", "2026-08-31 14:30:00", freq="1h", tz="UTC")
    df_signals = aggregator.get_sentiment_features("ATW.BVC", bars, half_life_hours=4.0)

    assert len(df_signals) == len(bars)
    assert df_signals.loc[pd.Timestamp("2026-08-31 08:30:00+00:00"), "sentiment_score_ema"] == 0.0
    assert df_signals.loc[pd.Timestamp("2026-08-31 09:30:00+00:00"), "sentiment_score_ema"] > 0.20
    assert not df_signals.isna().any().any()
