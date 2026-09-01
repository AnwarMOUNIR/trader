## 2026-08-31T15:30:00Z
You are m2_worker_1, a teamwork_preview_worker for Milestone M2 (RSS News Harvester & Sentiment NLP Engine).
Your working directory is: c:\Users\Anwar\Documents\trader\.agents\m2_worker_1
Your parent orchestrator is: c:\Users\Anwar\Documents\trader\.agents\orchestrator_1 (conversation ID: 221d5d39-4a0b-46d8-8d77-134da45be38f)
Authoritative user request file: c:\Users\Anwar\Documents\trader\.agents\ORIGINAL_REQUEST.md
Master project specification: c:\Users\Anwar\Documents\trader\PROJECT.md
Detailed technical analysis reference: c:\Users\Anwar\Documents\trader\.agents\survey_explorer_1\analysis.md
Mathematical specification reference: c:\Users\Anwar\Documents\trader\.agents\survey_spec_miner_1\analysis.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Scope & Exclusively Owned Files:
- `src/sentiment/__init__.py`
- `src/sentiment/rss_crawler.py`
- `src/sentiment/deduplicator.py`
- `src/sentiment/nlp_engine.py`
- `src/sentiment/aggregator.py`
- `tests/test_sentiment_pipeline.py`

Task:
Implement the complete RSS News Harvester & Sentiment NLP Engine meeting all R2 requirements:
1. `src/sentiment/rss_crawler.py`: Asynchronous/threaded polling of Yahoo Finance RSS, Google News RSS, and public financial feeds with ticker-specific query formulation for US and Moroccan stocks.
2. `src/sentiment/deduplicator.py`: Article deduplication using SHA-256 URL hashing and SimHash/token Jaccard similarity to eliminate syndication duplicates.
3. `src/sentiment/nlp_engine.py`: Dual-tier NLP sentiment analyzer (Tier 1: Loughran-McDonald & VADER financial lexicon for instant zero-latency scoring; Tier 2: FinBERT / HuggingFace transformer model with automatic batching and fallback). Output continuous sentiment scores in `[-1.0, 1.0]` and confidence in `[0.0, 1.0]`.
4. `src/sentiment/aggregator.py`: Point-in-time exponential time-decay aggregator computing rolling hourly/daily sentiment signals (half-life 4h & 24h, generating `sentiment_score_ema`, `sentiment_confidence_ema`, `news_volume_1h`, `news_volume_24h`, `sentiment_dispersion`, `sentiment_momentum`, `bull_bear_ratio`).
5. `tests/test_sentiment_pipeline.py`: Comprehensive unit and integration tests for crawler, deduplicator, NLP engine, and aggregator.
6. Run `pytest tests/test_sentiment_pipeline.py` to ensure 100% pass.
7. Write your execution report and handoff to `c:\Users\Anwar\Documents\trader\.agents\m2_worker_1\handoff.md`.
Send a completion message when finished.
