"""
Article Deduplication and SQLite Storage Subsystem.

Provides URL canonicalization, SHA-256 URL hashing, 64-bit SimHash,
normalized token Jaccard similarity, and ACID-compliant SQLite news persistence.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


# Tracking query parameters to strip during URL canonicalization
TRACKING_PARAMS: Set[str] = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "fbclid",
    "gclid",
    "_ga",
    "ncid",
    "guccounter",
    "guce_referrer",
    "guce_referrer_sig",
    "cmp",
    "source",
    "ocid",
    "soc_src",
    "soc_trk",
}


def canonicalize_url(url: str) -> str:
    """
    Canonicalize a URL by normalizing scheme/netloc, removing tracking parameters,
    sorting query parameters, and stripping fragments and trailing slashes.
    """
    if not url:
        return ""
    parsed = urllib.parse.urlparse(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()

    # Normalize www. prefix
    if netloc.startswith("www."):
        netloc = netloc[4:]

    # Clean path: strip trailing slashes (unless path is empty or '/')
    path = parsed.path.rstrip("/")
    if not path:
        path = "/"

    # Clean query parameters
    query_params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
    filtered_params = [
        (k.lower(), v) for k, v in query_params if k.lower() not in TRACKING_PARAMS
    ]
    filtered_params.sort(key=lambda x: (x[0], x[1]))
    clean_query = urllib.parse.urlencode(filtered_params)

    # Reconstruct URL without fragment
    return urllib.parse.urlunparse((scheme, netloc, path, "", clean_query, ""))


def compute_url_hash(url: str) -> str:
    """Compute SHA-256 hash of a canonicalized URL."""
    canonical = canonicalize_url(url)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens of length >= 2."""
    if not text:
        return []
    tokens = re.findall(r"\b[a-zA-Z0-9_\u0600-\u06FF\u00C0-\u017F]{2,}\b", text.lower())
    return tokens


def extract_simhash_features(text: str) -> List[str]:
    """
    Extract multi-level features (words, word bigrams, character 4-grams) for robust SimHash.
    """
    tokens = tokenize_text(text)
    if not tokens:
        return []

    features = list(tokens)
    # Word bigrams
    for i in range(len(tokens) - 1):
        features.append(f"{tokens[i]}_{tokens[i+1]}")

    # Character 4-grams for sub-word robustness
    norm_text = " ".join(tokens)
    if len(norm_text) >= 4:
        for i in range(len(norm_text) - 3):
            features.append(norm_text[i : i + 4])

    return features


def compute_simhash(text: str, hash_bits: int = 64) -> int:
    """
    Compute a 64-bit SimHash integer fingerprint for fuzzy text deduplication.
    """
    features = extract_simhash_features(text)
    if not features:
        return 0

    v = [0] * hash_bits
    for feat in features:
        # Generate 64-bit hash using MD5
        feat_md5 = hashlib.md5(feat.encode("utf-8")).digest()
        feat_hash = int.from_bytes(feat_md5[:8], byteorder="big")

        for bit_idx in range(hash_bits):
            bit = (feat_hash >> bit_idx) & 1
            if bit:
                v[bit_idx] += 1
            else:
                v[bit_idx] -= 1

    fingerprint = 0
    for bit_idx in range(hash_bits):
        if v[bit_idx] > 0:
            fingerprint |= 1 << bit_idx

    return fingerprint


def simhash_hamming_distance(hash1: int, hash2: int) -> int:
    """Compute Hamming distance (number of differing bits) between two SimHashes."""
    return (hash1 ^ hash2).bit_count()


def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """Compute normalized token Jaccard similarity between two texts."""
    tokens1 = set(tokenize_text(text1))
    tokens2 = set(tokenize_text(text2))
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


class ArticleDeduplicator:
    """
    Multi-stage deduplication engine:
    1. Exact URL match via SHA-256 canonical URL hash
    2. Fuzzy match via 64-bit SimHash Hamming distance
    3. Fuzzy match via Token Jaccard similarity across syndication windows
    """

    def __init__(
        self,
        time_window_hours: float = 24.0,
        simhash_max_distance: int = 12,
        jaccard_threshold: float = 0.50,
    ) -> None:
        self.time_window_hours = time_window_hours
        self.simhash_max_distance = simhash_max_distance
        self.jaccard_threshold = jaccard_threshold

        # In-memory index of seen items: {symbol: [(pub_timestamp, url_hash, simhash, token_set, title)]}
        self._seen_url_hashes: Set[str] = set()
        self._symbol_article_index: Dict[str, List[Tuple[float, str, int, Set[str], str]]] = {}

    def is_duplicate(
        self,
        title: str,
        url: str,
        published_at: datetime,
        symbol: str = "GLOBAL",
        summary: str = "",
    ) -> bool:
        """
        Check if an article is a duplicate of any previously seen or indexed article.
        """
        url_hash = compute_url_hash(url)
        if url_hash in self._seen_url_hashes:
            return True

        combined_text = f"{title} {summary}".strip()
        simhash = compute_simhash(combined_text)
        tokens = set(tokenize_text(combined_text))

        pub_ts = (
            published_at.timestamp()
            if published_at.tzinfo
            else published_at.replace(tzinfo=timezone.utc).timestamp()
        )
        window_seconds = self.time_window_hours * 3600.0

        # Check against existing symbol articles within time window
        candidates = self._symbol_article_index.get(symbol, [])
        for ex_ts, ex_url_hash, ex_simhash, ex_tokens, ex_title in candidates:
            # Check time window
            if abs(pub_ts - ex_ts) <= window_seconds:
                # 1. SimHash distance check
                if simhash_hamming_distance(simhash, ex_simhash) <= self.simhash_max_distance:
                    return True

                # 2. Token Jaccard check
                if tokens and ex_tokens:
                    union_len = len(tokens | ex_tokens)
                    if union_len > 0 and (len(tokens & ex_tokens) / union_len) >= self.jaccard_threshold:
                        return True

                # 3. Exact title normalized check
                if title.strip().lower() == ex_title.strip().lower():
                    return True

        return False

    def register(
        self,
        title: str,
        url: str,
        published_at: datetime,
        symbol: str = "GLOBAL",
        summary: str = "",
    ) -> str:
        """
        Register an article in the deduplication cache.
        Returns the SHA-256 URL hash.
        """
        url_hash = compute_url_hash(url)
        self._seen_url_hashes.add(url_hash)

        combined_text = f"{title} {summary}".strip()
        simhash = compute_simhash(combined_text)
        tokens = set(tokenize_text(combined_text))

        pub_ts = (
            published_at.timestamp()
            if published_at.tzinfo
            else published_at.replace(tzinfo=timezone.utc).timestamp()
        )

        if symbol not in self._symbol_article_index:
            self._symbol_article_index[symbol] = []
        self._symbol_article_index[symbol].append((pub_ts, url_hash, simhash, tokens, title))

        return url_hash

    def filter_duplicates(
        self,
        articles: Sequence,
    ) -> List:
        """
        Filter a batch of NewsArticle objects, discarding duplicates and registering unique ones.
        """
        unique_articles = []
        for art in articles:
            if not self.is_duplicate(
                title=art.title,
                url=art.url,
                published_at=art.published_at_utc,
                symbol=art.symbol,
                summary=art.summary,
            ):
                self.register(
                    title=art.title,
                    url=art.url,
                    published_at=art.published_at_utc,
                    symbol=art.symbol,
                    summary=art.summary,
                )
                unique_articles.append(art)
        return unique_articles


class SQLiteNewsStore:
    """
    ACID-compliant SQLite persistence store for raw news articles and sentiment scores.
    """

    def __init__(self, db_path: str = "data_cache/news_sentiment.db") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_articles (
                    article_id TEXT PRIMARY KEY,
                    url_hash TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    market TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT,
                    source TEXT,
                    url TEXT NOT NULL,
                    published_at_utc TEXT NOT NULL,
                    harvested_at_utc TEXT NOT NULL,
                    raw_content_hash TEXT
                );

                CREATE TABLE IF NOT EXISTS sentiment_scores (
                    article_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    published_at_utc TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    sentiment_confidence REAL NOT NULL,
                    sentiment_label TEXT NOT NULL,
                    pos_prob REAL,
                    neu_prob REAL,
                    neg_prob REAL,
                    model_name TEXT NOT NULL,
                    inference_latency_ms REAL,
                    processed_at_utc TEXT NOT NULL,
                    FOREIGN KEY (article_id) REFERENCES raw_articles(article_id)
                );

                CREATE INDEX IF NOT EXISTS idx_art_sym_pub ON raw_articles(symbol, published_at_utc);
                CREATE INDEX IF NOT EXISTS idx_art_pub ON raw_articles(published_at_utc);
                CREATE INDEX IF NOT EXISTS idx_sent_sym_pub ON sentiment_scores(symbol, published_at_utc);
                CREATE INDEX IF NOT EXISTS idx_sent_pub ON sentiment_scores(published_at_utc);
                """
            )

    def store_articles(self, articles: Sequence) -> int:
        """
        Store unique articles into the raw_articles table.
        Skips articles that already exist by url_hash or article_id.
        Returns count of newly inserted articles.
        """
        if not articles:
            return 0

        inserted = 0
        with self._get_connection() as conn:
            for art in articles:
                pub_iso = (
                    art.published_at_utc.isoformat()
                    if isinstance(art.published_at_utc, datetime)
                    else str(art.published_at_utc)
                )
                harv_iso = (
                    art.harvested_at_utc.isoformat()
                    if isinstance(art.harvested_at_utc, datetime)
                    else str(art.harvested_at_utc)
                )
                url_hash = compute_url_hash(art.url)
                try:
                    conn.execute(
                        """
                        INSERT INTO raw_articles (
                            article_id, url_hash, symbol, market, title,
                            summary, source, url, published_at_utc,
                            harvested_at_utc, raw_content_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            art.article_id,
                            url_hash,
                            art.symbol,
                            art.market,
                            art.title,
                            art.summary,
                            art.source,
                            art.url,
                            pub_iso,
                            harv_iso,
                            getattr(art, "raw_content_hash", None),
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    # Duplicate url_hash or article_id, ignore
                    pass
        return inserted

    def store_sentiment_results(self, results: Sequence) -> int:
        """
        Store NLP sentiment scores into sentiment_scores table.
        Returns count of inserted records.
        """
        if not results:
            return 0

        inserted = 0
        with self._get_connection() as conn:
            for res in results:
                pub_iso = (
                    res.published_at_utc.isoformat()
                    if isinstance(res.published_at_utc, datetime)
                    else str(res.published_at_utc)
                )
                proc_iso = (
                    res.processed_at_utc.isoformat()
                    if hasattr(res, "processed_at_utc") and isinstance(res.processed_at_utc, datetime)
                    else datetime.now(timezone.utc).isoformat()
                )
                probs = getattr(res, "probabilities", {}) or {}
                pos_prob = probs.get("positive", 0.0)
                neu_prob = probs.get("neutral", 0.0)
                neg_prob = probs.get("negative", 0.0)

                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO sentiment_scores (
                            article_id, symbol, published_at_utc, sentiment_score,
                            sentiment_confidence, sentiment_label, pos_prob,
                            neu_prob, neg_prob, model_name, inference_latency_ms,
                            processed_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            res.article_id,
                            res.symbol,
                            pub_iso,
                            float(res.sentiment_score),
                            float(res.sentiment_confidence),
                            res.sentiment_label,
                            pos_prob,
                            neu_prob,
                            neg_prob,
                            res.model_name,
                            getattr(res, "inference_latency_ms", 0.0),
                            proc_iso,
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass
        return inserted

    def get_articles(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Query raw articles matching filter parameters."""
        query = "SELECT * FROM raw_articles WHERE 1=1"
        params: List[object] = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_time:
            query += " AND published_at_utc >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND published_at_utc <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY published_at_utc ASC"
        if limit:
            query += f" LIMIT {int(limit)}"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_sentiment_scores(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Query sentiment scores matching filter parameters."""
        query = "SELECT * FROM sentiment_scores WHERE 1=1"
        params: List[object] = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_time:
            query += " AND published_at_utc >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND published_at_utc <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY published_at_utc ASC"
        if limit:
            query += f" LIMIT {int(limit)}"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
