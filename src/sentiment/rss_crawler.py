"""
Multi-Source RSS News Crawler for Financial Equities.

Polls Google News RSS, Yahoo Finance RSS, and specialized financial feeds
with ticker-specific query formulation for US and Moroccan (Casablanca BVC) stocks.
Includes multi-threaded concurrent fetching, robust XML/Atom parsing, and error recovery.
"""

from __future__ import annotations

import email.utils
import hashlib
import html
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from src.sentiment.deduplicator import canonicalize_url, compute_url_hash

logger = logging.getLogger(__name__)

# Default US and Moroccan Universe Company Mappings
US_COMPANY_MAP: Dict[str, str] = {
    "NVDA": "NVIDIA Corporation",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms",
    "GOOGL": "Alphabet Inc.",
    "TSLA": "Tesla Inc.",
    "AMD": "Advanced Micro Devices",
    "AVGO": "Broadcom Inc.",
    "PLTR": "Palantir Technologies",
    "SMCI": "Super Micro Computer",
    "NFLX": "Netflix Inc.",
    "CRM": "Salesforce Inc.",
    "INTC": "Intel Corporation",
    "PANW": "Palo Alto Networks",
    "CRWD": "CrowdStrike Holdings",
    "NOW": "ServiceNow Inc.",
    "UBER": "Uber Technologies",
    "ABNB": "Airbnb Inc.",
    "COIN": "Coinbase Global",
    "SPY": "SPDR S&P 500 ETF Trust",
}

BVC_COMPANY_MAP: Dict[str, str] = {
    "ATW.BVC": "Attijariwafa Bank",
    "BCP.BVC": "Banque Centrale Populaire",
    "IAM.BVC": "Maroc Telecom",
    "BOA.BVC": "Bank of Africa",
    "CSR.BVC": "Cosumar",
    "LBV.BVC": "Label Vie",
    "HPS.BVC": "Hightech Payment Systems",
    "TQM.BVC": "TotalEnergies Marketing Maroc",
    "TGCC.BVC": "TGCC",
    "MSA.BVC": "Marsa Maroc",
    "CIM.BVC": "Ciments du Maroc",
    "ADH.BVC": "Douja Promotion Groupe Addoha",
}


@dataclass
class NewsArticle:
    """Standardized Financial News Article Record."""

    article_id: str
    symbol: str
    market: str  # "US" or "BVC"
    title: str
    summary: str
    source: str
    url: str
    published_at_utc: datetime
    harvested_at_utc: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    raw_content_hash: Optional[str] = None

    @classmethod
    def create(
        cls,
        symbol: str,
        market: str,
        title: str,
        summary: str,
        source: str,
        url: str,
        published_at_utc: datetime,
        harvested_at_utc: Optional[datetime] = None,
    ) -> NewsArticle:
        """Create a NewsArticle with auto-generated SHA-256 article_id."""
        pub_dt = (
            published_at_utc
            if published_at_utc.tzinfo
            else published_at_utc.replace(tzinfo=timezone.utc)
        )
        harv_dt = (
            harvested_at_utc
            if harvested_at_utc is not None
            else datetime.now(timezone.utc)
        )
        canon_url = canonicalize_url(url)
        # Compute article_id as SHA256 of canonical URL and publication timestamp
        id_material = f"{canon_url}_{pub_dt.isoformat()}"
        article_id = hashlib.sha256(id_material.encode("utf-8")).hexdigest()
        raw_hash = hashlib.sha256(f"{title} {summary}".encode("utf-8")).hexdigest()

        return cls(
            article_id=article_id,
            symbol=symbol.upper(),
            market=market.upper(),
            title=title.strip(),
            summary=summary.strip(),
            source=source.strip(),
            url=canon_url,
            published_at_utc=pub_dt,
            harvested_at_utc=harv_dt,
            raw_content_hash=raw_hash,
        )


def clean_html_text(raw_html: str) -> str:
    """Remove HTML tags, decode HTML entities, and collapse whitespace."""
    if not raw_html:
        return ""
    # Strip HTML tags
    clean = re.sub(r"<[^>]+>", " ", raw_html)
    # Unescape HTML entities (e.g. &amp;, &quot;, &#39;)
    clean = html.unescape(clean)
    # Collapse multiple whitespaces
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def parse_rfc822_or_iso_date(date_str: str) -> datetime:
    """
    Parse RFC 822 / RFC 2822 / ISO 8601 date strings into timezone-aware UTC datetime.
    """
    if not date_str:
        return datetime.now(timezone.utc)
    date_str = date_str.strip()

    # Try RFC 2822 email format (standard for RSS pubDate)
    try:
        parsed_tuple = email.utils.parsedate_tz(date_str)
        if parsed_tuple is not None:
            timestamp = email.utils.mktime_tz(parsed_tuple)
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except Exception:
        pass

    # Try ISO 8601 / common date format patterns
    iso_formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
    ]
    for fmt in iso_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except ValueError:
            continue

    logger.warning("Failed to parse date string '%s', using current UTC time", date_str)
    return datetime.now(timezone.utc)


class RSSNewsCrawler:
    """
    Asynchronous and multi-threaded RSS news harvester.
    Polls Google News RSS, Yahoo Finance RSS, and public financial feeds.
    """

    USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        request_timeout: float = 8.0,
        max_retries: int = 3,
        user_agent: Optional[str] = None,
        custom_fetcher: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.user_agent = user_agent or self.USER_AGENT
        self._custom_fetcher = custom_fetcher

    def build_google_news_url(self, symbol: str, market: str = "US") -> str:
        """Construct Google News RSS search URL tailored by symbol and market."""
        sym_clean = symbol.upper()
        if market.upper() == "BVC" or sym_clean.endswith(".BVC") or sym_clean in BVC_COMPANY_MAP:
            company_name = BVC_COMPANY_MAP.get(sym_clean, sym_clean.replace(".BVC", ""))
            query = f"{company_name} Bourse Casablanca OR {sym_clean.replace('.BVC', '')}"
            encoded_query = urllib.parse.quote(query)
            return f"https://news.google.com/rss/search?q={encoded_query}&hl=fr&gl=MA&ceid=MA:fr"
        else:
            company_name = US_COMPANY_MAP.get(sym_clean, sym_clean)
            query = f"{sym_clean} stock OR {company_name}"
            encoded_query = urllib.parse.quote(query)
            return f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    def build_yahoo_finance_url(self, symbol: str) -> str:
        """Construct Yahoo Finance RSS headline feed URL."""
        sym_clean = symbol.upper().replace(".BVC", "")
        encoded_sym = urllib.parse.quote(sym_clean)
        return f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={encoded_sym}&region=US&lang=en-US"

    def _fetch_xml_content(self, url: str) -> Optional[str]:
        """Fetch raw XML content with timeout and retry logic."""
        if self._custom_fetcher is not None:
            try:
                return self._custom_fetcher(url)
            except Exception as e:
                logger.warning("Custom fetcher failed for URL %s: %s", url, e)
                return None

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
            },
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                    if response.status == 200:
                        content_bytes = response.read()
                        charset = response.headers.get_content_charset() or "utf-8"
                        return content_bytes.decode(charset, errors="replace")
                    else:
                        logger.warning(
                            "HTTP status %d fetching %s (attempt %d/%d)",
                            response.status,
                            url,
                            attempt,
                            self.max_retries,
                        )
            except Exception as e:
                logger.debug(
                    "Network error on attempt %d/%d for %s: %s",
                    attempt,
                    self.max_retries,
                    url,
                    e,
                )

        return None

    def parse_rss_xml(
        self,
        xml_text: str,
        symbol: str,
        market: str = "US",
        source_name: str = "RSS",
        max_items: int = 50,
    ) -> List[NewsArticle]:
        """
        Parse RSS 2.0 or Atom XML string into a list of NewsArticle objects.
        """
        if not xml_text or not xml_text.strip():
            return []

        articles: List[NewsArticle] = []
        try:
            root = ET.fromstring(xml_text.strip())
        except ET.ParseError as e:
            logger.warning("XML parsing error for symbol %s: %s", symbol, e)
            return []

        harvested_now = datetime.now(timezone.utc)

        # 1. RSS 2.0 Channel / Item Format
        channel = root.find("channel")
        items = channel.findall("item") if channel is not None else root.findall(".//item")

        if len(items) > 0:
            for item in items[:max_items]:
                title_elem = item.find("title")
                link_elem = item.find("link")
                desc_elem = item.find("description")
                pub_elem = item.find("pubDate")
                source_elem = item.find("source")

                title = clean_html_text(title_elem.text if title_elem is not None and title_elem.text else "")
                link = (link_elem.text if link_elem is not None and link_elem.text else "").strip()
                summary = clean_html_text(desc_elem.text if desc_elem is not None and desc_elem.text else "")
                pub_str = pub_elem.text if pub_elem is not None and pub_elem.text else ""
                src = (
                    clean_html_text(source_elem.text if source_elem is not None and source_elem.text else "")
                    or source_name
                )

                if not title or not link:
                    continue

                pub_dt = parse_rfc822_or_iso_date(pub_str)
                article = NewsArticle.create(
                    symbol=symbol,
                    market=market,
                    title=title,
                    summary=summary,
                    source=src,
                    url=link,
                    published_at_utc=pub_dt,
                    harvested_at_utc=harvested_now,
                )
                articles.append(article)

            return articles

        # 2. Atom Feed / Entry Format
        entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
        if len(entries) == 0:
            entries = root.findall(".//entry")

        for entry in entries[:max_items]:
            title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
            if title_elem is None:
                title_elem = entry.find("title")

            link_elem = entry.find("{http://www.w3.org/2005/Atom}link")
            if link_elem is None:
                link_elem = entry.find("link")

            summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
            if summary_elem is None:
                summary_elem = entry.find("summary")
            if summary_elem is None:
                summary_elem = entry.find("{http://www.w3.org/2005/Atom}content")
            if summary_elem is None:
                summary_elem = entry.find("content")

            pub_elem = entry.find("{http://www.w3.org/2005/Atom}published")
            if pub_elem is None:
                pub_elem = entry.find("published")
            if pub_elem is None:
                pub_elem = entry.find("{http://www.w3.org/2005/Atom}updated")
            if pub_elem is None:
                pub_elem = entry.find("updated")

            title = clean_html_text(title_elem.text if title_elem is not None and title_elem.text else "")
            link = ""
            if link_elem is not None:
                link = link_elem.attrib.get("href", "") or (link_elem.text or "").strip()
            summary = clean_html_text(summary_elem.text if summary_elem is not None and summary_elem.text else "")
            pub_str = pub_elem.text if pub_elem is not None and pub_elem.text else ""

            if not title or not link:
                continue

            pub_dt = parse_rfc822_or_iso_date(pub_str)
            article = NewsArticle.create(
                symbol=symbol,
                market=market,
                title=title,
                summary=summary,
                source=source_name,
                url=link,
                published_at_utc=pub_dt,
                harvested_at_utc=harvested_now,
            )
            articles.append(article)

        return articles

    def fetch_symbol_news(
        self,
        symbol: str,
        market: Optional[str] = None,
        max_items: int = 50,
        include_google: bool = True,
        include_yahoo: bool = True,
    ) -> List[NewsArticle]:
        """
        Fetch news articles for a single equity symbol from active RSS endpoints.
        """
        sym_clean = symbol.upper()
        if market is None:
            market = "BVC" if (sym_clean.endswith(".BVC") or sym_clean in BVC_COMPANY_MAP) else "US"

        articles: List[NewsArticle] = []

        # 1. Google News Feed
        if include_google:
            g_url = self.build_google_news_url(sym_clean, market)
            g_xml = self._fetch_xml_content(g_url)
            if g_xml:
                g_articles = self.parse_rss_xml(
                    g_xml,
                    symbol=sym_clean,
                    market=market,
                    source_name="GoogleNews",
                    max_items=max_items,
                )
                articles.extend(g_articles)

        # 2. Yahoo Finance Feed (primarily US or global)
        if include_yahoo and market.upper() == "US":
            y_url = self.build_yahoo_finance_url(sym_clean)
            y_xml = self._fetch_xml_content(y_url)
            if y_xml:
                y_articles = self.parse_rss_xml(
                    y_xml,
                    symbol=sym_clean,
                    market=market,
                    source_name="YahooFinance",
                    max_items=max_items,
                )
                articles.extend(y_articles)

        return articles

    def fetch_universe_news(
        self,
        symbols: Optional[Sequence[str]] = None,
        market_map: Optional[Dict[str, str]] = None,
        max_workers: int = 8,
        max_items_per_symbol: int = 50,
    ) -> Dict[str, List[NewsArticle]]:
        """
        Fetch news concurrently for a universe of symbols across multiple threads.
        """
        if symbols is None:
            symbols = list(US_COMPANY_MAP.keys()) + list(BVC_COMPANY_MAP.keys())

        results: Dict[str, List[NewsArticle]] = {}
        if not symbols:
            return results

        def _fetch_single(sym: str) -> Tuple[str, List[NewsArticle]]:
            mkt = None
            if market_map and sym in market_map:
                mkt = market_map[sym]
            arts = self.fetch_symbol_news(
                symbol=sym,
                market=mkt,
                max_items=max_items_per_symbol,
            )
            return sym, arts

        with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as executor:
            future_to_sym = {executor.submit(_fetch_single, sym): sym for sym in symbols}
            for future in as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    sym_res, arts = future.result()
                    results[sym_res] = arts
                except Exception as e:
                    logger.warning("Error fetching news for symbol %s: %s", sym, e)
                    results[sym] = []

        return results
