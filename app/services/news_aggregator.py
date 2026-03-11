"""Live news aggregator for USD/NGN rate intelligence.

Scrapes free RSS feeds and public web pages — no API keys required.
Uses concurrent fetching for speed, relevance scoring for signal quality,
and multi-layer caching for resilience.

Sources: Google News RSS (8 queries), 7 Nigerian financial RSS feeds,
CBN press releases, OilPrice.com, and IMF news.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NewsItem:
    title: str
    summary: str
    source: str
    published: datetime
    category: str       # naira_forex | cbn_policy | oil | global_macro | nigeria_economy
    url: str
    relevance: float    # 0.0–1.0 relevance to USD/NGN trading


@dataclass(frozen=True)
class SourceStatus:
    name: str
    ok: bool
    item_count: int
    latency_ms: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class NewsDigest:
    items: list[NewsItem]
    fetched_at: datetime
    source_statuses: list[SourceStatus]
    total_raw_items: int = 0
    total_after_dedup: int = 0


# ---------------------------------------------------------------------------
# Feed definitions
# ---------------------------------------------------------------------------

GOOGLE_NEWS_QUERIES: list[tuple[str, str]] = [
    # Direct USD/NGN and naira
    ('"USD NGN" OR "dollar naira" OR "naira dollar" OR "USDT NGN"', "naira_forex"),
    ('"CBN" naira OR "Central Bank of Nigeria" policy', "cbn_policy"),
    ('"Nigeria forex" OR "NAFEM" OR "parallel market" naira', "naira_forex"),
    # Oil — Nigeria's #1 revenue driver
    ('"Brent crude" price OR "oil price" today', "oil"),
    ('OPEC production OR "OPEC+" output cut', "oil"),
    ('"Nigeria oil" OR NNPC OR "oil revenue" Nigeria', "oil"),
    # Global macro affecting EM currencies
    ('"dollar index" OR DXY OR "US dollar" strength', "global_macro"),
    ('"Federal Reserve" rate OR "Fed rate" decision', "global_macro"),
    # Nigeria economy
    ('"Nigeria economy" OR "Nigerian GDP" OR "Nigeria inflation"', "nigeria_economy"),
    ('"Nigeria" "World Bank" OR "IMF" Nigeria OR "monetary policy" Nigeria', "nigeria_economy"),
    # Geopolitical risk affecting oil/EM
    ('"emerging markets" currency OR "EM FX" crisis', "global_macro"),
]

DIRECT_RSS_FEEDS: list[tuple[str, str, str]] = [
    # Nigerian financial media (primary)
    ("https://nairametrics.com/feed/", "Nairametrics", "naira_forex"),
    ("https://businessday.ng/feed/", "BusinessDay NG", "nigeria_economy"),
    ("https://www.vanguardngr.com/category/business/feed/", "Vanguard NG", "nigeria_economy"),
    ("https://www.thisdaylive.com/index.php/category/business/feed/", "ThisDay Live", "nigeria_economy"),
    ("https://www.premiumtimesng.com/business/feed", "Premium Times", "nigeria_economy"),
    ("https://punchng.com/topics/business/feed/", "Punch NG", "nigeria_economy"),
    # Oil and commodities
    ("https://oilprice.com/rss/main", "OilPrice.com", "oil"),
    # International macro
    ("https://www.imf.org/en/News/rss", "IMF News", "global_macro"),
]

# High-relevance keywords with weights for scoring
_RELEVANCE_KEYWORDS: dict[str, float] = {
    # Tier 1: Direct USD/NGN impact (weight 1.0)
    "naira": 1.0, "ngn": 1.0, "cbn": 1.0, "usdt": 1.0, "usd/ngn": 1.0,
    "nafem": 1.0, "exchange rate": 1.0, "parallel market": 1.0,
    "bureau de change": 1.0, "bdc": 1.0, "forex reserve": 1.0,
    "mpc": 0.95, "monetary policy": 0.95, "interest rate": 0.9,
    "devaluation": 1.0, "revaluation": 1.0, "fx intervention": 1.0,
    "cardoso": 0.9, "dollar scarcity": 1.0,
    # Tier 2: Oil/revenue (weight 0.7–0.8)
    "brent": 0.8, "crude oil": 0.8, "oil price": 0.8,
    "opec": 0.75, "nnpc": 0.8, "oil revenue": 0.85,
    "petroleum": 0.7, "barrel": 0.7,
    # Tier 3: Global macro (weight 0.5–0.6)
    "federal reserve": 0.6, "fed rate": 0.65, "dxy": 0.65,
    "dollar index": 0.6, "us dollar": 0.55,
    "inflation": 0.5, "treasury": 0.5, "emerging market": 0.55,
    # Tier 4: Nigeria economy (weight 0.3–0.5)
    "nigeria gdp": 0.5, "nigeria economy": 0.45, "nigeria debt": 0.5,
    "world bank nigeria": 0.4, "imf nigeria": 0.45,
}

# Category keywords for auto-classification
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "cbn_policy": [
        "cbn", "central bank", "monetary policy", "mpc", "cardoso",
        "interest rate", "mpr", "fx intervention", "forex reserve",
        "devaluation", "revaluation",
    ],
    "naira_forex": [
        "naira", "ngn", "forex", "usdt", "dollar", "fx", "exchange rate",
        "black market", "parallel market", "nafem", "bureau de change",
        "bdc", "dollar scarcity",
    ],
    "oil": [
        "brent", "crude", "opec", "nnpc", "oil price", "petroleum",
        "barrel", "oil revenue", "oil output",
    ],
    "global_macro": [
        "fed", "federal reserve", "imf", "world bank", "dxy",
        "dollar index", "inflation", "treasury", "emerging market",
    ],
}


# ---------------------------------------------------------------------------
# Lightweight XML parser (no external dependencies)
# ---------------------------------------------------------------------------

def _parse_rss_xml(xml_text: str) -> list[dict]:
    """Extract items from RSS/Atom XML."""
    items: list[dict] = []
    # RSS <item> blocks
    item_blocks = re.findall(r"<item[^>]*>(.*?)</item>", xml_text, re.DOTALL)
    # Also try Atom <entry> blocks
    if not item_blocks:
        item_blocks = re.findall(r"<entry[^>]*>(.*?)</entry>", xml_text, re.DOTALL)
    for block in item_blocks:
        title = _xml_tag(block, "title")
        link = _xml_tag(block, "link")
        # Atom uses href attribute on <link>
        if not link:
            link_match = re.search(r'<link[^>]+href="([^"]+)"', block)
            if link_match:
                link = link_match.group(1)
        desc = _xml_tag(block, "description") or _xml_tag(block, "summary") or _xml_tag(block, "content")
        pub_date = _xml_tag(block, "pubDate") or _xml_tag(block, "published") or _xml_tag(block, "updated")
        items.append({
            "title": _strip_html(title),
            "link": link.strip(),
            "description": _strip_html(desc),
            "pub_date": pub_date,
        })
    return items


def _xml_tag(text: str, tag: str) -> str:
    """Extract content of the first occurrence of <tag>...</tag>."""
    # Handle CDATA
    pattern = rf"<{tag}[^>]*>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&apos;", "'")
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_date(date_str: str) -> datetime | None:
    """Best-effort parse of RSS date strings."""
    if not date_str:
        return None
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%a, %d %b %Y %H:%M:%S",
        "%d %b %Y %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
    ]
    cleaned = date_str.strip()
    # Normalise timezone abbreviations
    for abbr, offset in [("GMT", "+0000"), ("UTC", "+0000"), ("EST", "-0500"),
                          ("EDT", "-0400"), ("WAT", "+0100"), ("CST", "-0600"),
                          ("CDT", "-0500"), ("PST", "-0800"), ("PDT", "-0700")]:
        cleaned = re.sub(rf"\s+{abbr}$", f" {offset}", cleaned)
    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

def _compute_relevance(title: str, summary: str) -> float:
    """Score 0.0–1.0 indicating how relevant this headline is to USD/NGN trading."""
    text = (title + " " + summary).lower()
    max_score = 0.0
    hit_count = 0
    for keyword, weight in _RELEVANCE_KEYWORDS.items():
        if keyword in text:
            max_score = max(max_score, weight)
            hit_count += 1
    # Combine: highest single keyword weight + bonus for multiple hits
    if hit_count == 0:
        return 0.1  # baseline for any news
    bonus = min(0.2, hit_count * 0.03)
    return min(1.0, max_score + bonus)


# ---------------------------------------------------------------------------
# Auto-categorise
# ---------------------------------------------------------------------------

def _auto_categorise(title: str, summary: str, default: str) -> str:
    text = (title + " " + summary).lower()
    best_cat = default
    best_score = 0
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _title_fingerprint(title: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def _jaccard(a: str, b: str) -> float:
    sa = set(re.sub(r"[^a-z0-9 ]", "", a.lower()).split())
    sb = set(re.sub(r"[^a-z0-9 ]", "", b.lower()).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _deduplicate(items: list[NewsItem]) -> list[NewsItem]:
    """Remove near-duplicate headlines (Jaccard > 0.55)."""
    seen: set[str] = set()
    kept: list[NewsItem] = []
    for item in items:
        fp = _title_fingerprint(item.title)
        if fp in seen:
            continue
        is_dup = any(_jaccard(item.title, ex.title) > 0.55 for ex in kept[-30:])
        if not is_dup:
            seen.add(fp)
            kept.append(item)
    return kept


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class NewsAggregatorService:
    MAX_ITEMS_PER_FEED = 10
    MAX_TOTAL_ITEMS = 60

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: NewsDigest | None = None
        self._cache_time: datetime | None = None
        self._disk_cache_path = settings.runtime_dir / "news_cache.json"

    @property
    def _cache_ttl(self) -> int:
        return getattr(self.settings, "news_cache_ttl_seconds", 900)

    @property
    def _max_age_hours(self) -> int:
        return getattr(self.settings, "news_max_age_hours", 72)

    @property
    def _fetch_timeout(self) -> float:
        return getattr(self.settings, "news_fetch_timeout_seconds", 10.0)

    # -- public API ----------------------------------------------------------

    def fetch(self) -> NewsDigest:
        """Return a NewsDigest, using in-memory cache if fresh."""
        if self._cache and self._cache_time:
            age = (datetime.now(UTC) - self._cache_time).total_seconds()
            if age < self._cache_ttl:
                return self._cache

        try:
            items, statuses, total_raw = self._fetch_all_concurrent()
        except Exception as exc:
            logger.warning("News fetch failed, trying disk cache: %s", exc)
            disk = self._load_disk_cache()
            if disk:
                return disk
            return NewsDigest(
                items=[], fetched_at=datetime.now(UTC),
                source_statuses=[SourceStatus(name="all", ok=False, item_count=0, error=str(exc)[:120])],
            )

        items = _deduplicate(items)
        total_after_dedup = len(items)

        # Sort by: relevance (desc) then recency (desc)
        items.sort(key=lambda x: (x.relevance, x.published.timestamp()), reverse=True)
        items = items[: self.MAX_TOTAL_ITEMS]

        digest = NewsDigest(
            items=items,
            fetched_at=datetime.now(UTC),
            source_statuses=statuses,
            total_raw_items=total_raw,
            total_after_dedup=total_after_dedup,
        )
        self._cache = digest
        self._cache_time = datetime.now(UTC)
        self._save_disk_cache(digest)
        return digest

    # -- concurrent fetcher --------------------------------------------------

    def _fetch_all_concurrent(self) -> tuple[list[NewsItem], list[SourceStatus], int]:
        """Fetch all sources concurrently using a thread pool."""
        all_items: list[NewsItem] = []
        statuses: list[SourceStatus] = []
        cutoff = datetime.now(UTC) - timedelta(hours=self._max_age_hours)

        # Build task list: (callable, args)
        tasks: list[tuple[str, str, str, str]] = []

        # Google News queries
        for query, category in GOOGLE_NEWS_QUERIES:
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query, safe='')}&hl=en-NG&gl=NG&ceid=NG:en"
            name = f"Google News: {query[:45]}"
            tasks.append(("rss", url, name, category))

        # Direct RSS feeds
        for url, source_name, category in DIRECT_RSS_FEEDS:
            tasks.append(("rss", url, source_name, category))

        # CBN press releases
        tasks.append(("cbn", "https://www.cbn.gov.ng/Press/", "CBN Press Releases", "cbn_policy"))

        def _do_fetch(task_type: str, url: str, name: str, category: str) -> tuple[list[NewsItem], SourceStatus]:
            import time as _time
            t0 = _time.monotonic()
            try:
                with httpx.Client(
                    timeout=self._fetch_timeout,
                    headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
                    follow_redirects=True,
                ) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    elapsed = (_time.monotonic() - t0) * 1000

                    if task_type == "cbn":
                        items = self._parse_cbn_html(resp.text, cutoff)
                    else:
                        raw = _parse_rss_xml(resp.text)
                        items = self._raw_to_news_items(raw, name, category, cutoff)

                    items = items[: self.MAX_ITEMS_PER_FEED]
                    return items, SourceStatus(name=name, ok=True, item_count=len(items), latency_ms=round(elapsed))
            except Exception as exc:
                elapsed = (_time.monotonic() - t0) * 1000
                return [], SourceStatus(name=name, ok=False, item_count=0, latency_ms=round(elapsed), error=str(exc)[:120])

        # Run all fetches concurrently
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {
                pool.submit(_do_fetch, *task): task[2]
                for task in tasks
            }
            for future in as_completed(futures):
                items, status = future.result()
                all_items.extend(items)
                statuses.append(status)

        total_raw = len(all_items)
        return all_items, statuses, total_raw

    # -- parsers -------------------------------------------------------------

    def _raw_to_news_items(
        self, raw: list[dict], source: str, default_category: str, cutoff: datetime
    ) -> list[NewsItem]:
        items: list[NewsItem] = []
        for entry in raw:
            title = entry.get("title", "").strip()
            if not title or len(title) < 10:
                continue
            summary = (entry.get("description", "") or "")[:300]
            pub = _parse_date(entry.get("pub_date", ""))
            if pub is None:
                pub = datetime.now(UTC)
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=UTC)
            if pub < cutoff:
                continue
            category = _auto_categorise(title, summary, default_category)
            relevance = _compute_relevance(title, summary)
            items.append(NewsItem(
                title=title,
                summary=summary,
                source=source,
                published=pub,
                category=category,
                url=entry.get("link", ""),
                relevance=relevance,
            ))
        return items

    def _parse_cbn_html(self, html: str, cutoff: datetime) -> list[NewsItem]:
        """Extract press release titles from CBN website HTML."""
        items: list[NewsItem] = []
        links = re.findall(r'<a[^>]+href="([^"]*)"[^>]*>([^<]+)</a>', html, re.IGNORECASE)
        cbn_keywords = [
            "communique", "press release", "monetary policy", "exchange rate",
            "intervention", "circular", "guidelines", "naira", "forex",
            "interest rate", "inflation", "mpc", "reserve", "mpr",
        ]
        for href, text in links:
            text = _strip_html(text).strip()
            if len(text) < 15:
                continue
            text_lower = text.lower()
            if any(kw in text_lower for kw in cbn_keywords):
                url = href if href.startswith("http") else f"https://www.cbn.gov.ng{href}"
                relevance = _compute_relevance(text, "")
                items.append(NewsItem(
                    title=text, summary="", source="CBN",
                    published=datetime.now(UTC),
                    category="cbn_policy", url=url, relevance=max(relevance, 0.85),
                ))
        return items

    # -- disk cache ----------------------------------------------------------

    def _save_disk_cache(self, digest: NewsDigest) -> None:
        try:
            data = {
                "fetched_at": digest.fetched_at.isoformat(),
                "total_raw_items": digest.total_raw_items,
                "total_after_dedup": digest.total_after_dedup,
                "items": [
                    {
                        "title": it.title, "summary": it.summary,
                        "source": it.source,
                        "published": it.published.isoformat(),
                        "category": it.category, "url": it.url,
                        "relevance": it.relevance,
                    }
                    for it in digest.items
                ],
                "source_statuses": [
                    {"name": s.name, "ok": s.ok, "item_count": s.item_count,
                     "latency_ms": s.latency_ms, "error": s.error}
                    for s in digest.source_statuses
                ],
            }
            self._disk_cache_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_disk_cache(self) -> NewsDigest | None:
        """Load cached news from disk (fallback when live fetch fails)."""
        try:
            if not self._disk_cache_path.exists():
                return None
            data = json.loads(self._disk_cache_path.read_text())
            fetched_at = datetime.fromisoformat(data["fetched_at"])
            # Don't use disk cache older than 2 hours
            if (datetime.now(UTC) - fetched_at).total_seconds() > 7200:
                return None
            items = [
                NewsItem(
                    title=r["title"], summary=r["summary"], source=r["source"],
                    published=datetime.fromisoformat(r["published"]),
                    category=r["category"], url=r["url"],
                    relevance=r.get("relevance", 0.5),
                )
                for r in data["items"]
            ]
            statuses = [
                SourceStatus(
                    name=s["name"], ok=s["ok"], item_count=s["item_count"],
                    latency_ms=s.get("latency_ms", 0), error=s.get("error", ""),
                )
                for s in data.get("source_statuses", [])
            ]
            return NewsDigest(
                items=items, fetched_at=fetched_at, source_statuses=statuses,
                total_raw_items=data.get("total_raw_items", len(items)),
                total_after_dedup=data.get("total_after_dedup", len(items)),
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Formatting for Gemini prompt — structured for consistency
# ---------------------------------------------------------------------------

def format_news_for_prompt(digest: NewsDigest, max_items: int = 25) -> str:
    """Convert a NewsDigest into a structured text block for the AI prompt.

    Each headline is numbered and tagged so the AI can reference them by ID.
    This prevents hallucination — the AI can only cite what it actually sees.
    """
    if not digest.items:
        return ""

    lines: list[str] = []
    # Header with metadata
    ok_sources = sum(1 for s in digest.source_statuses if s.ok)
    total_sources = len(digest.source_statuses)
    lines.append(
        f"[{len(digest.items)} headlines from {ok_sources}/{total_sources} live sources, "
        f"fetched {digest.fetched_at.strftime('%Y-%m-%d %H:%M UTC')}]"
    )
    lines.append("")

    # Group by category for structured reading
    categories_order = ["cbn_policy", "naira_forex", "oil", "global_macro", "nigeria_economy"]
    category_labels = {
        "cbn_policy": "CBN / MONETARY POLICY",
        "naira_forex": "NAIRA / FOREX",
        "oil": "OIL / COMMODITIES",
        "global_macro": "GLOBAL MACRO",
        "nigeria_economy": "NIGERIA ECONOMY",
    }

    items_by_cat: dict[str, list[NewsItem]] = {c: [] for c in categories_order}
    for item in digest.items[:max_items]:
        cat = item.category if item.category in items_by_cat else "nigeria_economy"
        items_by_cat[cat].append(item)

    headline_id = 0
    for cat in categories_order:
        cat_items = items_by_cat[cat]
        if not cat_items:
            continue
        lines.append(f"--- {category_labels.get(cat, cat.upper())} ---")
        for item in cat_items:
            headline_id += 1
            ts = item.published.strftime("%b %d %H:%M")
            rel_tag = f"[rel={item.relevance:.1f}]" if item.relevance >= 0.7 else ""
            lines.append(f"  H{headline_id}. [{ts} | {item.source}] {item.title} {rel_tag}")
            if item.summary:
                lines.append(f"      {item.summary[:200]}")
        lines.append("")

    return "\n".join(lines)


def format_news_summary_stats(digest: NewsDigest) -> dict:
    """Return summary statistics for the transparency panel."""
    if not digest.items:
        return {"total": 0, "by_category": {}, "sources_ok": 0, "sources_total": 0, "high_relevance": 0}

    by_cat: dict[str, int] = {}
    high_rel = 0
    for item in digest.items:
        by_cat[item.category] = by_cat.get(item.category, 0) + 1
        if item.relevance >= 0.7:
            high_rel += 1

    return {
        "total": len(digest.items),
        "by_category": by_cat,
        "sources_ok": sum(1 for s in digest.source_statuses if s.ok),
        "sources_total": len(digest.source_statuses),
        "sources_failed": [s.name for s in digest.source_statuses if not s.ok],
        "high_relevance": high_rel,
        "fetched_at": digest.fetched_at.isoformat(),
        "total_raw_before_dedup": digest.total_raw_items,
    }
