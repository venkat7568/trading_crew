#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_client.py — Class-based news/search with NO company extraction
========================================================================

- Moneycontrol list-page scraper
- Brave News search (optional: BRAVE_API_KEY)
- No symbol or company-name extraction (agents will read headlines directly)
- Compact mode: {headline, date, source, url, summary}

Usage:
  python news_client.py --recent --days 2 --max 12 --compact
  python news_client.py --search "Maruti Suzuki earnings OR upgrade" --days 7 --compact
"""

from __future__ import annotations
import os, re, json, logging, requests, urllib.parse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

# -------------------- env / logging --------------------
try:
    from dotenv import load_dotenv, find_dotenv
    _DOT = find_dotenv(usecwd=True)
    if _DOT:
        load_dotenv(_DOT, override=False)
except Exception:
    pass


def _log(name="news_client_lite", default="WARNING") -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        lg.addHandler(h)
    lg.setLevel(os.environ.get("NEWS_LOG_LEVEL", default).upper())
    return lg


log = _log()

# -------------------- constants --------------------
IST = timezone(timedelta(hours=5, minutes=30))
UA_DEFAULT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HEADERS_HTML = {
    "User-Agent": os.environ.get("NEWS_USER_AGENT", UA_DEFAULT),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
SEARCH_TIMEOUT = float(os.environ.get("SEARCH_TIMEOUT", "10"))
BRAVE_NEWS_ENDPOINT = "https://api.search.brave.com/res/v1/news/search"
BRAVE_API_KEY = (os.environ.get("BRAVE_API_KEY") or "").strip()


# -------------------- helpers --------------------
def _host(u: str) -> str:
    try:
        return urlparse(u).netloc
    except Exception:
        return ""


def _canon(u: str) -> str:
    try:
        p = urllib.parse.urlparse(u or "")
        return urllib.parse.urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
    except Exception:
        return u or ""


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _ist_noon_to_utc_iso(date_yyyy_mm_dd: str) -> str:
    d = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d").replace(
        tzinfo=IST, hour=12, minute=0
    )
    return _iso_utc(d)


def _infer_date_from_url(u: str) -> Optional[str]:
    try:
        parsed = urlparse(u)
        s = (parsed.path or "") + "?" + (parsed.query or "")
        s = s.lower()
        for pat in (
            r"/(20\d{2})/([01]\d)/([0-3]\d)/",
            r"/(20\d{2})-([01]\d)-([0-3]\d)",
            r"(20\d{2})([01]\d)([0-3]\d)",
        ):
            m = re.search(pat, s)
            if m:
                y, mo, d = map(int, m.groups())
                return f"{y:04d}-{mo:02d}-{d:02d}"
    except Exception:
        pass
    return None


def _within_ist(date_str: Optional[str], start_d, end_d) -> bool:
    if not date_str:
        return False
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return start_d <= d <= end_d
    except Exception:
        return False


# -------------------- data model --------------------
@dataclass
class NewsItem:
    headline: str
    url: str
    publisher: str
    published_at: Optional[str] = None      # UTC ISO
    published_local_date: Optional[str] = None
    summary: Optional[str] = None
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------- client --------------------
class NewsClient:
    MONEYCONTROL = "https://www.moneycontrol.com/news/business/stocks/"

    def __init__(self, session: Optional[requests.Session] = None, user_agent: Optional[str] = None):
        self.sess = session or requests.Session()
        self.headers = dict(HEADERS_HTML)
        if user_agent:
            self.headers["User-Agent"] = user_agent

    @staticmethod
    def _clamp(s: Optional[str], n: int = 140) -> Optional[str]:
        if not s:
            return None
        s = " ".join(s.split())
        return s[:n] + ("…" if len(s) > n else "")

    @staticmethod
    def _to_compact(it: NewsItem) -> Dict[str, Any]:
        date_iso = it.published_local_date or _infer_date_from_url(it.url)
        return {
            "headline": it.headline,
            "date": date_iso,
            "source": _host(it.url),
            "url": _canon(it.url),
            "summary": NewsClient._clamp(it.summary),
        }

    @staticmethod
    def _compact(items: List[NewsItem], max_items: int) -> List[Dict[str, Any]]:
        def k(x: NewsItem):
            try:
                return (
                    datetime.fromisoformat(x.published_at)
                    if x.published_at
                    else datetime.min.replace(tzinfo=timezone.utc)
                )
            except Exception:
                return datetime.min.replace(tzinfo=timezone.utc)

        ranked = sorted(items, key=k, reverse=True)[:max_items]
        seen = set()
        rows: List[Dict[str, Any]] = []
        for r in map(NewsClient._to_compact, ranked):
            key = (r.get("url"), (r.get("headline") or "").lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
        return rows

    # ---------- Moneycontrol scraper ----------
    def _scrape_mc(self, url: Optional[str] = None) -> List[NewsItem]:
        r = self.sess.get(url or self.MONEYCONTROL, headers=self.headers, timeout=SEARCH_TIMEOUT)
        r.raise_for_status()
        html = r.text
        out: List[NewsItem] = []

        # Primary pattern: <ul id="cagetory">
        m = re.search(r'<ul\s+id=["\']?cagetory["\']?[^>]*>(.*?)</ul>', html, flags=re.S | re.I)
        if m:
            cont = m.group(1)
            for li in re.finditer(
                r'<li[^>]*id=["\']newslist-(\d+)["\'][^>]*>(.*?)</li>',
                cont,
                flags=re.S | re.I,
            ):
                nid, block = li.group(1), li.group(2)
                if "ads-div-detect" in block:
                    continue
                a = re.search(
                    r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                    block,
                    flags=re.S | re.I,
                )
                if not a:
                    continue
                href, inner = a.group(1), a.group(2)
                if href.startswith("/"):
                    item_url = f"https://www.moneycontrol.com{href}"
                elif href.startswith("http"):
                    item_url = href
                else:
                    item_url = f"https://www.moneycontrol.com/{href}"

                h2 = re.search(r"<h2[^>]*>(.*?)</h2>", inner, flags=re.S | re.I)
                headline = re.sub(r"<.*?>", "", (h2.group(1) if h2 else inner)).strip()
                headline = re.sub(r"\s+", " ", headline)
                if len(headline) < 8:
                    continue

                p = re.search(r"</a>\s*<p[^>]*>(.*?)</p>", block, flags=re.S | re.I)
                summary = re.sub(r"<.*?>", "", p.group(1)).strip() if p else None

                pub_local = _infer_date_from_url(item_url) or datetime.now(IST).date().isoformat()
                pub_at = _ist_noon_to_utc_iso(pub_local)

                out.append(
                    NewsItem(
                        headline=headline,
                        url=_canon(item_url),
                        publisher=_host(item_url),
                        summary=summary,
                        published_at=pub_at,
                        published_local_date=pub_local,
                    )
                )

        # Fallback: looser anchors if the main block wasn't found
        if not out:
            for a in re.finditer(
                r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                html,
                flags=re.S | re.I,
            ):
                href, inner = a.group(1), a.group(2)
                if "/news/" not in href or "/business/stocks/" not in href:
                    continue
                item_url = f"https://www.moneycontrol.com{href}" if href.startswith("/") else href
                h2 = re.search(r"<h2[^>]*>(.*?)</h2>", inner, flags=re.S | re.I)
                headline = re.sub(r"<.*?>", "", (h2.group(1) if h2 else inner))
                headline = re.sub(r"\s+", " ", headline).strip()
                if len(headline) < 8:
                    continue
                pub_local = _infer_date_from_url(item_url) or datetime.now(IST).date().isoformat()
                pub_at = _ist_noon_to_utc_iso(pub_local)
                out.append(
                    NewsItem(
                        headline=headline,
                        url=_canon(item_url),
                        publisher=_host(item_url),
                        published_at=pub_at,
                        published_local_date=pub_local,
                    )
                )

        # de-dup by canonical URL
        seen = set()
        ded: List[NewsItem] = []
        for it in out:
            if it.url in seen:
                continue
            seen.add(it.url)
            ded.append(it)
        return ded

    # ---------- Brave News (optional) ----------
    def search_news(
        self,
        query: str,
        *,
        lookback_days: Optional[int] = None,
        max_results: int = 50,
        sort_by_date: bool = True,
        offset: Optional[int] = None,
        sitedomain: Optional[str] = None,
        compact: bool = False,
        today: Optional[str] = None,
        mode: str = "live",
    ) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"error": "empty_query", "message": "query cannot be empty", "items": []}
        if not BRAVE_API_KEY:
            return {"error": "no_api_key", "message": "BRAVE_API_KEY not set", "items": []}

        params: Dict[str, Any] = {
            "q": q,
            "count": max(5, min(50, int(max_results))),
            "country": "in",
            "safesearch": "moderate",
            "spellcheck": 1,
        }
        if isinstance(lookback_days, int) and lookback_days > 0:
            params["freshness"] = f"{lookback_days}d"
        if sort_by_date:
            params["sort"] = "date"
        if isinstance(offset, int) and offset >= 0:
            params["offset"] = offset
        if sitedomain:
            params["sitedomain"] = sitedomain

        r = self.sess.get(
            BRAVE_NEWS_ENDPOINT,
            params=params,
            headers={
                "Accept": "application/json",
                "User-Agent": self.headers["User-Agent"],
                "X-Subscription-Token": BRAVE_API_KEY,
            },
            timeout=SEARCH_TIMEOUT,
        )
        r.raise_for_status()
        raw = r.json()

        items: List[NewsItem] = []
        for it in (raw.get("results") or []):
            title = (it.get("title") or "").strip()
            url = _canon((it.get("url") or "").strip())
            if not (title and url):
                continue
            snippet = (it.get("description") or "").strip() or None
            publisher = (it.get("meta_url") or {}).get("hostname") or _host(url)

            pf = (it.get("page_fetched") or "").strip()
            published_at = published_local_date = None
            if pf:
                try:
                    dt = (
                        datetime.fromisoformat(pf.replace("Z", "+00:00"))
                        .astimezone(timezone.utc)
                    )
                    published_at = dt.isoformat()
                    published_local_date = dt.astimezone(IST).date().isoformat()
                except Exception:
                    pass
            if not published_local_date:
                d = _infer_date_from_url(url)
                if d:
                    published_local_date = d
                    published_at = _ist_noon_to_utc_iso(d)

            items.append(
                NewsItem(
                    headline=title,
                    url=url,
                    publisher=publisher,
                    summary=snippet,
                    published_at=published_at,
                    published_local_date=published_local_date,
                )
            )

        # sort newest first; keep top max_results
        dated = [x for x in items if x.published_at]
        undated = [x for x in items if not x.published_at]
        dated.sort(key=lambda x: datetime.fromisoformat(x.published_at), reverse=True)
        items = (dated + undated)[:max_results]

        base_date = today or datetime.now(timezone.utc).date().isoformat()
        window_from = (
            (datetime.now(IST) - timedelta(days=lookback_days)).date().isoformat()
            if lookback_days
            else None
        )
        return {
            "date": base_date,
            "mode": mode,
            "query": q,
            "window_ist": {"from": window_from, "to": datetime.now(IST).date().isoformat()},
            "items": (self._compact(items, max_results) if compact else [x.as_dict() for x in items]),
        }

    # ---------- Moneycontrol recent ----------
    def get_recent_news_and_calls(
        self,
        *,
        today: Optional[str] = None,
        lookback_days: int = 2,
        max_items: int = 20,
        mode: str = "live",
        compact: bool = False,
    ) -> List[Dict[str, Any]]:
        base_ist = (
            datetime.strptime(today, "%Y-%m-%d").replace(tzinfo=IST)
            if today
            else datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
        )
        start_d = (base_ist - timedelta(days=max(0, lookback_days))).date()
        end_d = base_ist.date()

        rows = self._scrape_mc(self.MONEYCONTROL)
        rows = [r for r in rows if _within_ist(r.published_local_date, start_d, end_d)]

        if mode.lower() == "backtest":
            cutoff_utc = base_ist.astimezone(timezone.utc).replace(
                hour=23, minute=59, second=59
            )
            rows = [
                r
                for r in rows
                if r.published_at and datetime.fromisoformat(r.published_at) <= cutoff_utc
            ]

        rows.sort(
            key=lambda x: datetime.fromisoformat(x.published_at)
            if x.published_at
            else datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        rows = rows[:max_items]
        return self._compact(rows, max_items) if compact else [r.as_dict() for r in rows]

    @staticmethod
    def to_json(obj: Any, **kw) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2, **kw)


# -------------------- CLI --------------------
def _cli():
    import argparse

    ap = argparse.ArgumentParser(
        description="NewsClient Lite (no symbol/company extraction; agents read headlines directly)"
    )
    ap.add_argument("--scrape-mc", action="store_true")
    ap.add_argument("--recent", action="store_true")
    ap.add_argument("--search")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--max", type=int, default=20)
    ap.add_argument("--today")
    ap.add_argument("--domain")
    ap.add_argument("--offset", type=int)
    ap.add_argument("--mode", default="live")
    ap.add_argument("--compact", action="store_true")
    args = ap.parse_args()

    cli = NewsClient()
    if args.scrape_mc:
        print(NewsClient.to_json([x.as_dict() for x in cli._scrape_mc()]))
        return
    if args.recent:
        print(
            NewsClient.to_json(
                cli.get_recent_news_and_calls(
                    today=args.today,
                    lookback_days=args.days,
                    max_items=args.max,
                    mode=args.mode,
                    compact=args.compact,
                )
            )
        )
        return
    if args.search:
        out = cli.search_news(
            args.search,
            lookback_days=args.days,
            max_results=args.max,
            sitedomain=args.domain,
            offset=args.offset,
            compact=args.compact,
        )
        print(NewsClient.to_json(out))
        return
    ap.print_help()


if __name__ == "__main__":
    _cli()
