#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upstox_technical.py — minimal CLI (symbol, days) + 30m candles + indicators
===========================================================================

Inputs:
  --symbol  (company name / trading symbol / ISIN / instrument_key)
  --days    (lookback, default 30)

Fixed:
  interval = 30minute
  returns: current price, merged OHLC (historical + today's intraday),
           indicators (SMA20/50, EMA20/50, RSI14, MACD, ATR14, VWAP-today)
  candles shaped as labeled dicts:
      {timestamp, open, high, low, close, volume, oi}

Resolver (automatic, no flags):
  1) Use local ./complete.json if present (zero network)
  2) Else download Upstox instruments (NSE.json.gz preferred)
  3) If still unresolved, try Wikipedia→ISIN discovery, then validate ISIN
     against Upstox instruments. Never trust web result without validation.

Environment (optional):
  UPSTOX_API_BASE  (default https://api.upstox.com)
  UPSTOX_ACCESS_TOKEN
  REQUESTS_CA_BUNDLE or ALLOW_INSECURE_SSL=true (TLS issues)
  UPSTOX_CACHE_DIR (default ./.cache_upstox)
  UPSTOX_INSTR_MAX_AGE_H (default 24)
  UPSTOX_NSE_ONLY=1 (prefer NSE.json.gz) / 0 (complete.json.gz)
  TZ=Asia/Kolkata
"""

from __future__ import annotations
import os, io, json, time, gzip, logging, pathlib, datetime as dt, csv, re
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set
from zoneinfo import ZoneInfo

# ---------- dotenv (optional) ----------
try:
    from dotenv import load_dotenv, find_dotenv
    _DOTENV = find_dotenv(usecwd=True)
    if _DOTENV:
        load_dotenv(_DOTENV, override=False)
except Exception:
    pass

# ---------- requests ----------
try:
    import requests
    _HAS_REQ = True
except Exception:
    _HAS_REQ = False


# ---------- utils ----------
def _mk_logger(name: str, level_env: str = "TECH_LOG_LEVEL", default_level: str = "WARNING") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    log.setLevel(os.environ.get(level_env, default_level).upper())
    return log

def _round(x: Optional[float], n: int = 2) -> Optional[float]:
    try:
        return round(float(x), n)
    except Exception:
        return None

def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in (s or "") if ch.isalnum())

_STOPWORDS = {
    "limited","ltd","industries","industry","company","co","india","plc","pvt","private","public",
    "services","service","bank","finance","financial","infrastructure","infra","energy","enterprises",
    "enterprise","technologies","technology","tech","pharma","pharmaceuticals","pharmaceutical","motor",
    "motors","pass","veh","vehicle","vehicles","life","general","insurance","steel","cement","auto","autos",
    "systems","system"
}
def _ntoks(s: str) -> List[str]:
    raw = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in (s or ""))
    toks = [t for t in raw.split() if t]
    toks = [t for t in toks if t not in _STOPWORDS]
    return toks

def _ed1(a: str, b: str) -> bool:
    if a == b: return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1: return False
    if la > lb: a, b = b, a; la, lb = lb, la
    i = j = diff = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1; j += 1
        else:
            diff += 1
            if diff > 1: return False
            if la == lb: i += 1; j += 1
            else:       j += 1
    if j < lb or i < la:
        diff += 1
    return diff <= 1


# ---------- instrument cache ----------
class InstrumentCache:
    DEFAULT_COMPLETE = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
    DEFAULT_NSE      = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

    def __init__(self, log: logging.Logger, cache_dir: Optional[str] = None, verify_tls: Optional[bool] = None):
        self.log = log
        self.verify_tls = True if verify_tls is None else bool(verify_tls)

        root = pathlib.Path(cache_dir or (os.environ.get("UPSTOX_CACHE_DIR") or ".cache_upstox"))
        root.mkdir(parents=True, exist_ok=True)
        self.cache_json = root / "instruments.json"
        self.local_complete = pathlib.Path(os.getcwd()) / "complete.json"
        self.rows: List[Dict[str, Any]] = []

        self._by_isin: Dict[str, Dict[str, Any]] = {}
        self._by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        self._by_token: Dict[str, List[Dict[str, Any]]] = {}

    def ensure(self, max_age_h: int = 24, prefer_nse_only: bool = True) -> None:
        if self.local_complete.exists():
            self.log.info("Using local instruments: %s", self.local_complete)
            txt = self.local_complete.read_text(encoding="utf-8")
            self._load_index(txt); return

        need = (not self.cache_json.exists()) or self._is_stale(self.cache_json, max_age_h)
        if need:
            url = os.environ.get("UPSTOX_INSTRUMENTS_URL") or (self.DEFAULT_NSE if prefer_nse_only else self.DEFAULT_COMPLETE)
            self._download(url)
        self._load_index(self.cache_json.read_text(encoding="utf-8"))

    def search(self, query: str, k: int = 15) -> List[Dict[str, Any]]:
        qn = _norm(query)
        toks = _ntoks(query)
        cands: List[Tuple[int, Dict[str, Any]]] = []

        # Exact ISIN / symbol
        hit = self._by_isin.get(qn)
        if hit: cands.append((1000, hit))
        for r in self._by_symbol.get(qn, []):
            cands.append((900, r))

        # Token bucket + fuzzy
        bucket: Set[int] = set()
        seeds: List[Dict[str, Any]] = []
        for t in toks:
            for r in self._by_token.get(t, []):
                rid = id(r)
                if rid not in bucket:
                    bucket.add(rid); seeds.append(r)

        if len(seeds) < 80:
            for key in list(self._by_token.keys()):
                if any(_ed1(t, key) for t in toks):
                    for r in self._by_token.get(key, []):
                        rid = id(r)
                        if rid not in bucket:
                            bucket.add(rid); seeds.append(r)

        for r in seeds:
            score = 0
            ik = r.get("instrument_key") or ""
            seg = (r.get("segment") or "").upper()
            ex  = (r.get("exchange") or "").upper()
            if ik.startswith("NSE_EQ|"): score += 50
            if ex in {"NSE","BSE"} and "EQ" in seg: score += 30
            name = r.get("name") or ""
            nname = _norm(name)
            sym = _norm(r.get("symbol") or "")
            if qn in nname: score += 20
            if qn in sym:   score += 20
            for t in toks:
                if t in _ntoks(name) or t in sym:
                    score += 15
            cands.append((score, r))

        # India equities and indices
        filt = []
        for s, r in cands:
            ex = (r.get("exchange") or "").upper()
            seg = (r.get("segment") or "").upper()
            ik = r.get("instrument_key", "")
            # Include: NSE/BSE equities OR NSE indices (for NIFTY, BANKNIFTY, etc.)
            is_equity = ex in {"NSE","BSE"} and ("EQ" in seg or ik.startswith(("NSE_EQ|","BSE_EQ|")))
            is_index = ex == "NSE" and ("INDEX" in seg or ik.startswith("NSE_INDEX|"))
            if is_equity or is_index:
                filt.append((s, r))

        seen = set(); out: List[Dict[str, Any]] = []
        for s, r in sorted(filt, key=lambda x: -x[0]):
            ik = r.get("instrument_key")
            if ik and ik not in seen:
                seen.add(ik); out.append(r)
            if len(out) >= k: break
        return out

    def resolve(self, query: str) -> Optional[Dict[str, Any]]:
        # instrument_key directly
        if "|" in query:
            return {"instrument_key": query, "symbol": None, "name": None}
        c = self.search(query, k=15)
        if not c: return None

        # For index queries (NIFTY, BANKNIFTY, etc.), prioritize INDEX instruments
        query_upper = query.upper()
        is_index_query = any(idx in query_upper for idx in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"])

        if is_index_query:
            # Try INDEX first for index queries
            for r in c:
                if (r.get("instrument_key") or "").startswith("NSE_INDEX|"):
                    return r

        # Default priority: NSE_EQ, then BSE_EQ, then first result
        for r in c:
            if (r.get("instrument_key") or "").startswith("NSE_EQ|"):
                return r
        for r in c:
            if (r.get("instrument_key") or "").startswith("BSE_EQ|"):
                return r
        return c[0]

    # internals
    def _is_stale(self, p: pathlib.Path, max_age_h: int) -> bool:
        try:
            return (time.time() - p.stat().st_mtime)/3600.0 > max_age_h
        except Exception:
            return True

    def _download(self, url: str) -> None:
        if not _HAS_REQ:
            raise RuntimeError("requests not installed")
        insecure = (os.environ.get("ALLOW_INSECURE_SSL","").lower() == "true")
        self.log.info("Downloading instruments: %s", url)
        try:
            r = requests.get(url, timeout=120, verify=self.verify_tls and not insecure)
        except Exception as e:
            if not insecure: raise
            self.log.warning("TLS verify failed, retrying insecure: %s", e)
            r = requests.get(url, timeout=120, verify=False)
        r.raise_for_status()
        data = r.content
        if url.endswith(".gz"):
            data = gzip.decompress(data)
        txt = data.decode("utf-8","ignore")
        self.cache_json.write_text(txt, encoding="utf-8")
        self.log.info("Saved instruments cache: %s", self.cache_json)

    def _load_index(self, txt: str) -> None:
        obj = json.loads(txt)
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            rows = obj["data"]
        elif isinstance(obj, list):
            rows = obj
        else:
            rows = []
        self.rows = [self._norm_row(x) for x in rows if isinstance(x, dict)]

        self._by_isin.clear(); self._by_symbol.clear(); self._by_token.clear()
        for r in self.rows:
            isin = r.get("isin"); sym = r.get("symbol"); name = r.get("name")
            if isin: self._by_isin[_norm(isin)] = r
            if sym:  self._by_symbol.setdefault(_norm(sym), []).append(r)
            for t in set(_ntoks(name or "")) | set(_ntoks(sym or "")):
                if t: self._by_token.setdefault(t, []).append(r)

        self.log.info("Instruments indexed: %d", len(self.rows))

    def _norm_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        def pick(*keys: str) -> Optional[str]:
            for k in keys:
                if k in row and row[k] not in (None, ""):
                    return str(row[k])
            return None
        instrument_key = pick("instrument_key","instrumentKey","instrument")
        if not instrument_key:
            exch = pick("exchange","Exchange"); token = pick("token","symbol_token","instrument_token","exchange_token")
            if exch and token: instrument_key = f"{exch}|{token}"
        return {
            "instrument_key": instrument_key,
            "symbol": pick("trading_symbol","tradingsymbol","symbol","Symbol"),
            "name": pick("name","company_name","security_name","description","CompanyName"),
            "isin": pick("isin","ISIN","isin_code","isinCode"),
            "exchange": pick("exchange","Exchange"),
            "segment": pick("segment","Segment"),
        }


# ---------- web fallback (Wikipedia → ISIN) ----------
class WebISINFallback:
    WIKI = "https://en.wikipedia.org/w/api.php"
    def __init__(self, log: logging.Logger, session: Optional["requests.Session"]=None, verify_tls: bool=True):
        self.log = log
        self.sess = session or requests.Session()
        self.verify_tls = verify_tls

    def _get(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            r = self.sess.get(self.WIKI, params=params, timeout=20, verify=self.verify_tls)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            self.log.warning("web fallback failed: %s", e)
        return None

    def find_isin(self, query: str) -> Optional[str]:
        q = (query or "").strip()
        if not q: return None
        js = self._get({"action":"query","list":"search","srsearch":q,"format":"json","srlimit":5})
        if not js or "query" not in js: return None
        for item in js["query"].get("search", []):
            pid = item.get("pageid")
            if not pid: continue
            page = self._get({"action":"query","prop":"extracts","explaintext":1,"pageids":pid,"format":"json"})
            if not page or "query" not in page or "pages" not in page["query"]: continue
            extract = list(page["query"]["pages"].values())[0].get("extract") or ""
            m = re.search(r"\b(INE[A-Z0-9]{9,})\b", extract, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        return None


# ---------- indicators ----------
class TA:
    @staticmethod
    def closes(c): return [float(x[4]) for x in c if x and len(x)>=5]
    @staticmethod
    def highs(c):  return [float(x[2]) for x in c if x and len(x)>=5]
    @staticmethod
    def lows(c):   return [float(x[3]) for x in c if x and len(x)>=5]

    @staticmethod
    def sma(series: List[float], p: int) -> Optional[float]:
        if len(series) < p: return None
        return sum(series[-p:]) / p

    @staticmethod
    def ema(series: List[float], p: int) -> Optional[float]:
        if not series: return None
        k = 2.0/(p+1)
        e = series[0]
        for v in series[1:]:
            e = (v - e)*k + e
        return e

    @staticmethod
    def rsi(series: List[float], p: int=14) -> Optional[float]:
        if len(series) < p+1: return None
        gains=losses=0.0
        for i in range(1, p+1):
            ch = series[-p-1+i] - series[-p-2+i]
            gains += max(ch,0.0); losses += max(-ch,0.0)
        avg_gain = gains/p; avg_loss = losses/p
        if avg_loss == 0: return 100.0
        rs = avg_gain/avg_loss
        return 100.0 - 100.0/(1.0+rs)

    @staticmethod
    def macd(series: List[float], f:int=12, s:int=26, sig:int=9) -> Tuple[Optional[float],Optional[float],Optional[float]]:
        if len(series) < s+sig: return (None,None,None)
        def ema(p):
            k=2.0/(p+1); e=series[0]
            for v in series[1:]: e=(v-e)*k+e
            return e
        ema_f = ema(f); ema_s = ema(s)
        macd = ema_f - ema_s
        # signal from MACD history approx via re-EMA over closes (simple cheap proxy):
        # more correct is EMA over MACD series; for simplicity keep minimal compute:
        # (still good enough for hinting)
        # Here compute a short backwindow MACD trail:
        macd_series=[]
        ef=None; es=None
        kf=2.0/(f+1); ks=2.0/(s+1)
        e_sig=None; ks2=2.0/(sig+1)
        for v in series[-(s+sig+5):]:
            ef = v if ef is None else (v-ef)*kf + ef
            es = v if es is None else (v-es)*ks + es
            macd_series.append(ef-es)
        for m in macd_series:
            e_sig = m if e_sig is None else (m-e_sig)*ks2 + e_sig
        signal = e_sig
        hist = macd - signal if (signal is not None) else None
        return (macd, signal, hist)

    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], p:int=14) -> Optional[float]:
        if len(closes) < p+1: return None
        trs=[]
        prev = closes[-p-1]
        for i in range(-p,0):
            h,l,c = highs[i], lows[i], closes[i]
            trs.append(max(h-l, abs(h-prev), abs(l-prev)))
            prev = c
        return sum(trs)/p

    @staticmethod
    def vwap_today(candles: List[List[Any]]) -> Optional[float]:
        if not candles: return None
        today = candles[0][0][:10]
        pv=v=0.0
        for c in candles:
            if c[0][:10] != today: break
            h,l,cl = float(c[2]), float(c[3]), float(c[4])
            vol = float(c[5]) if len(c)>5 and c[5] is not None else 0.0
            tp = (h+l+cl)/3.0
            pv += tp*vol; v += vol
        return (pv/v) if v>0 else None


# ---------- client ----------
class UpstoxTechnicalClient:
    IST = ZoneInfo(os.environ.get("TZ","Asia/Kolkata"))

    def __init__(self, access_token: Optional[str]=None, api_base: Optional[str]=None, verify_tls: bool=True):
        if not _HAS_REQ: raise RuntimeError("requests is required")
        self.api_base = (api_base or os.environ.get("UPSTOX_API_BASE","https://api.upstox.com")).rstrip("/")
        self.access_token = (access_token or os.environ.get("UPSTOX_ACCESS_TOKEN") or "").strip()
        self.verify_tls = verify_tls
        self.sess = requests.Session()
        self.log = _mk_logger("upstox_tech")

        # instruments
        self.cache = InstrumentCache(self.log, verify_tls=verify_tls)
        max_age = int(os.environ.get("UPSTOX_INSTR_MAX_AGE_H","24"))
        prefer_nse = bool(int(os.environ.get("UPSTOX_NSE_ONLY","1")))
        self.cache.ensure(max_age_h=max_age, prefer_nse_only=prefer_nse)

        # web fallback
        self.web = WebISINFallback(self.log, session=self.sess, verify_tls=verify_tls)

    # http
    def _headers(self) -> Dict[str,str]:
        h = {"Accept":"application/json"}
        if self.access_token: h["Authorization"] = f"Bearer {self.access_token}"
        return h

    def _get(self, path: str, params: Optional[Dict[str,Any]]=None, timeout:int=30) -> Tuple[int, Any]:
        url = path if path.startswith("http") else (self.api_base + path)
        insecure = (os.environ.get("ALLOW_INSECURE_SSL","").lower()=="true")
        try:
            r = self.sess.get(url, headers=self._headers(), params=params, timeout=timeout, verify=self.verify_tls and not insecure)
            if r.status_code == 200 and r.content:
                try: return r.status_code, r.json()
                except Exception: return r.status_code, {}
            return r.status_code, (r.json() if r.content else {})
        except Exception as e:
            if not insecure:
                self.log.warning("GET %s failed: %s", url, e)
                return 0, {"error": str(e)}
            self.log.warning("Retrying insecure due to ALLOW_INSECURE_SSL=true: %s", e)
            try:
                r = self.sess.get(url, headers=self._headers(), params=params, timeout=timeout, verify=False)
                if r.status_code == 200 and r.content:
                    try: return r.status_code, r.json()
                    except Exception: return r.status_code, {}
                return r.status_code, (r.json() if r.content else {})
            except Exception as e2:
                return 0, {"error": str(e2)}

    # resolve
    def resolve(self, query: str) -> Dict[str,Any]:
        row = self.cache.resolve(query)
        if row: return row
        # web fallback → ISIN → validate in cache
        isin = self.web.find_isin(query)
        if isin:
            row2 = self.cache.resolve(isin)
            if row2: return row2
        raise RuntimeError(f"Unable to resolve instrument for: {query}")

    # candles
    @staticmethod
    def _dedupe_sort_desc(c: List[List[Any]]) -> List[List[Any]]:
        seen=set(); out=[]
        for x in c:
            if not x or len(x)<5 or not isinstance(x[0],str): continue
            if x[0] in seen: continue
            seen.add(x[0]); out.append(x)
        out.sort(key=lambda r: r[0], reverse=True)
        return out

    def ohlc_30m_with_today(self, instrument_key: str, days: int) -> List[List[Any]]:
        interval = "30minute"
        end = dt.datetime.now(tz=self.IST).date()
        start = end - dt.timedelta(days=max(days+5, 35))
        sc, d = self._get(f"/v2/historical-candle/{instrument_key}/{interval}/{end:%Y-%m-%d}/{start:%Y-%m-%d}")
        merged: List[List[Any]] = []
        if sc == 200 and isinstance(d, dict):
            merged.extend(((d.get("data") or {}).get("candles") or []))
        sc, d = self._get(f"/v2/historical-candle/intraday/{instrument_key}/{interval}")
        if sc == 200 and isinstance(d, dict):
            merged.extend(((d.get("data") or {}).get("candles") or []))
        return self._dedupe_sort_desc(merged)

    def ltp(self, instrument_key: str) -> Tuple[Optional[float], str]:
        sc, data = self._get("/v2/market-quote/quotes", {"instrument_key": instrument_key})
        if sc == 200 and isinstance(data, dict):
            qd = data.get("data", {}) or {}
            for k in (instrument_key, instrument_key.replace("|",":")):
                if k in qd:
                    q = qd[k] or {}
                    lp = q.get("last_price")
                    if lp is not None:
                        return float(lp), "live"
                    o = (q.get("ohlc") or {}).get("close")
                    if o is not None:
                        return float(o), "previous_close"
        return None, "unavailable"

    def snapshot(self, symbol_or_name: str, days: int=30) -> Dict[str, Any]:
        row = self.resolve(symbol_or_name)
        ik = row["instrument_key"]
        candles = self.ohlc_30m_with_today(ik, days=days)

        closes = TA.closes(candles)[::-1]  # oldest→newest
        highs  = TA.highs(candles)[::-1]
        lows   = TA.lows(candles)[::-1]

        sma20 = TA.sma(closes, 20); sma50 = TA.sma(closes, 50)
        ema20 = TA.ema(closes, 20); ema50 = TA.ema(closes, 50)
        rsi14 = TA.rsi(closes, 14)
        macd, signal, hist = TA.macd(closes, 12, 26, 9)
        atr14 = TA.atr(highs, lows, closes, 14)
        vwap  = TA.vwap_today(candles)

        ltp_val, ltp_src = self.ltp(ik)
        last_close = TA.closes(candles)[0] if candles else None
        prev_close = TA.closes(candles)[1] if len(candles)>1 else None
        change_pct = ((last_close - prev_close)/prev_close*100.0) if (last_close and prev_close) else None

        # labeled recent candles
        labeled: List[Dict[str, Any]] = []
        for c in candles[: max(20, min(200, len(candles)))]:
            labeled.append({
                "timestamp": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": (float(c[5]) if len(c)>5 and c[5] is not None else None),
                "oi": (float(c[6]) if len(c)>6 and c[6] is not None else None),
            })

        latest = labeled[0] if labeled else None

        return {
            "query": symbol_or_name,
            "instrument_key": ik,
            "symbol": row.get("symbol"),
            "company_name": row.get("name"),
            "interval": "30minute",
            "lookback_days": days,
            "current_price": _round(ltp_val if ltp_val is not None else last_close),
            "price_source": ltp_src if ltp_val is not None else "last_candle",
            "last_close": _round(last_close),
            "prev_close": _round(prev_close),
            "change_percent": _round(change_pct),
            "latest_candle": latest,
            "indicators": {
                "sma20": _round(sma20),
                "sma50": _round(sma50),
                "ema20": _round(ema20),
                "ema50": _round(ema50),
                "rsi14": _round(rsi14),
                "macd": _round(macd),
                "signal": _round(signal),
                "hist": _round(hist),
                "atr14": _round(atr14),
                "vwap_today": _round(vwap),
            },
            "candles": labeled,  # most-recent first
            "generated_at": dt.datetime.now(tz=self.IST).isoformat(),
            "data_source": "upstox.combined",
        }


# ---------- minimal CLI ----------
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Upstox 30m technical snapshot (only: --symbol, --days)")
    ap.add_argument("--symbol", required=True, help="Company name / trading symbol / ISIN / instrument_key")
    ap.add_argument("--days", type=int, default=30, help="Lookback in days (default 30)")
    args = ap.parse_args()

    client = UpstoxTechnicalClient()
    snap = client.snapshot(args.symbol, days=args.days)
    print(json.dumps(snap, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _main()
