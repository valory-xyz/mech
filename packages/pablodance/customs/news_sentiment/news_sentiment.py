from __future__ import annotations
import json, logging, math, re, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
ALLOWED_TOOLS = ["news-sentiment", "news-sentiment-es"]
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Optional[Dict[str, Any]]]
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
API_TIMEOUT_SECS = 30
MAX_HEADLINES = 20
FETCH_TIMEOUT = 10
MAX_ARTICLES = 40
MAX_PARALLEL = 8
MIN_RELEVANCE = 0.05

STATIC_FEEDS = [
    ("https://feeds.bbci.co.uk/news/world/rss.xml", "bbc.com", "en"),
    ("https://feeds.reuters.com/reuters/worldNews", "reuters.com", "en"),
    ("https://rss.ap.org/apf-topnews", "apnews.com", "en"),
    ("https://www.aljazeera.com/xml/rss/all.xml", "aljazeera.com", "en"),
    ("https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "nytimes.com", "en"),
    ("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/internacional/portada", "elpais.com", "es"),
    ("https://www.bbc.com/mundo/index.xml", "bbc.com/mundo", "es"),
    ("https://www.infobae.com/feeds/rss/", "infobae.com", "es"),
    ("https://rss.clarin.com/rss/internacionales.xml", "clarin.com", "es"),
]

class Article:
    __slots__ = ("title", "source", "summary")
    def __init__(self, title, source, summary=""):
        self.title = title; self.source = source; self.summary = summary

def _google_news_url(query, lang):
    q = urllib.parse.quote(query)
    if lang == "es":
        return f"https://news.google.com/rss/search?q={q}&hl=es-419&gl=US&ceid=US:es-419"
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

def _relevance(title, keywords):
    tl = title.lower()
    hits = sum(1 for kw in keywords if kw in tl)
    return hits / len(keywords) if keywords else 0.0

def _fetch_feed(url, source, keywords, min_rel):
    try:
        import feedparser
    except ImportError:
        return []
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
        arts = []
        for e in feed.entries[:25]:
            t = getattr(e, "title", "").strip()
            if not t: continue
            if keywords and _relevance(t, keywords) < min_rel: continue
            arts.append(Article(t, source, getattr(e, "summary", "")[:300]))
        return arts
    except Exception:
        return []

def scrape_news(query, lang="en"):
    keywords = [w.lower() for w in re.split(r"\W+", query) if len(w) > 2]
    tasks = [(_google_news_url(query, lang), "news.google.com", [], 0.0)]
    for url, src, fl in STATIC_FEEDS:
        if fl == lang:
            tasks.append((url, src, keywords, MIN_RELEVANCE))
    all_arts = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futs = {ex.submit(_fetch_feed, u, s, k, r): s for u, s, k, r in tasks}
        for f in as_completed(futs, timeout=FETCH_TIMEOUT+2):
            try: all_arts.extend(f.result())
            except Exception: pass
    seen = set(); unique = []
    for a in all_arts:
        if a.title not in seen:
            seen.add(a.title); unique.append(a)
    return unique[:MAX_ARTICLES]

def _build_prompt(query, articles, lang):
    hb = "\n".join(f"{i+1}. [{a.source}] {a.title}" + (f"\n   Context: {a.summary}" if a.summary else "") for i, a in enumerate(articles))
    if lang == "es":
        ins = ("Eres un analista de sentimiento de noticias financiero experto. "
               "Analiza los siguientes titulares en relacion con la consulta y devuelve "
               "UNICAMENTE un JSON valido con estos campos:\n\n"
               '{"sentiment":"positive"|"neutral"|"negative","score":<-1.0..1.0>,"confidence":<0.0..1.0>,"summary":"<1-3 oraciones>"}\n\n'
               "- score: -1.0=muy negativo, 0.0=neutro, +1.0=muy positivo\n"
               "- Responde SOLO con el JSON, sin markdown\n\n")
        return f"{ins}Consulta: {query}\n\nTitulares:\n{hb}"
    ins = ("You are an expert financial news sentiment analyst. "
           "Analyse the following headlines in relation to the query "
           "and return ONLY a valid JSON with these fields:\n\n"
           '{"sentiment":"positive"|"neutral"|"negative","score":<-1.0..1.0>,"confidence":<0.0..1.0>,"summary":"<1-3 sentences>"}\n\n'
           "- Reply ONLY with JSON, no markdown\n\n")
    return f"{ins}Query: {query}\n\nHeadlines:\n{hb}"

def _parse(text):
    text = text.strip()
    try: return json.loads(text)
    except: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
    return None

def _bull(score, confidence):
    p = 1.0 / (1.0 + math.exp(-score * confidence * 3.0))
    return round(max(0.01, min(0.99, p)), 4)

def analyse_sentiment(query, articles, lang, api_key):
    try: import anthropic
    except ImportError: return {"error": "anthropic not installed"}
    if not articles:
        return {"query": query, "sentiment": "neutral", "score": 0.0, "confidence": 0.0,
                "bull_probability": 0.5, "summary": "No articles found.", "sources": [], "headlines_used": 0, "error": None}
    selected = articles[:MAX_HEADLINES]
    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=API_TIMEOUT_SECS)
        r = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, messages=[{"role":"user","content":_build_prompt(query,selected,lang)}])
        raw = r.content[0].text
    except Exception as exc:
        return {"query": query, "sentiment": "neutral", "score": 0.0, "confidence": 0.0,
                "bull_probability": 0.5, "summary": f"API error: {type(exc).__name__}", "sources": [], "headlines_used": 0, "error": str(exc)}
    parsed = _parse(raw)
    if not parsed:
        return {"query": query, "sentiment": "neutral", "score": 0.0, "confidence": 0.0,
                "bull_probability": 0.5, "summary": "Could not parse response.", "sources": [], "headlines_used": 0, "error": "parse_error"}
    s = float(parsed.get("score", 0.0)); c = float(parsed.get("confidence", 0.5))
    return {"query": query, "sentiment": parsed.get("sentiment","neutral"), "score": round(s,4),
            "confidence": round(c,4), "bull_probability": _bull(s,c), "summary": parsed.get("summary",""),
            "sources": list(dict.fromkeys(a.source for a in selected)), "headlines_used": len(selected), "error": None}

def run(**kwargs):
    prompt = kwargs.get("prompt")
    if not prompt:
        return json.dumps({"error": "No prompt provided."}), prompt, None, None, None
    tool = kwargs.get("tool", "news-sentiment")
    if tool not in ALLOWED_TOOLS:
        return json.dumps({"error": f"Unknown tool {tool!r}."}), prompt, None, None, None
    lang = "es" if tool == "news-sentiment-es" else "en"
    api_key = kwargs.get("anthropic_api_key")
    if not api_key:
        ak = kwargs.get("api_keys", {})
        api_key = ak.get("anthropic") or ak.get("anthropic_api_key")
    if not api_key:
        return json.dumps({"error": "Missing anthropic_api_key."}), prompt, None, None, None
    articles = scrape_news(query=prompt, lang=lang)
    result = analyse_sentiment(query=prompt, articles=articles, lang=lang, api_key=api_key)
    return json.dumps(result, ensure_ascii=False), prompt, None, None, None