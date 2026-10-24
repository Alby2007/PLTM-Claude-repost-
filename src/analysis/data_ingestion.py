"""
Data Ingestion Pipeline

Ingests data from multiple sources, extracts semantic triples via LLM,
and stores them as PLTM atoms. Sources: URLs, text, files, arXiv, Wikipedia.

Uses Groq (free tier) for triple extraction to keep costs at $0.
"""

import json
import os
import re
import sqlite3
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger


class _HTMLTextExtractor(HTMLParser):
    """Extract readable text from HTML."""
    def __init__(self):
        super().__init__()
        self._text = []
        self._skip = False
        self._skip_tags = {"script", "style", "noscript", "svg"}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "li", "tr"):
            self._text.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._text.append(data.strip())

    def get_text(self) -> str:
        return " ".join(self._text)


def _strip_html(html: str) -> str:
    """Strip HTML tags and return clean text."""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html)
        return extractor.get_text()
    except Exception:
        return re.sub(r'<[^>]+>', ' ', html)


def _fetch_url(url: str, timeout: int = 15) -> str:
    """Fetch URL content."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "PLTM/2.0 (Research Bot)",
        "Accept": "text/html,application/xml,text/plain,application/json",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


_ROUTER = None


def _get_router():
    """Lazy shared ModelRouter (avoids _ensure_tables DDL + ollama ping per call)."""
    global _ROUTER
    if _ROUTER is None:
        from src.analysis.model_router import ModelRouter
        _ROUTER = ModelRouter()
    return _ROUTER


def _extract_triples_via_groq(text: str, domain: str, source: str,
                                max_triples: int = 15) -> List[Dict]:
    """Use Groq to extract semantic triples from text."""
    router = _get_router()

    # Truncate text to fit context window
    text = text[:6000]

    prompt = f"""Extract {max_triples} factual subject-predicate-object triples from this text.
Domain: {domain}
Source: {source}

Rules:
- Subject: entity or concept name (e.g. "North_Korea", "IIT_theory", "Gold_reserves")
- Predicate: relationship verb (e.g. "tested", "claims", "increased_by", "published")
- Object: factual claim with specifics (dates, numbers, quotes)
- Only extract factual claims, not opinions
- Use underscores in multi-word subjects/predicates

Return ONLY a JSON array, no other text:
[{{"s":"subject","p":"predicate","o":"object"}}]

Text:
{text}"""

    result = router.call(
        prompt=prompt,
        provider="groq",
        task_type="extraction",
        max_tokens=2000,
        temperature=0.1,
    )

    if not result.get("ok"):
        # Fallback to ollama if groq fails
        result = router.call(
            prompt=prompt,
            provider="ollama",
            task_type="extraction",
            max_tokens=2000,
            temperature=0.1,
        )

    if not result.get("ok"):
        return []

    # Parse JSON from response
    response_text = result.get("text", "")
    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
    if not json_match:
        return []

    try:
        triples = json.loads(json_match.group())
        raw = [t for t in triples if isinstance(t, dict) and "s" in t and "p" in t and "o" in t]
        return _filter_triples(raw, text)
    except (json.JSONDecodeError, TypeError):
        return []


_PLACEHOLDER = {"subject", "predicate", "object", "s", "p", "o", "entity", "value"}


def _filter_triples(triples: List[Dict], text: str) -> List[Dict]:
    """Drop garbage triples: prompt-echo placeholders and subjects with no
    grounding in the source text (small LLMs echo the prompt examples)."""
    text_l = text.lower()
    kept = []
    for t in triples:
        s, p, o = (str(t.get(k, "")).strip() for k in "spo")
        if not s or not p or not o:
            continue
        if s.lower() in _PLACEHOLDER or p.lower() in _PLACEHOLDER:
            continue
        # Subject grounding: at least one >3-char word must appear in source
        words = [w for w in re.split(r"[_\s]+", s.lower()) if len(w) > 3]
        if words and not any(w in text_l for w in words):
            continue
        kept.append(t)
    return kept


def _store_triples(triples: List[Dict], domain: str, source: str,
                   source_type: str, confidence: float = 0.8) -> Dict:
    """Store extracted triples as PLTM atoms.

    SafetyJudge screens each triple — raw INSERT bypasses the memory jury,
    so SSNs/passwords/valid card numbers must be filtered here or PII would
    land in the store unchecked.
    """
    from src.db_config import get_db_path
    from src.memory.memory_jury import SafetyJudge, Verdict
    safety = SafetyJudge()

    conn = sqlite3.connect(str(get_db_path()))
    conn.execute("PRAGMA busy_timeout=2000")  # fail fast — sync writer in async loop

    stored = 0
    skipped_dupes = 0
    skipped_safety = 0
    for t in triples:
        atom_id = str(uuid4())
        subject = str(t.get("s", ""))[:200]
        predicate = str(t.get("p", ""))[:100]
        obj = str(t.get("o", ""))[:500]

        if not subject or not predicate or not obj:
            continue

        if safety.evaluate(f"{subject} {predicate} {obj}").verdict == Verdict.REJECT:
            skipped_safety += 1
            continue

        metadata = json.dumps({
            "contexts": [domain, source_type],
            "provenance": "inferred",
            "source_user": f"ingestion_{source_type}",
            "source": source[:200],
            "ingested_at": time.time(),
            "strength": confidence,
        })

        now = time.time()
        try:
            conn.execute(
                """INSERT INTO atoms (id, atom_type, graph, subject, predicate, object,
                   confidence, first_observed, last_accessed, metadata)
                   VALUES (?, 'state', 'substantiated', ?, ?, ?, ?, ?, ?, ?)""",
                (atom_id, subject, predicate, obj, confidence, now, now, metadata)
            )
            stored += 1
        except sqlite3.IntegrityError:
            skipped_dupes += 1  # UNIQUE(subject,predicate,object,graph) — reassertion
        except Exception as e:
            logger.warning(f"Failed to store triple: {e}")

    conn.commit()
    conn.close()
    return {"stored": stored, "total_extracted": len(triples),
            "skipped_dupes": skipped_dupes, "skipped_safety": skipped_safety}


def _llm_note(result: Dict) -> Dict:
    """If nothing was extracted, explain why (missing keys beats silent zeros)."""
    if result.get("extracted", result.get("total_extracted", 0)):
        return result
    router = _get_router()
    ok = [p for p in router.get_available_providers()
          if p["status"] == "available" and p["id"] in ("groq", "ollama")]
    if not ok:
        result["note"] = ("No LLM available — set GROQ_API_KEY or start "
                          "Ollama; extraction returned 0 triples")
    return result


class DataIngestion:
    """Multi-source data ingestion pipeline."""

    def ingest_url(self, url: str, domain: str = "general",
                   max_triples: int = 15) -> Dict:
        """Scrape a URL, extract triples, store as atoms."""
        try:
            raw = _fetch_url(url)
            text = _strip_html(raw)
            text = re.sub(r'\s+', ' ', text).strip()

            if len(text) < 50:
                return {"ok": False, "err": "Page too short or empty"}

            triples = _extract_triples_via_groq(text, domain, url, max_triples)
            result = _store_triples(triples, domain, url, "url")

            return _llm_note({
                "ok": True,
                "source": url,
                "domain": domain,
                "text_length": len(text),
                "triples": result["stored"],
                "extracted": result["total_extracted"],
                "skipped_dupes": result["skipped_dupes"],
                "skipped_safety": result["skipped_safety"],
                "sample": triples[:3] if triples else [],
            })
        except Exception as e:
            return {"ok": False, "err": str(e)[:200], "source": url}

    def ingest_text(self, text: str, domain: str = "general",
                    source: str = "user_input", max_triples: int = 15) -> Dict:
        """Extract triples from raw text and store as atoms."""
        if len(text) < 20:
            return {"ok": False, "err": "Text too short"}

        triples = _extract_triples_via_groq(text, domain, source, max_triples)
        result = _store_triples(triples, domain, source, "text")

        return _llm_note({
            "ok": True,
            "source": source,
            "domain": domain,
            "text_length": len(text),
            "triples": result["stored"],
            "extracted": result["total_extracted"],
            "skipped_dupes": result["skipped_dupes"],
            "skipped_safety": result["skipped_safety"],
            "sample": triples[:3] if triples else [],
        })

    def ingest_file(self, file_path: str, domain: str = "general",
                    max_triples: int = 20) -> Dict:
        """Read a local file and extract triples."""
        path = Path(file_path)
        if not path.exists():
            return {"ok": False, "err": f"File not found: {file_path}"}

        ext = path.suffix.lower()
        try:
            if ext in (".txt", ".md", ".rst", ".log"):
                text = path.read_text(errors="replace")
            elif ext == ".csv":
                text = path.read_text(errors="replace")
                # Take first 100 lines for CSV
                lines = text.split("\n")[:100]
                text = "\n".join(lines)
            elif ext == ".json":
                data = json.loads(path.read_text())
                text = json.dumps(data, indent=2)[:6000]
            else:
                return {"ok": False, "err": f"Unsupported file type: {ext}. Supported: .txt, .md, .csv, .json, .rst, .log"}

            triples = _extract_triples_via_groq(text, domain, str(path.name), max_triples)
            result = _store_triples(triples, domain, str(path), "file")

            return _llm_note({
                "ok": True,
                "source": str(path),
                "domain": domain,
                "file_size": path.stat().st_size,
                "triples": result["stored"],
                "extracted": result["total_extracted"],
                "skipped_dupes": result["skipped_dupes"],
                "skipped_safety": result["skipped_safety"],
                "sample": triples[:3] if triples else [],
            })
        except Exception as e:
            return {"ok": False, "err": str(e)[:200]}

    def ingest_arxiv(self, query: str, domain: str = "science",
                     max_results: int = 5, max_triples_per_paper: int = 10) -> Dict:
        """Search arXiv, fetch abstracts, extract triples from each paper."""
        try:
            # arXiv API search
            search_url = (
                f"http://export.arxiv.org/api/query?"
                f"search_query=all:{urllib.request.quote(query)}"
                f"&start=0&max_results={max_results}"
                f"&sortBy=relevance&sortOrder=descending"
            )
            raw = _fetch_url(search_url, timeout=20)
            root = ET.fromstring(raw)

            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

            papers = []
            total_stored = 0
            total_extracted = 0

            for entry in root.findall("atom:entry", ns):
                title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
                abstract = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
                arxiv_id = entry.findtext("atom:id", "", ns).strip().split("/abs/")[-1]
                published = entry.findtext("atom:published", "", ns)[:10] if entry.findtext("atom:published", "", ns) else ""

                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.findtext("atom:name", "", ns)
                    if name:
                        authors.append(name)

                categories = []
                for cat in entry.findall("arxiv:primary_category", ns):
                    term = cat.get("term", "")
                    if term:
                        categories.append(term)
                for cat in entry.findall("atom:category", ns):
                    term = cat.get("term", "")
                    if term and term not in categories:
                        categories.append(term)

                # Build rich text for extraction
                paper_text = f"Title: {title}\nAuthors: {', '.join(authors[:5])}\narXiv: {arxiv_id}\nPublished: {published}\nCategories: {', '.join(categories)}\n\nAbstract: {abstract}"

                source = f"arxiv:{arxiv_id}"
                triples = _extract_triples_via_groq(paper_text, domain, source, max_triples_per_paper)

                # Add provenance triple for the paper itself
                paper_triple = {
                    "s": arxiv_id.replace("/", "_"),
                    "p": "arxiv_paper",
                    "o": f"{title[:150]} | {', '.join(authors[:3])} | {published}"
                }
                triples.insert(0, paper_triple)

                result = _store_triples(triples, domain, source, "arxiv")
                total_stored += result["stored"]
                total_extracted += result["total_extracted"]

                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": title[:100],
                    "authors": authors[:3],
                    "published": published,
                    "categories": categories[:3],
                    "triples_stored": result["stored"],
                })

            return {
                "ok": True,
                "query": query,
                "domain": domain,
                "papers_found": len(papers),
                "total_triples_stored": total_stored,
                "total_extracted": total_extracted,
                "papers": papers,
            }
        except Exception as e:
            return {"ok": False, "err": str(e)[:200], "query": query}

    def ingest_wikipedia(self, topic: str, domain: str = "general",
                         max_triples: int = 15) -> Dict:
        """Fetch Wikipedia article summary and extract triples."""
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.request.quote(topic)}"
            raw = _fetch_url(url)
            data = json.loads(raw)

            title = data.get("title", topic)
            extract = data.get("extract", "")
            description = data.get("description", "")
            page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

            if not extract:
                return {"ok": False, "err": f"No Wikipedia article found for: {topic}"}

            text = f"Title: {title}\nDescription: {description}\n\n{extract}"
            source = f"wikipedia:{title}"

            triples = _extract_triples_via_groq(text, domain, source, max_triples)
            result = _store_triples(triples, domain, source, "wikipedia")

            return {
                "ok": True,
                "source": source,
                "title": title,
                "domain": domain,
                "text_length": len(extract),
                "url": page_url,
                "triples": result["stored"],
                "extracted": result["total_extracted"],
                "sample": triples[:3] if triples else [],
            }
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"ok": False, "err": f"Wikipedia article not found: {topic}"}
            return {"ok": False, "err": str(e)[:200]}
        except Exception as e:
            return {"ok": False, "err": str(e)[:200]}

    def ingest_rss(self, feed_url: str, domain: str = "general",
                   max_items: int = 5, max_triples_per_item: int = 8) -> Dict:
        """Fetch RSS/Atom feed, extract triples from each item."""
        try:
            raw = _fetch_url(feed_url, timeout=15)
            root = ET.fromstring(raw)

            items = []
            total_stored = 0

            # Try RSS 2.0 format
            rss_items = root.findall(".//item")
            # Try Atom format
            if not rss_items:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                rss_items = root.findall("atom:entry", ns)

            for item in rss_items[:max_items]:
                # RSS 2.0
                title = item.findtext("title", "") or ""
                description = item.findtext("description", "") or ""
                link = item.findtext("link", "") or ""
                pub_date = item.findtext("pubDate", "") or ""

                # Atom fallback
                if not title:
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    title = item.findtext("atom:title", "", ns) or ""
                    description = item.findtext("atom:summary", "", ns) or item.findtext("atom:content", "", ns) or ""
                    link_el = item.find("atom:link", ns)
                    link = link_el.get("href", "") if link_el is not None else ""
                    pub_date = item.findtext("atom:published", "", ns) or item.findtext("atom:updated", "", ns) or ""

                text = _strip_html(f"Title: {title}\nDate: {pub_date}\n\n{_strip_html(description)}")
                if len(text) < 30:
                    continue

                source = link or feed_url
                triples = _extract_triples_via_groq(text, domain, source, max_triples_per_item)
                result = _store_triples(triples, domain, source, "rss")
                total_stored += result["stored"]

                items.append({
                    "title": title[:80],
                    "link": link[:100],
                    "triples_stored": result["stored"],
                })

            return {
                "ok": True,
                "feed": feed_url,
                "domain": domain,
                "items_processed": len(items),
                "total_triples_stored": total_stored,
                "items": items,
            }
        except Exception as e:
            return {"ok": False, "err": str(e)[:200], "feed": feed_url}
