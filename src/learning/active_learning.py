"""
Active Learning Loops

Two core loops that enable PLTM to actively seek and validate knowledge:

1. SEARCH → EXTRACT → VERIFY → STORE Pipeline
   - Pose a question or topic
   - Search web/arXiv for answers
   - Extract structured facts with provenance
   - Verify against existing knowledge (conflict detection)
   - Store verified facts, flag conflicts for review

2. HYPOTHESIS → TEST → UPDATE Cycle
   - Generate hypotheses from existing knowledge gaps
   - Design verification strategies (what evidence would confirm/refute)
   - Evaluate incoming evidence against open hypotheses
   - Update belief confidence based on evidence
   - Track hypothesis lifecycle (proposed → testing → confirmed/refuted/revised)

Both loops persist state to SQLite for cross-session continuity.
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import aiosqlite
from loguru import logger


@dataclass
class SearchResult:
    """A result from a knowledge search."""
    query: str
    source: str  # arxiv, web, github
    title: str
    content: str
    url: str
    relevance: float


@dataclass
class VerifiedFact:
    """A fact that has been verified against existing knowledge."""
    fact_id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    source_url: str
    verification_status: str  # new, confirmed, conflicting, duplicate
    conflict_details: Optional[str] = None


@dataclass
class Hypothesis:
    """A hypothesis being tracked through its lifecycle."""
    hypothesis_id: str
    statement: str
    domains: List[str]
    prior_confidence: float  # Initial belief strength
    current_confidence: float
    status: str  # proposed, testing, confirmed, refuted, revised
    evidence_for: List[Dict[str, Any]] = field(default_factory=list)
    evidence_against: List[Dict[str, Any]] = field(default_factory=list)
    verification_strategy: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    resolution: str = ""


class ActiveLearningEngine:
    """
    Active learning loops: search→verify→store and hypothesis→test→update.

    All state persists to SQLite for cross-session continuity.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent.parent.parent / "data" / "pltm_mcp.db"
        self._conn: Optional[aiosqlite.Connection] = None
        self._store = None
        from src.analysis.model_router import any_llm_available
        self._llm_enabled = any_llm_available()

    async def connect(self, store=None):
        """Open DB connection and ensure tables."""
        self._conn = await aiosqlite.connect(str(self.db_path))
        await self._conn.execute("PRAGMA journal_mode=WAL")
        if store:
            self._store = store
        await self._ensure_tables()
        logger.info("ActiveLearningEngine connected")

    async def _ensure_tables(self):
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS active_searches (
                search_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                source TEXT DEFAULT 'arxiv',
                status TEXT DEFAULT 'pending',
                results_count INTEGER DEFAULT 0,
                facts_extracted INTEGER DEFAULT 0,
                facts_verified INTEGER DEFAULT 0,
                facts_stored INTEGER DEFAULT 0,
                conflicts_found INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                completed_at REAL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_as_status ON active_searches(status);

            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                domains TEXT DEFAULT '[]',
                prior_confidence REAL DEFAULT 0.5,
                current_confidence REAL DEFAULT 0.5,
                status TEXT DEFAULT 'proposed',
                evidence_for TEXT DEFAULT '[]',
                evidence_against TEXT DEFAULT '[]',
                verification_strategy TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                resolution TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_hyp_status ON hypotheses(status);

            CREATE TABLE IF NOT EXISTS evidence_log (
                evidence_id TEXT PRIMARY KEY,
                hypothesis_id TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                content TEXT NOT NULL,
                source_url TEXT DEFAULT '',
                direction TEXT NOT NULL,
                strength REAL DEFAULT 0.5,
                logged_at REAL NOT NULL,
                FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
            );
            CREATE INDEX IF NOT EXISTS idx_ev_hyp ON evidence_log(hypothesis_id);
        """)
        await self._conn.commit()

    # ══════════════════════════════════════════════════════════════════════════
    # PIPELINE 1: SEARCH → EXTRACT → VERIFY → STORE
    # ══════════════════════════════════════════════════════════════════════════

    async def search_and_learn(
        self,
        query: str,
        source: str = "arxiv",
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Full pipeline: search → extract → verify → store.

        Args:
            query: What to search for
            source: Where to search (arxiv, web, github)
            max_results: Max results to process

        Returns:
            Pipeline results with facts stored and conflicts found
        """
        search_id = f"search_{uuid4().hex[:8]}"
        now = time.time()

        await self._conn.execute(
            "INSERT INTO active_searches (search_id, query, source, status, created_at) VALUES (?,?,?,?,?)",
            (search_id, query[:500], source, "running", now),
        )
        await self._conn.commit()

        try:
            # Step 1: SEARCH
            results = await self._search(query, source, max_results)

            # Step 2: EXTRACT facts from results
            extracted_facts = []
            for result in results:
                facts = await self._extract_facts(result)
                extracted_facts.extend(facts)

            # Step 3: VERIFY against existing knowledge
            verified = []
            conflicts = []
            for fact in extracted_facts:
                verification = await self._verify_fact(fact)
                verified.append(verification)
                if verification.verification_status == "conflicting":
                    conflicts.append(verification)

            # Step 4: STORE verified facts
            stored = 0
            for vf in verified:
                if vf.verification_status in ("new", "confirmed"):
                    success = await self._store_verified_fact(vf)
                    if success:
                        stored += 1

            # Also check against open hypotheses
            hyp_matches = await self._check_against_hypotheses(verified)

            # Update search record
            await self._conn.execute(
                """UPDATE active_searches
                   SET status='completed', results_count=?, facts_extracted=?,
                       facts_verified=?, facts_stored=?, conflicts_found=?, completed_at=?
                   WHERE search_id=?""",
                (len(results), len(extracted_facts), len(verified), stored,
                 len(conflicts), time.time(), search_id),
            )
            await self._conn.commit()

            return {
                "ok": True,
                "search_id": search_id,
                "query": query[:100],
                "source": source,
                "results": len(results),
                "facts_extracted": len(extracted_facts),
                "facts_stored": stored,
                "conflicts": len(conflicts),
                "conflict_details": [
                    {"subject": c.subject, "predicate": c.predicate,
                     "new": c.object[:80], "issue": c.conflict_details[:100] if c.conflict_details else ""}
                    for c in conflicts[:5]
                ],
                "hypothesis_matches": hyp_matches,
            }

        except Exception as e:
            await self._conn.execute(
                "UPDATE active_searches SET status='failed' WHERE search_id=?",
                (search_id,),
            )
            await self._conn.commit()
            logger.error(f"Search pipeline failed: {e}")
            return {"ok": False, "err": str(e)[:200], "search_id": search_id}

    async def _search(self, query: str, source: str, max_results: int) -> List[SearchResult]:
        """Execute search against specified source."""
        results = []

        if source == "arxiv":
            results = await self._search_arxiv(query, max_results)
        elif source == "web":
            results = await self._search_web(query, max_results)
        elif source == "github":
            results = await self._search_github(query, max_results)

        return results

    async def _search_arxiv(self, query: str, max_results: int) -> List[SearchResult]:
        """Search arXiv for papers with retry on rate-limit."""
        import asyncio
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET

        # Encode query — preserve colons for field prefixes, + for spaces
        encoded = urllib.parse.quote(query, safe=":+")
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded}&max_results={max_results}"

        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "PLTM/1.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    xml_data = resp.read().decode("utf-8")

                root = ET.fromstring(xml_data)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                results = []

                for entry in root.findall("atom:entry", ns):
                    title_el = entry.find("atom:title", ns)
                    summary_el = entry.find("atom:summary", ns)
                    id_el = entry.find("atom:id", ns)

                    title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
                    summary = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
                    arxiv_url = id_el.text if id_el is not None else ""

                    results.append(SearchResult(
                        query=query,
                        source="arxiv",
                        title=title,
                        content=summary,
                        url=arxiv_url,
                        relevance=0.8,
                    ))

                return results
            except urllib.request.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    wait = 3 * (attempt + 1)
                    logger.info(f"arXiv rate-limited, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                logger.warning(f"arXiv search failed: {e}")
                return []
            except Exception as e:
                logger.warning(f"arXiv search failed: {e}")
                return []
        return []

    async def _search_web(self, query: str, max_results: int) -> List[SearchResult]:
        """Search web via HN Algolia API (free, no key needed)."""
        import urllib.request
        import urllib.parse

        try:
            encoded = urllib.parse.quote(query)
            url = f"https://hn.algolia.com/api/v1/search?query={encoded}&hitsPerPage={max_results}"
            req = urllib.request.Request(url, headers={"User-Agent": "PLTM/1.0"})

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []
            for hit in data.get("hits", [])[:max_results]:
                title = hit.get("title", "") or hit.get("story_title", "")
                url = hit.get("url", "") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
                content = hit.get("comment_text", "") or hit.get("story_text", "") or title

                if title:
                    results.append(SearchResult(
                        query=query,
                        source="web",
                        title=title[:200],
                        content=content[:1000],
                        url=url,
                        relevance=0.6,
                    ))
            return results
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []

    async def _search_github(self, query: str, max_results: int) -> List[SearchResult]:
        """Search GitHub repos."""
        import urllib.request
        import urllib.parse

        try:
            encoded = urllib.parse.quote(query)
            url = f"https://api.github.com/search/repositories?q={encoded}&sort=stars&per_page={max_results}"
            req = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "PLTM/1.0",
            })

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []
            for item in data.get("items", [])[:max_results]:
                results.append(SearchResult(
                    query=query,
                    source="github",
                    title=item.get("full_name", ""),
                    content=item.get("description", "")[:500],
                    url=item.get("html_url", ""),
                    relevance=0.7,
                ))
            return results
        except Exception as e:
            logger.warning(f"GitHub search failed: {e}")
            return []

    async def _extract_facts(self, result: SearchResult) -> List[VerifiedFact]:
        """Extract structured facts from a search result."""
        import re

        facts = []
        content = result.content
        sentences = re.split(r"(?<=[.!?])\s+", content)

        for sentence in sentences[:20]:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 500:
                continue

            # Pattern-based extraction
            patterns = [
                (r"(.{5,60}?) (?:is|are) (?:a|an|the) (.{5,200})", "is_a"),
                (r"(.{5,60}?) (?:shows?|demonstrates?) (?:that )?(.{5,200})", "shows"),
                (r"(.{5,60}?) (?:causes?|leads? to) (.{5,200})", "causes"),
                (r"(.{5,60}?) (?:improves?|increases?|reduces?) (.{5,200})", "affects"),
                (r"(.{5,60}?) (?:proposes?|introduces?) (.{5,200})", "proposes"),
            ]

            for pattern, predicate in patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    facts.append(VerifiedFact(
                        fact_id=f"fact_{uuid4().hex[:8]}",
                        subject=match.group(1).strip()[:100],
                        predicate=predicate,
                        object=match.group(2).strip()[:300],
                        confidence=result.relevance * 0.8,
                        source_url=result.url,
                        verification_status="pending",
                    ))
                    break

        return facts

    async def _verify_fact(self, fact: VerifiedFact) -> VerifiedFact:
        """Verify a fact against existing knowledge."""
        if not self._store:
            fact.verification_status = "new"
            return fact

        try:
            # Check for existing facts with same subject+predicate
            existing = await self._store.find_by_triple(
                fact.subject, fact.predicate, exclude_historical=True
            )

            if not existing:
                fact.verification_status = "new"
                return fact

            # Check for conflicts
            for ex in existing:
                if ex.object.lower().strip() == fact.object.lower().strip():
                    fact.verification_status = "duplicate"
                    return fact

                # Simple conflict detection: same subject+predicate, different object
                fact.verification_status = "conflicting"
                fact.conflict_details = f"Existing: {ex.object[:100]}. New: {fact.object[:100]}"
                return fact

            fact.verification_status = "confirmed"
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            fact.verification_status = "new"

        return fact

    async def _store_verified_fact(self, fact: VerifiedFact) -> bool:
        """Store a verified fact as a knowledge atom."""
        if not self._store:
            return False

        try:
            from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
            atom = MemoryAtom(
                atom_type=AtomType.STATE,
                subject=fact.subject,
                predicate=fact.predicate,
                object=fact.object,
                confidence=fact.confidence,
                strength=fact.confidence,
                provenance=Provenance.INFERRED,
                source_user="active_learning",
                contexts=["active_learning", fact.source_url[:80]],
                graph=GraphType.SUBSTANTIATED,
            )
            await self._store.add_atom(atom)
            return True
        except Exception as e:
            logger.warning(f"Failed to store fact: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════════════
    # PIPELINE 2: HYPOTHESIS → TEST → UPDATE
    # ══════════════════════════════════════════════════════════════════════════

    async def propose_hypothesis(
        self,
        statement: str,
        domains: Optional[List[str]] = None,
        prior_confidence: float = 0.5,
        verification_strategy: str = "",
    ) -> Dict[str, Any]:
        """
        Propose a new hypothesis to track.

        Args:
            statement: The hypothesis statement
            domains: Relevant knowledge domains
            prior_confidence: Initial belief strength (0-1)
            verification_strategy: How to test this hypothesis

        Returns:
            Created hypothesis record
        """
        hyp_id = f"hyp_{uuid4().hex[:8]}"
        now = time.time()

        await self._conn.execute(
            """INSERT INTO hypotheses
               (hypothesis_id, statement, domains, prior_confidence, current_confidence,
                status, verification_strategy, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'proposed', ?, ?, ?)""",
            (hyp_id, statement[:500], json.dumps(domains or []),
             prior_confidence, prior_confidence, verification_strategy[:500], now, now),
        )
        await self._conn.commit()

        logger.info(f"Hypothesis proposed: {hyp_id} - {statement[:60]}")
        return {
            "ok": True,
            "id": hyp_id,
            "statement": statement[:100],
            "confidence": prior_confidence,
            "status": "proposed",
        }

    async def submit_evidence(
        self,
        hypothesis_id: str,
        evidence: str,
        direction: str = "for",  # "for" or "against"
        strength: float = 0.5,
        source_url: str = "",
    ) -> Dict[str, Any]:
        """
        Submit evidence for or against a hypothesis.
        Updates confidence using Bayesian-inspired update.

        Args:
            hypothesis_id: Which hypothesis
            evidence: Description of the evidence
            direction: "for" or "against"
            strength: How strong is this evidence (0-1)
            source_url: Where the evidence came from
        """
        # Get current hypothesis
        cursor = await self._conn.execute(
            "SELECT statement, current_confidence, evidence_for, evidence_against, status FROM hypotheses WHERE hypothesis_id=?",
            (hypothesis_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return {"ok": False, "err": "hypothesis not found"}

        statement, current_conf, ev_for_json, ev_against_json, status = row
        if status in ("confirmed", "refuted"):
            return {"ok": False, "err": f"hypothesis already {status}"}

        # Log evidence
        ev_id = f"ev_{uuid4().hex[:8]}"
        now = time.time()
        await self._conn.execute(
            """INSERT INTO evidence_log
               (evidence_id, hypothesis_id, evidence_type, content, source_url, direction, strength, logged_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ev_id, hypothesis_id, "observation", evidence[:500], source_url[:200],
             direction, strength, now),
        )

        # Bayesian-inspired confidence update
        # P(H|E) ∝ P(E|H) * P(H)
        # Simplified: shift confidence toward 1.0 (for) or 0.0 (against)
        # proportional to evidence strength
        update_factor = strength * 0.15  # Max 15% shift per evidence
        if direction == "for":
            new_conf = current_conf + update_factor * (1.0 - current_conf)
        else:
            new_conf = current_conf - update_factor * current_conf
        new_conf = round(max(0.01, min(0.99, new_conf)), 3)

        # Update evidence lists
        ev_for = json.loads(ev_for_json or "[]")
        ev_against = json.loads(ev_against_json or "[]")
        ev_entry = {"id": ev_id, "text": evidence[:100], "strength": strength, "at": now}
        if direction == "for":
            ev_for.append(ev_entry)
        else:
            ev_against.append(ev_entry)

        # Auto-resolve if confidence is extreme
        new_status = "testing"
        resolution = ""
        if new_conf >= 0.90 and len(ev_for) >= 3:
            new_status = "confirmed"
            resolution = f"Confirmed with {len(ev_for)} supporting evidence (confidence: {new_conf})"
        elif new_conf <= 0.10 and len(ev_against) >= 3:
            new_status = "refuted"
            resolution = f"Refuted with {len(ev_against)} counter-evidence (confidence: {new_conf})"

        await self._conn.execute(
            """UPDATE hypotheses
               SET current_confidence=?, status=?, evidence_for=?, evidence_against=?,
                   updated_at=?, resolution=?
               WHERE hypothesis_id=?""",
            (new_conf, new_status, json.dumps(ev_for), json.dumps(ev_against),
             now, resolution, hypothesis_id),
        )
        await self._conn.commit()

        return {
            "ok": True,
            "hypothesis_id": hypothesis_id,
            "direction": direction,
            "confidence": f"{current_conf:.3f}→{new_conf:.3f}",
            "status": new_status,
            "evidence_count": {"for": len(ev_for), "against": len(ev_against)},
            "resolution": resolution if resolution else None,
        }

    async def get_active_hypotheses(self, limit: int = 10) -> Dict[str, Any]:
        """Get all active (non-resolved) hypotheses."""
        cursor = await self._conn.execute(
            """SELECT hypothesis_id, statement, domains, current_confidence, status,
                      evidence_for, evidence_against, verification_strategy, created_at
               FROM hypotheses
               WHERE status IN ('proposed', 'testing')
               ORDER BY current_confidence DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()

        hypotheses = []
        for r in rows:
            ev_for = json.loads(r[5] or "[]")
            ev_against = json.loads(r[6] or "[]")
            hypotheses.append({
                "id": r[0],
                "statement": r[1][:100],
                "domains": json.loads(r[2] or "[]"),
                "confidence": r[3],
                "status": r[4],
                "evidence": {"for": len(ev_for), "against": len(ev_against)},
                "strategy": r[7][:80] if r[7] else "",
                "age_hours": round((time.time() - r[8]) / 3600, 1),
            })

        return {"ok": True, "n": len(hypotheses), "hypotheses": hypotheses}

    async def get_hypothesis(self, hypothesis_id: str) -> Dict[str, Any]:
        """Get full details of a specific hypothesis."""
        cursor = await self._conn.execute(
            """SELECT hypothesis_id, statement, domains, prior_confidence, current_confidence,
                      status, evidence_for, evidence_against, verification_strategy,
                      created_at, updated_at, resolution
               FROM hypotheses WHERE hypothesis_id=?""",
            (hypothesis_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return {"ok": False, "err": "not found"}

        return {
            "ok": True,
            "id": row[0],
            "statement": row[1],
            "domains": json.loads(row[2] or "[]"),
            "prior": row[3],
            "current": row[4],
            "status": row[5],
            "evidence_for": json.loads(row[6] or "[]"),
            "evidence_against": json.loads(row[7] or "[]"),
            "strategy": row[8],
            "created": datetime.fromtimestamp(row[9]).isoformat()[:16],
            "updated": datetime.fromtimestamp(row[10]).isoformat()[:16],
            "resolution": row[11],
        }

    async def resolve_hypothesis(
        self,
        hypothesis_id: str,
        status: str = "confirmed",
        resolution: str = "",
    ) -> Dict[str, Any]:
        """Manually resolve a hypothesis."""
        if status not in ("confirmed", "refuted", "revised"):
            return {"ok": False, "err": "status must be confirmed, refuted, or revised"}

        await self._conn.execute(
            "UPDATE hypotheses SET status=?, resolution=?, updated_at=? WHERE hypothesis_id=?",
            (status, resolution[:500], time.time(), hypothesis_id),
        )
        await self._conn.commit()

        # If confirmed, store as a belief atom
        if status == "confirmed" and self._store:
            cursor = await self._conn.execute(
                "SELECT statement, current_confidence, domains FROM hypotheses WHERE hypothesis_id=?",
                (hypothesis_id,),
            )
            row = await cursor.fetchone()
            if row:
                try:
                    from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
                    atom = MemoryAtom(
                        atom_type=AtomType.HYPOTHESIS,
                        subject=f"hypothesis:{hypothesis_id}",
                        predicate="confirmed_as",
                        object=row[0][:300],
                        confidence=row[1],
                        strength=row[1],
                        provenance=Provenance.INFERRED,
                        source_user="active_learning",
                        contexts=["confirmed_hypothesis"] + json.loads(row[2] or "[]"),
                        graph=GraphType.SUBSTANTIATED,
                    )
                    await self._store.add_atom(atom)
                except Exception as e:
                    logger.warning(f"Failed to store confirmed hypothesis: {e}")

        return {"ok": True, "id": hypothesis_id, "status": status}

    async def _check_against_hypotheses(self, facts: List[VerifiedFact]) -> List[Dict]:
        """Check verified facts against open hypotheses for matches."""
        cursor = await self._conn.execute(
            "SELECT hypothesis_id, statement, domains FROM hypotheses WHERE status IN ('proposed', 'testing')"
        )
        hypotheses = await cursor.fetchall()
        if not hypotheses:
            return []

        matches = []
        for hyp_id, statement, domains_json in hypotheses:
            hyp_words = set(statement.lower().split())
            hyp_words -= {"the", "a", "an", "is", "are", "that", "this", "of", "in", "to", "and", "or"}

            for fact in facts:
                fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
                fact_words = set(fact_text.split())
                overlap = hyp_words & fact_words
                if len(overlap) >= 3:
                    relevance = len(overlap) / max(len(hyp_words), 1)
                    if relevance >= 0.2:
                        matches.append({
                            "hypothesis_id": hyp_id,
                            "hypothesis": statement[:80],
                            "fact": f"{fact.subject}: {fact.object[:60]}",
                            "relevance": round(relevance, 2),
                        })

        return matches[:10]

    # ── Search History ───────────────────────────────────────────────────────

    async def get_search_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent search history."""
        cursor = await self._conn.execute(
            """SELECT search_id, query, source, status, results_count,
                      facts_stored, conflicts_found, created_at
               FROM active_searches ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        searches = []
        for r in rows:
            searches.append({
                "id": r[0],
                "query": r[1][:80],
                "source": r[2],
                "status": r[3],
                "results": r[4],
                "stored": r[5],
                "conflicts": r[6],
                "at": datetime.fromtimestamp(r[7]).isoformat()[:16],
            })
        return {"ok": True, "n": len(searches), "searches": searches}

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None
