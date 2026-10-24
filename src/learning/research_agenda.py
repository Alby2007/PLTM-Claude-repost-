"""
Research Agenda System

Gives PLTM directed curiosity — open questions it actively tries to answer.

Features:
- Maintain open research questions with priority/status
- Evaluate new knowledge against open questions (cheap FTS, no LLM)
- Claude suggests targeted searches to answer questions (on demand)
- Persist to SQLite for durability

Token-efficient: FTS-first matching (zero LLM cost), compact responses,
LLM only for suggest_searches when explicitly triggered.
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from src.storage.sqlite_store import SQLiteGraphStore
from src.core.models import MemoryAtom
from loguru import logger


@dataclass
class OpenQuestion:
    """A research question the system is trying to answer"""
    question_id: str
    question: str
    domains: List[str]
    priority: float  # 0-1
    status: str  # open, partially_answered, answered, closed
    evidence_for: int = 0
    evidence_against: int = 0
    created_at: str = ""
    updated_at: str = ""
    answer: str = ""


@dataclass
class AgendaMatch:
    """A match between new knowledge and an open question"""
    question_id: str
    question: str
    matched_atom_subject: str
    matched_atom_object: str
    relevance: float


class ResearchAgenda:
    """
    Directed curiosity system for PLTM.
    
    Maintains open research questions and evaluates incoming
    knowledge against them. Uses FTS for cheap matching,
    LLM only for search suggestions.
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self._initialized = False
        from src.analysis.model_router import any_llm_available
        self._llm_enabled = any_llm_available()
        
        logger.info("ResearchAgenda initialized")
    
    async def _ensure_table(self) -> None:
        """Create research_questions table if not exists"""
        if self._initialized:
            return
        
        if not self.store._conn:
            raise RuntimeError("Database not connected")
        
        await self.store._conn.execute("""
            CREATE TABLE IF NOT EXISTS research_questions (
                question_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                domains TEXT NOT NULL,
                priority REAL DEFAULT 0.5,
                status TEXT DEFAULT 'open',
                evidence_for INTEGER DEFAULT 0,
                evidence_against INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                answer TEXT DEFAULT ''
            )
        """)
        
        await self.store._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rq_status
            ON research_questions(status)
        """)
        
        await self.store._conn.commit()
        self._initialized = True
    
    async def add_question(
        self,
        question: str,
        domains: Optional[List[str]] = None,
        priority: float = 0.5,
    ) -> Dict[str, Any]:
        """Add a new research question"""
        await self._ensure_table()
        
        now = datetime.now().isoformat()
        # Stable hash — Python's hash() is salted per-process, so the same
        # question would get a different id suffix on every restart
        qhash = int.from_bytes(hashlib.sha256(question.encode()).digest()[:2], "little") % 10000
        qid = f"rq_{datetime.now().strftime('%Y%m%d%H%M%S')}_{qhash:04d}"
        
        await self.store._conn.execute(
            """INSERT OR REPLACE INTO research_questions
            (question_id, question, domains, priority, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'open', ?, ?)""",
            (qid, question[:500], json.dumps(domains or []), priority, now, now)
        )
        await self.store._conn.commit()
        
        logger.info(f"Added research question: {qid} - {question[:60]}")
        return {"ok": True, "id": qid, "q": question[:80]}
    
    async def get_active_questions(self, limit: int = 10) -> Dict[str, Any]:
        """Get active (open/partially_answered) research questions"""
        await self._ensure_table()
        
        cursor = await self.store._conn.execute(
            """SELECT question_id, question, domains, priority, status,
                      evidence_for, evidence_against, created_at
            FROM research_questions
            WHERE status IN ('open', 'partially_answered')
            ORDER BY priority DESC
            LIMIT ?""",
            (limit,)
        )
        rows = await cursor.fetchall()
        
        questions = []
        for r in rows:
            questions.append({
                "id": r[0],
                "q": r[1][:100],
                "d": json.loads(r[2]) if r[2] else [],
                "p": r[3],
                "s": r[4],
                "ev+": r[5],
                "ev-": r[6],
            })
        
        return {"ok": True, "n": len(questions), "questions": questions}
    
    async def evaluate_against_agenda(
        self, new_atoms: List[MemoryAtom]
    ) -> List[AgendaMatch]:
        """
        Check if newly learned facts answer any open questions.
        
        Uses keyword/FTS matching — zero LLM cost.
        """
        await self._ensure_table()
        
        cursor = await self.store._conn.execute(
            """SELECT question_id, question, domains, priority
            FROM research_questions
            WHERE status IN ('open', 'partially_answered')"""
        )
        questions = await cursor.fetchall()
        
        if not questions:
            return []
        
        matches = []
        
        for q_row in questions:
            qid, q_text, q_domains, q_priority = q_row
            q_words = set(q_text.lower().split())
            # Remove common words
            q_words -= {"what", "why", "how", "is", "the", "a", "an", "of", "in", "to", "and", "or", "does", "can", "are", "do"}
            
            if len(q_words) < 2:
                continue
            
            for atom in new_atoms:
                atom_text = f"{atom.subject} {atom.predicate} {atom.object}".lower()
                atom_words = set(atom_text.split())
                
                # Count keyword overlap
                overlap = q_words & atom_words
                if len(overlap) >= 2:
                    relevance = len(overlap) / len(q_words)
                    if relevance >= 0.3:
                        matches.append(AgendaMatch(
                            question_id=qid,
                            question=q_text[:100],
                            matched_atom_subject=atom.subject[:50],
                            matched_atom_object=atom.object[:80],
                            relevance=round(relevance, 2),
                        ))
                        
                        # Update evidence count
                        await self.store._conn.execute(
                            "UPDATE research_questions SET evidence_for = evidence_for + 1, updated_at = ? WHERE question_id = ?",
                            (datetime.now().isoformat(), qid)
                        )
        
        if matches:
            await self.store._conn.commit()
            logger.info(f"Agenda evaluation: {len(matches)} matches found for {len(new_atoms)} new atoms")
        
        return matches
    
    async def suggest_searches(self, question_id: str) -> Dict[str, Any]:
        """
        Claude suggests targeted searches to answer a question.
        
        Token budget: ≤500 input, ≤256 output. Only called on demand.
        """
        if not self._llm_enabled:
            return {"ok": False, "err": "LLM disabled"}
        
        await self._ensure_table()
        
        cursor = await self.store._conn.execute(
            "SELECT question, domains FROM research_questions WHERE question_id = ?",
            (question_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return {"ok": False, "err": "Question not found"}
        
        question, domains_json = row
        domains = json.loads(domains_json) if domains_json else []
        
        prompt = f"""Suggest 5 targeted search queries to answer this research question.

QUESTION: {question[:200]}
DOMAINS: {', '.join(domains[:5]) if domains else 'general'}

Return JSON array of search queries optimized for arXiv and web search.
Format: [{{"q":"search query","src":"arxiv|web|github","why":"brief reason"}}]

Rules:
- Be specific and targeted
- Include domain-specific terminology
- Mix arxiv and web sources
- Maximum 5 suggestions"""

        try:
            import asyncio
            from src.analysis.model_router import ModelRouter
            result = await asyncio.to_thread(
                ModelRouter().call,
                prompt=prompt,
                task_type="analysis",
                temperature=0.3,
                max_tokens=256,
            )
            if not result.get("ok"):
                return {"ok": False, "err": result.get("err", "LLM call failed")[:100]}
            response = result.get("text", "")
            
            # Parse JSON array
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                suggestions = json.loads(text[start:end + 1])
            else:
                suggestions = []
            
            return {
                "ok": True,
                "qid": question_id,
                "n": len(suggestions),
                "searches": [
                    {"q": s.get("q", "")[:100], "src": s.get("src", "web"), "why": s.get("why", "")[:60]}
                    for s in suggestions[:5]
                ]
            }
        except Exception as e:
            logger.error(f"Search suggestion failed: {e}")
            return {"ok": False, "err": str(e)[:100]}
    
    async def close_question(
        self,
        question_id: str,
        answer: str = "",
        status: str = "answered",
    ) -> Dict[str, Any]:
        """Mark a question as answered/closed"""
        await self._ensure_table()
        
        await self.store._conn.execute(
            """UPDATE research_questions
            SET status = ?, answer = ?, updated_at = ?
            WHERE question_id = ?""",
            (status, answer[:500], datetime.now().isoformat(), question_id)
        )
        await self.store._conn.commit()
        
        return {"ok": True, "id": question_id, "s": status}
    
    # ========== COMPACT RESULTS ==========
    
    def matches_to_compact(self, matches: List[AgendaMatch]) -> Dict[str, Any]:
        """Token-efficient serialization"""
        return {
            "ok": True,
            "n": len(matches),
            "matches": [
                {
                    "qid": m.question_id,
                    "q": m.question[:60],
                    "atom": f"{m.matched_atom_subject} ... {m.matched_atom_object[:40]}",
                    "rel": m.relevance,
                }
                for m in matches[:10]
            ],
        }
