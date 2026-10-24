"""
LLM-Powered Knowledge Extraction

Replaces regex-based extraction with Claude-powered semantic understanding.
Extracts structured triples, findings, methods, and concepts from any text.

Token-efficient design:
- Input truncated to max_chars before sending
- Structured JSON output with minimal keys
- Tight max_tokens per call type
- Regex fallback when API key unavailable
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class ExtractedTriple:
    """A structured knowledge triple from LLM extraction"""
    subject: str
    predicate: str
    object: str
    confidence: float
    domain: str


@dataclass
class PaperKnowledge:
    """Structured knowledge extracted from a research paper"""
    findings: List[Dict[str, Any]]
    methods: List[Dict[str, Any]]
    concepts: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]


class LLMKnowledgeExtractor:
    """
    Uses Claude to extract structured knowledge from text.
    
    Token-efficient:
    - Truncates input to max_chars
    - Minimal JSON keys in output format
    - Tight max_tokens budgets
    - Falls back to empty results (not regex) when disabled
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self._enabled: Optional[bool] = None  # resolved on first use

        self.extraction_stats = {
            "llm_text": 0,
            "llm_paper": 0,
            "failed": 0,
        }

    @property
    def enabled(self) -> bool:
        """Any usable LLM provider (Groq/Anthropic/etc. or local Ollama)."""
        if self._enabled is None:
            try:
                from src.analysis.model_router import ModelRouter
                self._enabled = ModelRouter().route("extraction") is not None
            except Exception:
                self._enabled = False
            if not self._enabled:
                logger.warning("No LLM provider available - LLM extraction disabled")
        return self._enabled
    
    async def extract_from_text(
        self,
        content: str,
        source_type: str = "general",
        max_chars: int = 3000,
    ) -> List[ExtractedTriple]:
        """
        Extract structured triples from arbitrary text.
        
        Token budget: ≤3K input, ≤500 output
        """
        if not self.enabled:
            return []
        
        # Truncate input for token efficiency
        truncated = content[:max_chars]
        
        prompt = f"""Extract structured knowledge triples from this text.

TEXT ({source_type}):
{truncated}

Return a JSON array of triples.
Format: [{{"s":"subject","p":"predicate","o":"object","c":0.0-1.0,"d":"domain"}}]

Rules:
- s: the entity or concept (noun phrase)
- p: the relationship (verb phrase, e.g., "is_a", "causes", "contains", "located_in")
- o: the target entity/value
- c: confidence 0-1
- d: domain (e.g., "physics", "biology", "computer_science", "economics")
- Maximum 15 triples
- Focus on the most important/novel facts
- Skip trivial or obvious statements"""

        response = await self._call_llm(prompt, max_tokens=500)
        triples = self._parse_triples(response)
        
        if triples:
            self.extraction_stats["llm_text"] += 1
        else:
            self.extraction_stats["failed"] += 1
        
        return triples
    
    async def extract_from_paper(
        self,
        title: str,
        abstract: str,
        content: str = "",
        max_content_chars: int = 2000,
    ) -> PaperKnowledge:
        """
        Extract structured knowledge from a research paper.
        
        Token budget: ≤3K input, ≤800 output
        """
        if not self.enabled:
            return PaperKnowledge([], [], [], [])
        
        # Build compact input: title + abstract + truncated content
        content_snippet = content[:max_content_chars] if content else ""
        
        paper_text = f"TITLE: {title[:200]}\nABSTRACT: {abstract[:500]}"
        if content_snippet:
            paper_text += f"\nCONTENT: {content_snippet}"
        
        prompt = f"""Extract structured knowledge from this research paper.

{paper_text}

Return JSON with these sections:
{{
  "findings": [{{"topic":"...","result":"...","c":0.0-1.0}}],
  "methods": [{{"name":"...","purpose":"..."}}],
  "concepts": [{{"name":"...","d":"domain","related":["concept1","concept2"]}}],
  "relationships": [{{"s":"subject","p":"predicate","o":"object","c":0.0-1.0}}]
}}

Rules:
- findings: key results/discoveries (max 5)
- methods: techniques/approaches used (max 3)
- concepts: important concepts introduced or discussed (max 5)
- relationships: connections between concepts (max 8)
- Be specific, not generic
- c = confidence 0-1"""

        response = await self._call_llm(prompt, max_tokens=800)
        paper_knowledge = self._parse_paper_knowledge(response)
        
        if paper_knowledge.findings or paper_knowledge.concepts:
            self.extraction_stats["llm_paper"] += 1
        else:
            self.extraction_stats["failed"] += 1
        
        return paper_knowledge
    
    # ========== LLM CALL ==========
    
    def _router(self):
        if getattr(self, "_model_router", None) is None:
            from src.analysis.model_router import ModelRouter
            self._model_router = ModelRouter()
        return self._model_router

    async def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Call via ModelRouter — Groq first (free tier), provider fallback."""
        try:
            import asyncio

            result = await asyncio.to_thread(
                self._router().call,
                prompt=prompt,
                task_type="extraction",
                temperature=0.0,
                max_tokens=max_tokens,
            )
            if result.get("ok"):
                return result.get("text", "{}")
            logger.error(f"LLM extraction call failed: {result.get('err', '?')}")
            return "{}"
        except Exception as e:
            logger.error(f"LLM extraction call failed: {e}")
            return "{}"
    
    # ========== PARSERS ==========
    
    def _parse_triples(self, response: str) -> List[ExtractedTriple]:
        """Parse LLM JSON response into ExtractedTriple objects"""
        try:
            data = self._extract_json_array(response)
            triples = []
            for item in data[:15]:
                triples.append(ExtractedTriple(
                    subject=str(item.get("s", ""))[:100],
                    predicate=str(item.get("p", ""))[:50],
                    object=str(item.get("o", ""))[:200],
                    confidence=float(item.get("c", 0.5)),
                    domain=str(item.get("d", "general"))[:30],
                ))
            return triples
        except Exception as e:
            logger.warning(f"Failed to parse triples: {e}")
            return []
    
    def _parse_paper_knowledge(self, response: str) -> PaperKnowledge:
        """Parse LLM JSON response into PaperKnowledge"""
        try:
            data = self._extract_json_object(response)
            return PaperKnowledge(
                findings=[
                    {"topic": str(f.get("topic", ""))[:100],
                     "result": str(f.get("result", ""))[:200],
                     "confidence": float(f.get("c", 0.5))}
                    for f in data.get("findings", [])[:5]
                ],
                methods=[
                    {"name": str(m.get("name", ""))[:100],
                     "purpose": str(m.get("purpose", ""))[:200]}
                    for m in data.get("methods", [])[:3]
                ],
                concepts=[
                    {"name": str(c.get("name", ""))[:100],
                     "domain": str(c.get("d", "general"))[:30],
                     "related": [str(r)[:50] for r in c.get("related", [])[:5]]}
                    for c in data.get("concepts", [])[:5]
                ],
                relationships=[
                    {"subject": str(r.get("s", ""))[:100],
                     "predicate": str(r.get("p", ""))[:50],
                     "object": str(r.get("o", ""))[:200],
                     "confidence": float(r.get("c", 0.5))}
                    for r in data.get("relationships", [])[:8]
                ],
            )
        except Exception as e:
            logger.warning(f"Failed to parse paper knowledge: {e}")
            return PaperKnowledge([], [], [], [])
    
    def _extract_json_array(self, text: str) -> List[Dict]:
        """Extract JSON array from LLM response"""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        return []
    
    def _extract_json_object(self, text: str) -> Dict:
        """Extract JSON object from LLM response"""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        return {}
    
    # ========== COMPACT RESULTS ==========
    
    def triples_to_compact(self, triples: List[ExtractedTriple]) -> Dict[str, Any]:
        """Token-efficient serialization for MCP responses"""
        return {
            "ok": True,
            "n": len(triples),
            "triples": [
                {"s": t.subject[:50], "p": t.predicate, "o": t.object[:80], "c": round(t.confidence, 2), "d": t.domain}
                for t in triples[:15]
            ],
        }
    
    def paper_to_compact(self, pk: PaperKnowledge) -> Dict[str, Any]:
        """Token-efficient serialization for MCP responses"""
        return {
            "ok": True,
            "findings": len(pk.findings),
            "methods": len(pk.methods),
            "concepts": len(pk.concepts),
            "rels": len(pk.relationships),
            "top_findings": [f["topic"][:60] for f in pk.findings[:3]],
            "top_concepts": [c["name"][:40] for c in pk.concepts[:3]],
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return self.extraction_stats
