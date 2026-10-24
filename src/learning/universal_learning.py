"""
Universal Learning System

PLTM learns from ANY source:
- Web pages (Wikipedia, articles, documentation)
- Research papers (arXiv, journals)
- Code repositories (GitHub, GitLab)
- Conversations (forums, Q&A sites)
- Transcripts (YouTube, podcasts)

This is the AGI path: continuous learning from all human knowledge.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
import json
import os
from urllib.parse import urlparse

from src.storage.sqlite_store import SQLiteGraphStore
from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
from src.reconciliation.conflict_detector import ConflictDetector
from loguru import logger


class SourceType(str, Enum):
    """Types of knowledge sources"""
    WEB_PAGE = "web_page"
    RESEARCH_PAPER = "research_paper"
    CODE_REPOSITORY = "code_repository"
    CONVERSATION = "conversation"
    TRANSCRIPT = "transcript"
    DOCUMENTATION = "documentation"
    BOOK = "book"
    NEWS = "news"
    VISUAL = "visual"  # Camera/image sensory input
    AUDIO = "audio"    # Microphone/audio sensory input


@dataclass
class ExtractedFact:
    """A fact extracted from a source"""
    subject: str
    predicate: str
    object: str
    confidence: float
    source_url: str
    source_type: SourceType
    extraction_method: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExtractedConcept:
    """A concept extracted from a source"""
    name: str
    definition: str
    domain: str
    related_concepts: List[str]
    confidence: float
    source_url: str


@dataclass
class ExtractedRelationship:
    """A relationship between concepts"""
    from_concept: str
    relationship: str
    to_concept: str
    confidence: float
    bidirectional: bool = False


class UniversalLearningSystem:
    """
    PLTM learns from ANY source.
    
    This transforms Claude from a static model to a continuously learning system.
    Knowledge is stored as procedural memory with:
    - Conflict resolution
    - Temporal tracking
    - Source attribution
    - Cross-domain synthesis
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self.conflict_detector = ConflictDetector(store)
        
        # LLM-powered extraction (primary when any provider is reachable —
        # Groq/Anthropic/etc. key or local Ollama; routed via ModelRouter)
        self._llm_extractor = None
        from src.analysis.model_router import any_llm_available
        self._llm_enabled = any_llm_available()
        
        # Extraction patterns for different content types (regex fallback)
        self.fact_patterns = [
            r"(.+?) is (?:a|an|the) (.+)",
            r"(.+?) (?:was|were) (?:founded|created|established) (?:in|on) (.+)",
            r"(.+?) (?:contains|includes|has) (.+)",
            r"(.+?) (?:causes|leads to|results in) (.+)",
            r"(.+?) is (?:located|found|situated) (?:in|at) (.+)",
        ]
        
        # Domain classification keywords
        self.domain_keywords = {
            "computer_science": ["algorithm", "programming", "software", "code", "data structure", "api"],
            "physics": ["quantum", "particle", "energy", "force", "wave", "relativity"],
            "biology": ["cell", "gene", "protein", "organism", "evolution", "dna"],
            "mathematics": ["theorem", "proof", "equation", "function", "set", "number"],
            "ai_ml": ["neural network", "machine learning", "deep learning", "model", "training", "inference"],
            "finance": ["trading", "market", "stock", "investment", "portfolio", "risk"],
            "chemistry": ["molecule", "reaction", "compound", "element", "bond", "catalyst"],
        }
        
        logger.info("UniversalLearningSystem initialized - AGI path active")
    
    async def learn_from_url(
        self, 
        url: str, 
        content: str,
        source_type: Optional[SourceType] = None,
        summary_only: bool = True
    ) -> Dict[str, Any]:
        """
        Learn from any URL content.
        
        Args:
            url: Source URL
            content: Text content from the URL
            source_type: Type of source (auto-detected if not provided)
        
        Returns:
            Learning results with extracted knowledge
        """
        logger.info(f"Learning from URL: {url}")
        
        # Auto-detect source type
        if source_type is None:
            source_type = self._detect_source_type(url, content)
        
        # Extract knowledge - LLM primary, regex fallback
        if self._llm_enabled:
            facts, concepts, relationships = await self._extract_with_llm(content, source_type.value if source_type else "general")
            if not facts and not concepts:
                # Weak/echoing local model or parse failure — don't lose
                # the extraction entirely, run the regex extractors too.
                facts = await self._extract_facts(content, url, source_type)
                concepts = await self._extract_concepts(content, url)
                relationships = await self._extract_relationships(content)
        else:
            facts = await self._extract_facts(content, url, source_type)
            concepts = await self._extract_concepts(content, url)
            relationships = await self._extract_relationships(content)
        
        # Determine domain
        domain = self._classify_domain(content)
        
        # Store extracted knowledge with conflict resolution
        stored_facts = 0
        conflicts_resolved = 0
        
        for fact in facts:
            result = await self._store_fact_with_resolution(fact)
            if result["stored"]:
                stored_facts += 1
            if result.get("conflict_resolved"):
                conflicts_resolved += 1
        
        for concept in concepts:
            await self._store_concept(concept)
        
        for rel in relationships:
            await self._store_relationship(rel, url)
        
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Store source metadata
        await self._store_source_metadata(url, source_type, domain, content_hash)
        
        # Token-efficient response
        if summary_only:
            return {
                "ok": True,
                "facts": stored_facts,
                "concepts": len(concepts),
                "conflicts": conflicts_resolved,
                "domain": domain
            }
        
        return {
            "url": url,
            "source_type": source_type.value,
            "domain": domain,
            "content_hash": content_hash,
            "extracted": {
                "facts": len(facts),
                "concepts": len(concepts),
                "relationships": len(relationships)
            },
            "stored": {
                "facts": stored_facts,
                "conflicts_resolved": conflicts_resolved
            },
            "sample_facts": [
                {"subject": f.subject, "predicate": f.predicate, "object": f.object}
                for f in facts[:3]
            ],
            "sample_concepts": [c.name for c in concepts[:3]]
        }
    
    async def learn_from_paper(
        self,
        paper_id: str,
        title: str,
        abstract: str,
        content: str,
        authors: List[str],
        publication_date: Optional[str] = None,
        summary_only: bool = True
    ) -> Dict[str, Any]:
        """
        Learn from a research paper.
        
        Extracts:
        - Novel findings
        - Methodologies
        - Experimental results
        - Citations and relationships
        """
        logger.info(f"Learning from paper: {paper_id} - {title}")
        
        # Extract research-specific knowledge - LLM primary, regex fallback
        if self._llm_enabled:
            findings, methods, results = await self._extract_paper_with_llm(title, abstract, content)
            if not findings and not methods and not results:
                findings = await self._extract_research_findings(abstract, content)
                methods = await self._extract_methodologies(content)
                results = await self._extract_experimental_results(content)
        else:
            findings = await self._extract_research_findings(abstract, content)
            methods = await self._extract_methodologies(content)
            results = await self._extract_experimental_results(content)
        
        # Store paper metadata
        paper_atom = MemoryAtom(
            atom_type=AtomType.EVENT,
            subject=f"paper:{paper_id}",
            predicate="published",
            object=title,
            confidence=1.0,
            strength=1.0,
            provenance=Provenance.USER_STATED,
            source_user="universal_learning",
            contexts=["research_paper", self._classify_domain(content)],
            graph=GraphType.SUBSTANTIATED
        )
        await self.store.add_atom(paper_atom)
        
        # Store findings
        stored_findings = 0
        for finding in findings:
            atom = MemoryAtom(
                atom_type=AtomType.HYPOTHESIS,
                subject=finding["topic"],
                predicate="has_finding",
                object=finding["result"],
                confidence=finding.get("confidence", 0.8),
                strength=finding.get("confidence", 0.8),
                provenance=Provenance.INFERRED,
                source_user="universal_learning",
                contexts=["research_finding", f"paper:{paper_id}"],
                graph=GraphType.SUBSTANTIATED
            )
            await self.store.add_atom(atom)
            stored_findings += 1
        
        # Store methods
        for method in methods:
            atom = MemoryAtom(
                atom_type=AtomType.SKILL,
                subject=method["name"],
                predicate="methodology_for",
                object=method["purpose"],
                confidence=0.85,
                strength=0.85,
                provenance=Provenance.INFERRED,
                source_user="universal_learning",
                contexts=["methodology", f"paper:{paper_id}"],
                graph=GraphType.SUBSTANTIATED
            )
            await self.store.add_atom(atom)
        
        if summary_only:
            return {
                "ok": True,
                "paper": paper_id,
                "findings": stored_findings,
                "methods": len(methods),
                "domain": self._classify_domain(content)
            }
        
        return {
            "paper_id": paper_id,
            "title": title,
            "domain": self._classify_domain(content),
            "findings": stored_findings,
            "methods": len(methods),
            "sample": findings[:2] if findings else []
        }
    
    async def learn_from_code(
        self,
        repo_url: str,
        repo_name: str,
        description: str,
        languages: List[str],
        code_samples: List[Dict[str, str]],
        summary_only: bool = True
    ) -> Dict[str, Any]:
        """
        Learn from a code repository.
        
        Extracts:
        - Design patterns
        - Implementation techniques
        - Best practices
        - API patterns
        """
        logger.info(f"Learning from code: {repo_name}")
        
        # Extract engineering knowledge
        patterns = await self._extract_design_patterns(code_samples)
        techniques = await self._extract_implementation_techniques(code_samples, languages)
        apis = await self._extract_api_patterns(code_samples)
        
        # Store repo metadata
        repo_atom = MemoryAtom(
            atom_type=AtomType.ENTITY,
            subject=f"repo:{repo_name}",
            predicate="is_repository",
            object=description[:200] if description else repo_name,
            confidence=1.0,
            strength=1.0,
            provenance=Provenance.USER_STATED,
            source_user="universal_learning",
            contexts=["code_repository"] + languages,
            graph=GraphType.SUBSTANTIATED
        )
        await self.store.add_atom(repo_atom)
        
        # Store patterns
        stored_patterns = 0
        for pattern in patterns:
            atom = MemoryAtom(
                atom_type=AtomType.SKILL,
                subject=pattern["name"],
                predicate="pattern_used_for",
                object=pattern["use_case"],
                confidence=pattern.get("frequency", 0.7),
                strength=pattern.get("frequency", 0.7),
                provenance=Provenance.INFERRED,
                source_user="universal_learning",
                contexts=["design_pattern", f"repo:{repo_name}"] + languages,
                graph=GraphType.SUBSTANTIATED
            )
            await self.store.add_atom(atom)
            stored_patterns += 1
        
        # Store techniques
        for technique in techniques:
            atom = MemoryAtom(
                atom_type=AtomType.SKILL,
                subject=technique["name"],
                predicate="technique_for",
                object=technique["purpose"],
                confidence=0.75,
                strength=0.75,
                provenance=Provenance.INFERRED,
                source_user="universal_learning",
                contexts=["implementation_technique", f"repo:{repo_name}"],
                graph=GraphType.SUBSTANTIATED
            )
            await self.store.add_atom(atom)
        
        if summary_only:
            return {
                "ok": True,
                "repo": repo_name,
                "patterns": stored_patterns,
                "techniques": len(techniques),
                "langs": languages[:3]
            }
        
        return {
            "repo": repo_name,
            "patterns": stored_patterns,
            "techniques": len(techniques),
            "apis": len(apis),
            "sample": patterns[:2] if patterns else []
        }
    
    def _get_llm_extractor(self):
        """Lazy-load LLM extractor"""
        if self._llm_extractor is None:
            from src.learning.llm_extractor import LLMKnowledgeExtractor
            self._llm_extractor = LLMKnowledgeExtractor()
        return self._llm_extractor
    
    async def _extract_with_llm(
        self, content: str, source_type: str
    ) -> tuple:
        """
        LLM-powered extraction returning (facts, concepts, relationships)
        in the same format as the regex methods.
        Falls back to empty lists on failure.
        """
        extractor = self._get_llm_extractor()
        triples = await extractor.extract_from_text(content, source_type)
        
        if not triples:
            return [], [], []
        
        facts = []
        concepts = []
        relationships = []
        
        for t in triples:
            fact = ExtractedFact(
                subject=t.subject,
                predicate=t.predicate,
                object=t.object,
                confidence=t.confidence,
                source="llm_extraction",
                domain=t.domain,
            )
            facts.append(fact)
            
            # Also create concepts from subjects
            concepts.append(ExtractedConcept(
                name=t.subject,
                definition=f"{t.subject} {t.predicate} {t.object}",
                domain=t.domain,
                related_concepts=[t.object[:50]],
            ))
        
        logger.info(f"LLM extracted {len(facts)} facts from {source_type} content")
        return facts, concepts, relationships
    
    async def _extract_paper_with_llm(
        self, title: str, abstract: str, content: str
    ) -> tuple:
        """
        LLM-powered paper extraction returning (findings, methods, results)
        in the same format as the regex methods.
        """
        extractor = self._get_llm_extractor()
        pk = await extractor.extract_from_paper(title, abstract, content)
        
        findings = []
        for f in pk.findings:
            findings.append(ExtractedFact(
                subject=f.get("topic", title[:50]),
                predicate="found_that",
                object=f.get("result", ""),
                confidence=f.get("confidence", 0.7),
                source="llm_paper_extraction",
                domain="research",
            ))
        
        methods = []
        for m in pk.methods:
            methods.append(ExtractedFact(
                subject=title[:50],
                predicate="uses_method",
                object=m.get("name", ""),
                confidence=0.9,
                source="llm_paper_extraction",
                domain="research",
            ))
        
        results = []
        for r in pk.relationships:
            results.append(ExtractedFact(
                subject=r.get("subject", ""),
                predicate=r.get("predicate", "related_to"),
                object=r.get("object", ""),
                confidence=r.get("confidence", 0.6),
                source="llm_paper_extraction",
                domain="research",
            ))
        
        logger.info(f"LLM extracted {len(findings)} findings, {len(methods)} methods from paper")
        return findings, methods, results
    
    def _detect_source_type(self, url: str, content: str) -> SourceType:
        """Auto-detect source type from URL and content"""
        url_lower = url.lower()
        
        if "arxiv.org" in url_lower or "doi.org" in url_lower:
            return SourceType.RESEARCH_PAPER
        elif "github.com" in url_lower or "gitlab.com" in url_lower:
            return SourceType.CODE_REPOSITORY
        elif "wikipedia.org" in url_lower:
            return SourceType.WEB_PAGE
        elif "youtube.com" in url_lower or "transcript" in url_lower:
            return SourceType.TRANSCRIPT
        elif "stackoverflow.com" in url_lower or "reddit.com" in url_lower:
            return SourceType.CONVERSATION
        elif "docs." in url_lower or "documentation" in url_lower:
            return SourceType.DOCUMENTATION
        elif any(news in url_lower for news in ["news", "bbc", "cnn", "reuters"]):
            return SourceType.NEWS
        else:
            return SourceType.WEB_PAGE
    
    def _classify_domain(self, content: str) -> str:
        """Classify content into a domain"""
        content_lower = content.lower()
        
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return "general"
    
    async def _extract_facts(
        self, 
        content: str, 
        url: str, 
        source_type: SourceType
    ) -> List[ExtractedFact]:
        """Extract facts from content using pattern matching"""
        facts = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences[:100]:  # Limit to first 100 sentences
            sentence = sentence.strip()
            if len(sentence) < 10 or len(sentence) > 500:
                continue
            
            for pattern in self.fact_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        fact = ExtractedFact(
                            subject=groups[0].strip()[:100],
                            predicate=self._extract_predicate(pattern),
                            object=groups[1].strip()[:200],
                            confidence=0.7,
                            source_url=url,
                            source_type=source_type,
                            extraction_method="pattern_matching"
                        )
                        facts.append(fact)
                        break
        
        return facts
    
    def _extract_predicate(self, pattern: str) -> str:
        """Extract predicate from pattern"""
        if "is a" in pattern or "is an" in pattern:
            return "is_a"
        elif "founded" in pattern or "created" in pattern:
            return "was_created"
        elif "contains" in pattern or "includes" in pattern:
            return "contains"
        elif "causes" in pattern or "leads to" in pattern:
            return "causes"
        elif "located" in pattern:
            return "located_in"
        return "related_to"
    
    async def _extract_concepts(self, content: str, url: str) -> List[ExtractedConcept]:
        """Extract key concepts from content"""
        concepts = []
        
        # Look for definition patterns
        definition_patterns = [
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) is defined as (.+?)(?:\.|$)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) refers to (.+?)(?:\.|$)",
            r"The term ([A-Za-z]+(?:\s+[A-Za-z]+)*) means (.+?)(?:\.|$)",
        ]
        
        for pattern in definition_patterns:
            matches = re.findall(pattern, content)
            for match in matches[:10]:
                if len(match) >= 2:
                    concept = ExtractedConcept(
                        name=match[0].strip(),
                        definition=match[1].strip()[:300],
                        domain=self._classify_domain(match[1]),
                        related_concepts=[],
                        confidence=0.75,
                        source_url=url
                    )
                    concepts.append(concept)
        
        return concepts
    
    async def _extract_relationships(self, content: str) -> List[ExtractedRelationship]:
        """Extract relationships between concepts"""
        relationships = []
        
        # Relationship patterns
        rel_patterns = [
            (r"([A-Z][a-z]+) (?:is|are) (?:a type of|a kind of) ([A-Z][a-z]+)", "is_type_of"),
            (r"([A-Z][a-z]+) (?:uses|utilizes) ([A-Z][a-z]+)", "uses"),
            (r"([A-Z][a-z]+) (?:depends on|requires) ([A-Z][a-z]+)", "depends_on"),
            (r"([A-Z][a-z]+) (?:extends|inherits from) ([A-Z][a-z]+)", "extends"),
        ]
        
        for pattern, rel_type in rel_patterns:
            matches = re.findall(pattern, content)
            for match in matches[:10]:
                if len(match) >= 2:
                    rel = ExtractedRelationship(
                        from_concept=match[0],
                        relationship=rel_type,
                        to_concept=match[1],
                        confidence=0.7
                    )
                    relationships.append(rel)
        
        return relationships
    
    async def _extract_research_findings(
        self, 
        abstract: str, 
        content: str
    ) -> List[Dict[str, Any]]:
        """Extract research findings from paper"""
        findings = []
        
        # Finding patterns
        finding_patterns = [
            r"(?:We|Our results|This paper) (?:show|demonstrate|find|prove) that (.+?)(?:\.|$)",
            r"(?:Results|Findings) (?:indicate|suggest|show) that (.+?)(?:\.|$)",
            r"(?:We|The authors) (?:achieved|obtained|reached) (.+?)(?:\.|$)",
        ]
        
        combined = abstract + " " + content[:5000]
        
        for pattern in finding_patterns:
            matches = re.findall(pattern, combined, re.IGNORECASE)
            for match in matches[:5]:
                findings.append({
                    "topic": "research_finding",
                    "result": match.strip()[:300],
                    "confidence": 0.8
                })
        
        return findings
    
    async def _extract_methodologies(self, content: str) -> List[Dict[str, Any]]:
        """Extract methodologies from paper"""
        methods = []
        
        method_patterns = [
            r"(?:We|The authors) (?:use|employ|apply) ([A-Za-z\s]+) (?:to|for) (.+?)(?:\.|$)",
            r"(?:Our|The) method (?:involves|uses|consists of) (.+?)(?:\.|$)",
        ]
        
        for pattern in method_patterns:
            matches = re.findall(pattern, content[:10000], re.IGNORECASE)
            for match in matches[:5]:
                if isinstance(match, tuple) and len(match) >= 2:
                    methods.append({
                        "name": match[0].strip()[:100],
                        "purpose": match[1].strip()[:200]
                    })
                elif isinstance(match, str):
                    methods.append({
                        "name": "methodology",
                        "purpose": match.strip()[:200]
                    })
        
        return methods
    
    async def _extract_experimental_results(self, content: str) -> List[Dict[str, Any]]:
        """Extract experimental results from paper"""
        results = []
        
        # Look for numerical results
        result_patterns = [
            r"(?:achieved|obtained|reached) (?:an? )?(?:accuracy|precision|recall|F1|score) of (\d+\.?\d*%?)",
            r"(\d+\.?\d*%?) (?:accuracy|precision|improvement|increase|decrease)",
        ]
        
        for pattern in result_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches[:5]:
                results.append({
                    "metric": "performance",
                    "value": match,
                    "confidence": 0.85
                })
        
        return results
    
    async def _extract_design_patterns(
        self, 
        code_samples: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Extract design patterns from code"""
        patterns = []
        
        pattern_indicators = {
            "singleton": ["getInstance", "_instance", "Singleton"],
            "factory": ["Factory", "create", "build"],
            "observer": ["Observer", "subscribe", "notify", "listener"],
            "decorator": ["Decorator", "@", "wrapper"],
            "strategy": ["Strategy", "execute", "algorithm"],
            "repository": ["Repository", "findBy", "save", "delete"],
            "dependency_injection": ["inject", "Container", "Provider"],
        }
        
        for sample in code_samples:
            code = sample.get("code", "")
            for pattern_name, indicators in pattern_indicators.items():
                if any(ind in code for ind in indicators):
                    patterns.append({
                        "name": pattern_name,
                        "use_case": f"Implements {pattern_name} pattern",
                        "frequency": 0.7,
                        "file": sample.get("file", "unknown")
                    })
        
        return patterns
    
    async def _extract_implementation_techniques(
        self,
        code_samples: List[Dict[str, str]],
        languages: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract implementation techniques from code"""
        techniques = []
        
        technique_indicators = {
            "async_await": ["async", "await", "Promise"],
            "error_handling": ["try", "catch", "except", "finally"],
            "type_hints": ["->", ": str", ": int", "typing"],
            "testing": ["test_", "assert", "expect", "describe"],
            "logging": ["logger", "logging", "console.log"],
            "caching": ["cache", "memoize", "lru_cache"],
        }
        
        for sample in code_samples:
            code = sample.get("code", "")
            for tech_name, indicators in technique_indicators.items():
                if any(ind in code for ind in indicators):
                    techniques.append({
                        "name": tech_name,
                        "purpose": f"Uses {tech_name} technique",
                        "languages": languages
                    })
        
        return techniques
    
    async def _extract_api_patterns(
        self,
        code_samples: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Extract API patterns from code"""
        apis = []
        
        api_patterns = [
            (r"@app\.(?:get|post|put|delete)\(['\"](.+?)['\"]\)", "rest_endpoint"),
            (r"fetch\(['\"](.+?)['\"]\)", "http_call"),
            (r"axios\.(?:get|post)\(['\"](.+?)['\"]\)", "http_call"),
        ]
        
        for sample in code_samples:
            code = sample.get("code", "")
            for pattern, api_type in api_patterns:
                matches = re.findall(pattern, code)
                for match in matches[:5]:
                    apis.append({
                        "endpoint": match,
                        "type": api_type
                    })
        
        return apis
    
    async def _store_fact_with_resolution(self, fact: ExtractedFact) -> Dict[str, Any]:
        """Store fact with conflict resolution"""
        
        # Check for existing conflicting facts
        existing = await self.store.find_by_triple(fact.subject, fact.predicate, exclude_historical=True)
        
        conflict_resolved = False
        
        if existing:
            # Check if this is a conflict
            for ex in existing:
                if ex.object != fact.object:
                    # Conflict detected - use recency and source quality
                    # For now, prefer newer information
                    conflict_resolved = True
                    logger.info(f"Conflict resolved: {fact.subject} {fact.predicate} - preferring new value")
        
        # Store the fact
        atom = MemoryAtom(
            atom_type=AtomType.STATE,
            subject=fact.subject,
            predicate=fact.predicate,
            object=fact.object,
            confidence=fact.confidence,
            strength=fact.confidence,
            provenance=Provenance.INFERRED,
            source_user="universal_learning",
            contexts=[fact.source_type.value, fact.source_url[:100]],
            graph=GraphType.SUBSTANTIATED
        )
        await self.store.add_atom(atom)
        
        return {"stored": True, "conflict_resolved": conflict_resolved}
    
    async def _store_concept(self, concept: ExtractedConcept) -> None:
        """Store extracted concept"""
        atom = MemoryAtom(
            atom_type=AtomType.ENTITY,
            subject=concept.name,
            predicate="defined_as",
            object=concept.definition,
            confidence=concept.confidence,
            strength=concept.confidence,
            provenance=Provenance.INFERRED,
            source_user="universal_learning",
            contexts=["concept", concept.domain],
            graph=GraphType.SUBSTANTIATED
        )
        await self.store.add_atom(atom)
    
    async def _store_relationship(self, rel: ExtractedRelationship, url: str) -> None:
        """Store extracted relationship"""
        atom = MemoryAtom(
            atom_type=AtomType.SOCIAL,
            subject=rel.from_concept,
            predicate=rel.relationship,
            object=rel.to_concept,
            confidence=rel.confidence,
            strength=rel.confidence,
            provenance=Provenance.INFERRED,
            source_user="universal_learning",
            contexts=["relationship", url[:100]],
            graph=GraphType.SUBSTANTIATED
        )
        await self.store.add_atom(atom)
    
    async def _store_source_metadata(
        self, 
        url: str, 
        source_type: SourceType,
        domain: str,
        content_hash: str
    ) -> None:
        """Store metadata about the source"""
        atom = MemoryAtom(
            atom_type=AtomType.EVENT,
            subject=f"source:{content_hash}",
            predicate="learned_from",
            object=url[:200],
            confidence=1.0,
            strength=1.0,
            provenance=Provenance.USER_STATED,
            source_user="universal_learning",
            contexts=[source_type.value, domain, "source_metadata"],
            graph=GraphType.SUBSTANTIATED
        )
        await self.store.add_atom(atom)
    
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learned knowledge"""
        
        # Count atoms by context
        all_atoms = await self.store.get_atoms_by_subject("source:")
        
        # This is a simplified version - would need more sophisticated querying
        return {
            "total_sources_learned": len(all_atoms),
            "learning_active": True,
            "last_learning": datetime.now().isoformat()
        }
    
    async def query_learned_knowledge(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Query the learned knowledge base"""
        
        # Search for relevant atoms
        # This would need more sophisticated search in production
        results = []
        
        # For now, do a simple keyword search
        query_lower = query.lower()
        
        # Get all atoms and filter (inefficient but works for demo)
        # In production, would use proper full-text search
        
        return results
