"""
PLTM MCP Server

Model Context Protocol server for Procedural Long-Term Memory system.
Provides tools for personality tracking, mood detection, and memory management.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.types import Tool, TextContent

from src.storage.sqlite_store import SQLiteGraphStore
from src.pipeline.memory_pipeline import MemoryPipeline
from src.personality.personality_mood_agent import PersonalityMoodAgent
from src.personality.personality_synthesizer import PersonalitySynthesizer
from src.personality.mood_tracker import MoodTracker
from src.personality.mood_patterns import MoodPatterns
from src.personality.enhanced_conflict_resolver import EnhancedConflictResolver
from src.personality.contextual_personality import ContextualPersonality
from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
from mcp_server.handlers.registry import compact_json
from loguru import logger


# Initialize PLTM system
store: Optional[SQLiteGraphStore] = None
pipeline: Optional[MemoryPipeline] = None
personality_agent: Optional[PersonalityMoodAgent] = None
personality_synth: Optional[PersonalitySynthesizer] = None
mood_tracker: Optional[MoodTracker] = None
mood_patterns: Optional[MoodPatterns] = None
conflict_resolver: Optional[EnhancedConflictResolver] = None
contextual_personality: Optional[ContextualPersonality] = None
typed_memory_store = None  # TypedMemoryStore - initialized in initialize_pltm
embedding_store = None  # EmbeddingStore - initialized in initialize_pltm
typed_memory_pipeline = None  # TypedMemoryPipeline - initialized in initialize_pltm
decay_engine = None
consolidation_engine = None
contextual_retriever = None
conflict_surfacer = None
memory_clusterer = None
shared_memory_mgr = None
memory_portability = None
provenance_tracker = None
confidence_decay_engine = None
memory_auditor = None
phi_scorer = None
criticality_pruner = None
phi_consolidator = None
phi_context_builder = None
tool_analytics = None
arch_snapshotter = None
working_memory_compressor = None
trajectory_encoder = None
handoff_protocol = None
autonomous_learning = None
active_learning = None
memory_index = None
prefetch_engine = None
contradiction_resolver = None
emergence_detector = None
_current_session_id = ""
_db_path: str = ""  # Resolved at initialize_pltm() — handlers must use _get_db_path()


def _get_db_path() -> str:
    """Return the DB path chosen at init (or the default if not yet initialized)."""
    return _db_path or str(Path(__file__).parent.parent / "data" / "pltm_mcp.db")


async def initialize_pltm(custom_db_path: str = None):
    """Initialize PLTM system components"""
    global store, pipeline, personality_agent, personality_synth
    global mood_tracker, mood_patterns, conflict_resolver, contextual_personality
    global typed_memory_store, embedding_store, typed_memory_pipeline, _db_path

    # Initialize storage (absolute path so it works regardless of cwd)
    db_path = Path(custom_db_path) if custom_db_path else Path(__file__).parent.parent / "data" / "pltm_mcp.db"
    _db_path = str(db_path)
    # Modules that resolve the DB via db_config (epistemic monitor, etc.)
    # must see the same path, not the default
    try:
        from src.db_config import set_db_path as _set_db_path
        _set_db_path(str(db_path))
    except Exception:
        pass
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteGraphStore(str(db_path))
    await store.connect()
    
    # Initialize pipeline
    pipeline = MemoryPipeline(store)
    
    # Initialize personality/mood components
    personality_agent = PersonalityMoodAgent(pipeline)
    personality_synth = PersonalitySynthesizer(store)
    mood_tracker = MoodTracker(store)
    mood_patterns = MoodPatterns(store)
    conflict_resolver = EnhancedConflictResolver(store)
    contextual_personality = ContextualPersonality(store)
    
    # Initialize embedding store + jury + typed memory store (uses same DB)
    from src.memory.memory_types import TypedMemoryStore
    from src.memory.embedding_store import EmbeddingStore
    from src.memory.memory_jury import MemoryJury
    embedding_store = EmbeddingStore(str(db_path))
    await embedding_store.connect()
    memory_jury = MemoryJury(enable_meta_judge=True, db_path=str(db_path))
    typed_memory_store = TypedMemoryStore(str(db_path), embedding_store=embedding_store, jury=memory_jury)
    await typed_memory_store.connect()
    
    # Initialize 3-lane pipeline (Extract → Jury → Reconcile → Store)
    from src.memory.memory_pipeline import TypedMemoryPipeline
    typed_memory_pipeline = TypedMemoryPipeline(typed_memory_store, embedding_store, memory_jury)
    
    # Initialize Memory Intelligence components
    from src.memory.memory_intelligence import (
        DecayEngine, ConsolidationEngine, ContextualRetriever,
        ImportanceScorer, ConflictSurfacer, MemoryClusterer,
        SharedMemoryManager, MemoryPortability, ProvenanceTracker,
        ConfidenceDecayEngine, MemoryAuditor,
    )
    global decay_engine, consolidation_engine, contextual_retriever
    global conflict_surfacer, memory_clusterer, shared_memory_mgr
    global memory_portability, provenance_tracker, confidence_decay_engine, memory_auditor
    decay_engine = DecayEngine(typed_memory_store)
    consolidation_engine = ConsolidationEngine(typed_memory_store, embedding_store)
    contextual_retriever = ContextualRetriever(typed_memory_store, embedding_store)
    conflict_surfacer = ConflictSurfacer(typed_memory_store, embedding_store)
    memory_clusterer = MemoryClusterer(typed_memory_store, embedding_store)
    shared_memory_mgr = SharedMemoryManager(typed_memory_store)
    memory_portability = MemoryPortability(typed_memory_store)
    provenance_tracker = ProvenanceTracker(typed_memory_store)
    typed_memory_store.provenance_tracker = provenance_tracker  # Wire auto-provenance
    confidence_decay_engine = ConfidenceDecayEngine(typed_memory_store, embedding_store)
    memory_auditor = MemoryAuditor(typed_memory_store, embedding_store, memory_jury)
    
    # Initialize ΦRMS components
    from src.memory.phi_rms import PhiMemoryScorer, CriticalityPruner, PhiConsolidator, PhiContextBuilder
    from src.meta.criticality import SelfOrganizedCriticality
    global phi_scorer, criticality_pruner, phi_consolidator, phi_context_builder
    phi_scorer = PhiMemoryScorer(typed_memory_store, embedding_store, str(db_path))
    await phi_scorer.connect()
    soc = SelfOrganizedCriticality(store)
    criticality_pruner = CriticalityPruner(phi_scorer, soc, typed_memory_store)
    phi_consolidator = PhiConsolidator(typed_memory_store, embedding_store, phi_scorer)
    phi_context_builder = PhiContextBuilder(phi_scorer, typed_memory_store, embedding_store)
    
    # Initialize Improvement Loop components
    from src.analysis.tool_analytics import ToolAnalytics
    from src.analysis.architecture_snapshots import ArchitectureSnapshotter
    global tool_analytics, arch_snapshotter
    tool_analytics = ToolAnalytics(db_path)
    arch_snapshotter = ArchitectureSnapshotter(db_path)
    
    # Initialize Session Continuity components
    from src.memory.session_continuity import (
        WorkingMemoryCompressor, TrajectoryEncoder, HandoffProtocol,
    )
    global working_memory_compressor, trajectory_encoder, handoff_protocol
    working_memory_compressor = WorkingMemoryCompressor(db_path)
    trajectory_encoder = TrajectoryEncoder(db_path)
    handoff_protocol = HandoffProtocol(db_path, working_memory_compressor, trajectory_encoder)
    
    # Populate shared registry for handler modules
    from mcp_server.handlers.registry import registry as _reg
    _reg.store = store
    _reg.pipeline = pipeline
    _reg.personality_agent = personality_agent
    _reg.personality_synth = personality_synth
    _reg.mood_tracker = mood_tracker
    _reg.mood_patterns = mood_patterns
    _reg.conflict_resolver = conflict_resolver
    _reg.contextual_personality = contextual_personality
    _reg.typed_memory_store = typed_memory_store
    _reg.embedding_store = embedding_store
    _reg.typed_memory_pipeline = typed_memory_pipeline
    _reg.decay_engine = decay_engine
    _reg.consolidation_engine = consolidation_engine
    _reg.contextual_retriever = contextual_retriever
    _reg.conflict_surfacer = conflict_surfacer
    _reg.memory_clusterer = memory_clusterer
    _reg.shared_memory_mgr = shared_memory_mgr
    _reg.memory_portability = memory_portability
    _reg.provenance_tracker = provenance_tracker
    _reg.confidence_decay_engine = confidence_decay_engine
    _reg.memory_auditor = memory_auditor
    _reg.phi_scorer = phi_scorer
    _reg.criticality_pruner = criticality_pruner
    _reg.phi_consolidator = phi_consolidator
    _reg.phi_context_builder = phi_context_builder
    _reg.tool_analytics = tool_analytics
    _reg.arch_snapshotter = arch_snapshotter
    _reg.working_memory_compressor = working_memory_compressor
    _reg.trajectory_encoder = trajectory_encoder
    _reg.handoff_protocol = handoff_protocol
    
    # Initialize Autonomous + Active Learning engines
    from src.learning.autonomous_learning import AutonomousLearningEngine
    from src.learning.active_learning import ActiveLearningEngine
    global autonomous_learning, active_learning
    autonomous_learning = AutonomousLearningEngine(db_path)
    await autonomous_learning.connect(store)
    active_learning = ActiveLearningEngine(db_path)
    await active_learning.connect(store)
    _reg.autonomous_learning = autonomous_learning
    _reg.active_learning = active_learning
    
    # Initialize Hierarchical Memory + Prefetch + Contradiction Resolver
    from src.memory.hierarchical_memory import (
        HierarchicalMemoryIndex, PredictivePrefetchEngine, AutoContradictionResolver,
    )
    global memory_index, prefetch_engine, contradiction_resolver
    memory_index = HierarchicalMemoryIndex(db_path)
    await memory_index.connect(store)
    prefetch_engine = PredictivePrefetchEngine(memory_index)
    prefetch_engine.set_store(store)
    contradiction_resolver = AutoContradictionResolver(db_path)
    await contradiction_resolver.connect(store)
    _reg.memory_index = memory_index
    _reg.prefetch_engine = prefetch_engine
    _reg.contradiction_resolver = contradiction_resolver
    
    # Initialize Emergence Detector (MetaCriticalityObserver)
    from src.meta.emergence_detector import MetaCriticalityObserver
    from src.analysis.phi_integration import PhiIntegrationCalculator
    from src.memory.knowledge_graph import KnowledgeNetworkGraph
    global emergence_detector, _knowledge_graph
    emergence_detector = MetaCriticalityObserver(db_path)
    # Wire the real subsystems — without these the observer silently degrades
    # to placeholder constants (r=1.0, Φ=0.0, bridges=0)
    _phi_calc = PhiIntegrationCalculator(db_path)
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeNetworkGraph(store)
    _kg = _knowledge_graph
    await emergence_detector.connect(
        store=store, criticality=soc,
        phi_calc=_phi_calc, knowledge_graph=_kg,
    )
    _reg.criticality = soc
    _reg.phi_calc = _phi_calc
    _reg.knowledge_graph = _kg
    _reg.emergence_detector = emergence_detector
    
    logger.info("PLTM MCP Server initialized (with embeddings + jury + pipeline + intelligence + ΦRMS + analytics + learning + hierarchical + emergence)")


# Create MCP server
app = Server("pltm-memory")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available PLTM tools"""
    return [
        Tool(
            name="store_memory_atom",
            description="Store a knowledge atom (subject-predicate-object triple).",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        
                    },
                    "atom_type": {
                        "type": "string",
                        "enum": ["fact", "personality_trait", "communication_style", "interaction_pattern", "mood", "preference"],
                        "description": "Type of memory atom"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Subject of the atom (usually user_id)"
                    },
                    "predicate": {
                        "type": "string",
                        "description": "Relationship/predicate (e.g., 'prefers_style', 'has_trait', 'is_feeling')"
                    },
                    "object": {
                        "type": "string",
                        "description": "Object/value (e.g., 'concise responses', 'technical depth', 'frustrated')"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "context": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Context tags (e.g., ['technical', 'work'])"
                    }
                },
                "required": ["user_id", "atom_type", "subject", "predicate", "object"]
            }
        ),
        
        Tool(
            name="query_personality",
            description="Get synthesized personality profile.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context filter (e.g., 'technical', 'casual')"
                    }
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="detect_mood",
            description="Detect mood from message text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        
                    },
                    "message": {
                        "type": "string",
                        
                    }
                },
                "required": ["user_id", "message"]
            }
        ),
        
        Tool(
            name="get_mood_patterns",
            description="Get mood patterns, volatility, and predictions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        
                    },
                    "window_days": {
                        "type": "number",
                        "description": "days to analyze, default 90",
                        "minimum": 1
                    }
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="resolve_conflict",
            description="Resolve conflicting personality traits.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        
                    },
                    "trait_objects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of conflicting trait objects (e.g., ['concise', 'detailed'])"
                    },
                    "traits": {"type": "array", "items": {"type": "string"}, "description": "Alias for trait_objects"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="extract_personality_traits",
            description="Extract personality traits from message.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        
                    },
                    "message": {
                        "type": "string",
                        
                    },
                    "ai_response": {
                        "type": "string",
                        
                    },
                    "user_reaction": {
                        "type": "string",
                        
                    }
                },
                "required": ["user_id", "message"]
            }
        ),
        
        Tool(
            name="get_adaptive_prompt",
            description="Get adaptive system prompt based on personality and mood.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        
                    },
                    "message": {
                        "type": "string",
                        
                    },
                    "context": {
                        "type": "string",
                        
                    }
                },
                "required": ["user_id", "message"]
            }
        ),
        
        Tool(
            name="get_personality_summary",
            description="Get human-readable personality summary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        
                    }
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="bootstrap_from_sample",
            description="Bootstrap PLTM with sample data for testing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        
                    }
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="bootstrap_from_messages",
            description="Bootstrap PLTM from conversation messages.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        
                    },
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"}
                            }
                        },
                        "description": "Conversation messages to analyze"
                    }
                },
                "required": ["user_id", "messages"]
            }
        ),
        
        Tool(
            name="track_trait_evolution",
            description="Track trait changes over time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "trait": {"type": "string", "description": "Trait name"},
                    "window_days": {"type": "number", "default": 90}
                },
                "required": ["user_id", "trait"]
            }
        ),
        
        Tool(
            name="predict_reaction",
            description="Predict user reaction to a stimulus.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "stimulus": {"type": "string"}
                },
                "required": ["user_id", "stimulus"]
            }
        ),
        
        Tool(
            name="get_meta_patterns",
            description="Get cross-context behavioral patterns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="learn_from_interaction",
            description="Learn from interaction outcome.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "my_response": {"type": "string", "description": "What I (AI) said"},
                    "user_reaction": {"type": "string"}
                },
                "required": ["user_id", "my_response", "user_reaction"]
            }
        ),
        
        Tool(
            name="predict_session",
            description="Predict session dynamics from greeting.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "greeting": {"type": "string", "description": "User's opening message"}
                },
                "required": ["user_id", "greeting"]
            }
        ),
        
        Tool(
            name="get_self_model",
            description="Get self-model for meta-cognition.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="init_claude_session",
            description="Initialize Claude personality session. Call at START of conversation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="update_claude_style",
            description="Update Claude's communication style attribute.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "attribute": {"type": "string", "description": "verbosity|formality|initiative|code_preference|energy_matching"},
                    "value": {"type": "string"}
                },
                "required": ["user_id", "attribute", "value"]
            }
        ),
        
        Tool(
            name="learn_interaction_dynamic",
            description="Learn what works or doesn't with this user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "behavior": {"type": "string"},
                    "works": {"type": "boolean"}
                },
                "required": ["user_id", "behavior", "works"]
            }
        ),
        
        Tool(
            name="record_milestone",
            description="Record a collaboration milestone.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "description": {"type": "string"},
                    "significance": {"type": "number", "default": 0.8}
                },
                "required": ["user_id", "description"]
            }
        ),
        
        Tool(
            name="add_shared_vocabulary",
            description="Add shared term/shorthand.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "term": {"type": "string"},
                    "meaning": {"type": "string"}
                },
                "required": ["user_id", "term", "meaning"]
            }
        ),
        
        Tool(
            name="get_claude_personality",
            description="Get Claude's personality for this user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="evolve_claude_personality",
            description="Evolve Claude personality from interaction outcome.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "my_response_style": {"type": "string", "description": "How Claude responded (e.g., 'verbose_explanation')"},
                    "user_reaction": {"type": "string", "description": "User's reaction"},
                    "was_positive": {"type": "boolean", "description": "Was the reaction positive?"}
                },
                "required": ["user_id", "my_response_style", "user_reaction", "was_positive"]
            }
        ),
        
        Tool(
            name="check_pltm_available",
            description="Check if user has PLTM data. Call FIRST to decide if to init.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="pltm_mode",
            description="Trigger phrase handler for 'PLTM mode' auto-init.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "trigger_phrase": {"type": "string"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="deep_personality_analysis",
            description="Run comprehensive personality analysis from history.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="enrich_claude_personality",
            description="Build rich Claude personality from deep analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "store_results": {"type": "boolean", "default": True}
                },
                "required": ["user_id"]
            }
        ),
        
        # === SENSORY INPUT (Multimodal Perception) ===
        Tool(
            name="sensory_start",
            description="Start camera and microphone capture. REQUIRES explicit user consent. Enables real-time multimodal perception.",
            inputSchema={
                "type": "object",
                "properties": {
                    "camera": {"type": "boolean", "description": "Enable camera (default: true)"},
                    "microphone": {"type": "boolean", "description": "Enable microphone (default: true)"},
                    "fps": {"type": "number", "description": "Camera FPS (default: 1.0)"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="sensory_stop",
            description="Stop all sensory capture (camera + microphone). Releases hardware and clears buffer.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="sensory_look",
            description="Get current visual snapshot: faces detected, emotions, scene description, objects.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="sensory_listen",
            description="Get current audio snapshot: speech transcript, tone/sentiment, ambient sound.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="sensory_state",
            description="Get unified multimodal state: current visual + audio + cross-modal correlations (e.g., face emotion + voice tone alignment).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="sensory_history",
            description="Get recent sensory timeline: last N seconds of visual and audio observations with temporal patterns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "seconds": {"type": "number", "description": "How far back to look (default: 10.0)"},
                    "modality": {"type": "string", "enum": ["visual", "audio", "both"], "description": "Filter by modality (default: both)"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="sensory_diagnostics",
            description="Get Whisper transcription diagnostics: shows recent transcription attempts with audio properties, success/failure, and actual output. Use this to debug why speech detection works but transcription is empty.",
            inputSchema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of recent attempts to show (default: 10)"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="learn_from_url",
            description="Learn from URL content. User must provide text in 'content' param.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Source URL"},
                    "content": {"type": "string", "description": "Actual text content from URL (user must provide)"},
                    "source_type": {"type": "string", "description": "web_page|research_paper|code_repository|conversation|transcript"}
                },
                "required": ["url", "content"]
            }
        ),
        
        Tool(
            name="learn_from_paper",
            description="Learn from a research paper. Extracts findings and methods.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Paper ID"},
                    "title": {"type": "string"},
                    "abstract": {"type": "string"},
                    "content": {"type": "string"},
                    "authors": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["paper_id", "title", "abstract", "content", "authors"]
            }
        ),
        
        Tool(
            name="learn_from_code",
            description="Learn from a code repository. Extracts patterns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string"},
                    "repo_name": {"type": "string"},
                    "description": {"type": "string"},
                    "languages": {"type": "array", "items": {"type": "string"}},
                    "code_samples": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["repo_url", "repo_name", "languages", "code_samples"]
            }
        ),
        
        Tool(
            name="get_learning_stats",
            description="Get learned knowledge statistics.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="batch_ingest_wikipedia",
            description="Batch ingest Wikipedia articles. Pass 'titles' (list of title strings) OR 'articles' (list of {title, content, url} objects).",
            inputSchema={
                "type": "object",
                "properties": {
                    "titles": {"type": "array", "items": {"type": "string"}, "description": "List of Wikipedia article titles to fetch and ingest"},
                    "articles": {"type": "array", "items": {"type": "object"}, "description": "Pre-fetched articles with {title, content, url}"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="batch_ingest_papers",
            description="Batch ingest arXiv papers. Pass 'arxiv_ids' (list of ID strings) OR 'papers' (list of {arxiv_id, title, abstract} objects).",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_ids": {"type": "array", "items": {"type": "string"}, "description": "List of arXiv IDs like '2301.04104'"},
                    "papers": {"type": "array", "items": {"type": "object"}, "description": "Paper objects with {arxiv_id, title, abstract}"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="batch_ingest_repos",
            description="Batch ingest GitHub repositories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repos": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["repos"]
            }
        ),
        
        Tool(
            name="get_learning_schedule",
            description="Get continuous learning schedule status.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="run_learning_task",
            description="Run a learning task: arxiv_latest, github_trending, news_feed, knowledge_consolidation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_name": {"type": "string", "description": "REQUIRED: arxiv_latest|github_trending|news_feed|knowledge_consolidation"},
                    "task": {"type": "string", "description": "Alias for task_name"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="cross_domain_synthesis",
            description="Discover meta-patterns across all learned knowledge.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="get_universal_principles",
            description="Get universal principles appearing across 3+ domains.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="get_transfer_suggestions",
            description="Get knowledge transfer suggestions between domains.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_domain": {"type": "string"},
                    "to_domain": {"type": "string"}
                },
                "required": ["from_domain", "to_domain"]
            }
        ),
        
        # === AUTONOMOUS LEARNING ENGINE ===
        Tool(
            name="auto_learn_schedules",
            description="Get all autonomous learning schedules with status (overdue tasks, run counts).",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="auto_learn_run_due",
            description="Run all overdue learning tasks (arXiv, GitHub, news, consolidation, Φ measurement).",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="auto_learn_run_task",
            description="Run a specific learning task immediately.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Task name from schedules (e.g. arxiv_consciousness, github_trending, consolidation, phi_measurement)"}
                },
                "required": ["task"]
            }
        ),
        Tool(
            name="auto_learn_digest",
            description="Get learning digest: what was learned since last session, Φ trajectory, graph growth.",
            inputSchema={
                "type": "object",
                "properties": {
                    "since_hours": {"type": "number", "description": "Hours to look back (default 24)"}
                },
                "required": []
            }
        ),
        Tool(
            name="auto_learn_phi_history",
            description="Get Φ measurement history and trend (growing/stable/declining).",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max snapshots (default 20)"}
                },
                "required": []
            }
        ),
        Tool(
            name="auto_learn_update_schedule",
            description="Update a learning schedule (interval, enabled/disabled).",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "interval_hours": {"type": "number"},
                    "enabled": {"type": "boolean"}
                },
                "required": ["task"]
            }
        ),
        
        # === ACTIVE LEARNING ENGINE ===
        Tool(
            name="search_and_learn",
            description="Full pipeline: search→extract→verify→store. Searches arXiv/web/GitHub, extracts facts, verifies against existing knowledge, stores new facts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "source": {"type": "string", "enum": ["arxiv", "web", "github"], "description": "Where to search (default: arxiv)"},
                    "max_results": {"type": "integer", "description": "Max results to process (default: 5)"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="propose_hypothesis",
            description="Propose a hypothesis to track. Confidence updates via Bayesian evidence accumulation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "description": "The hypothesis statement"},
                    "domains": {"type": "array", "items": {"type": "string"}, "description": "Relevant domains"},
                    "prior_confidence": {"type": "number", "description": "Initial belief (0-1, default 0.5)"},
                    "verification_strategy": {"type": "string", "description": "How to test this hypothesis"}
                },
                "required": ["statement"]
            }
        ),
        Tool(
            name="submit_evidence",
            description="Submit evidence for/against a hypothesis. Pass 'direction' as 'for' or 'against'. Auto-resolves at extreme confidence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "evidence": {"type": "string", "description": "Description of the evidence"},
                    "direction": {"type": "string", "enum": ["for", "against"], "description": "REQUIRED: 'for' or 'against'"},
                    "strength": {"type": "number", "description": "Evidence strength (0-1, default 0.5)"},
                    "source_url": {"type": "string"}
                },
                "required": ["hypothesis_id", "evidence"]
            }
        ),
        Tool(
            name="get_hypotheses",
            description="Get active hypotheses being tracked (proposed/testing).",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max to return (default 10)"}
                },
                "required": []
            }
        ),
        
        # === HIERARCHICAL MEMORY + PREFETCH + CONTRADICTION RESOLVER ===
        Tool(
            name="memory_tree",
            description="Get the hierarchical memory tree: domain→subtopic→count. Shows how memories are organized.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="memory_rebuild_index",
            description="Rebuild the hierarchical memory index from all atoms. Run after bulk imports.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="memory_smart_retrieve",
            description="Smart retrieval: classifies query→narrows to domain→returns relevant atoms. O(log n) instead of O(n).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "top_k": {"type": "integer", "description": "Max results (default 20)"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="memory_prefetch",
            description="Predictive prefetch: predict next-needed domains from trajectory and preload into cache.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="memory_prefetch_stats",
            description="Get prefetch cache stats: hit rate, cached domains, recent trajectory.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="auto_resolve_contradictions",
            description="Scan for contradictions and auto-resolve using confidence+recency. Surfaces only ambiguous conflicts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max conflicts to scan (default 200)"}
                },
                "required": []
            }
        ),
        Tool(
            name="get_unresolved_contradictions",
            description="Get contradictions that need manual review (couldn't be auto-resolved).",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max to return (default 20)"}
                },
                "required": []
            }
        ),
        
        # === EMERGENCE DETECTOR (MetaCriticalityObserver) ===
        Tool(
            name="emergence_observe",
            description="Take a full observation of system criticality: measures r, Φ, dH/dt, bridges → computes emergence probability. Call periodically or after major knowledge changes.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="emergence_record_outcome",
            description="After predicting emergence, record whether it actually happened. Triggers Bayesian weight update. Pass did_emerge=true/false.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pred_id": {"type": "string", "description": "Prediction ID from emergence_observe"},
                    "prediction_id": {"type": "string", "description": "Alias for pred_id"},
                    "did_emerge": {"type": "boolean", "description": "REQUIRED: true if emergence occurred, false if not"},
                    "emerged": {"type": "boolean", "description": "Alias for did_emerge"},
                    "outcome": {"type": "string", "description": "Text description of what happened"},
                    "details": {"type": "string", "description": "What emerged (or why not)"}
                },
                "required": []
            }
        ),
        Tool(
            name="emergence_history",
            description="Get recent emergence observations with trend analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max observations (default 20)"}
                },
                "required": []
            }
        ),
        Tool(
            name="emergence_predictions",
            description="Get emergence predictions with outcomes and accuracy stats.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_pending": {"type": "boolean", "description": "Include pending predictions (default true)"}
                },
                "required": []
            }
        ),
        Tool(
            name="emergence_dashboard",
            description="Full emergence dashboard: current state + history + predictions + learned signatures. One-call summary.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        
        Tool(
            name="learn_from_conversation",
            description="Extract valuable information from conversation messages.",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {"type": "array", "items": {"type": "object"}, "description": "Array of {role, content} messages"},
                    "topic": {"type": "string"},
                    "user_id": {"type": "string"}
                },
                "required": ["messages", "topic", "user_id"]
            }
        ),
        
        # PLTM 2.0 - Universal Optimization Principles
        Tool(
            name="quantum_add_state",
            description="Add memory state to superposition.",
            inputSchema={
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "value": {"type": "string"},
                    "confidence": {"type": "number"},
                    "source": {"type": "string"}
                },
                "required": ["subject", "predicate", "value", "confidence", "source"]
            }
        ),
        
        Tool(
            name="quantum_query",
            description="Query superposition with context-dependent collapse.",
            inputSchema={
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "context": {"type": "string"}
                },
                "required": ["subject", "predicate"]
            }
        ),
        
        Tool(
            name="quantum_peek",
            description="Peek at superposition without collapsing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"}
                },
                "required": ["subject", "predicate"]
            }
        ),
        
        Tool(
            name="attention_retrieve",
            description="Retrieve memories weighted by attention to query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                    "domain": {"type": "string"}
                },
                "required": ["user_id", "query"]
            }
        ),
        
        Tool(
            name="attention_multihead",
            description="Multi-head attention retrieval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "query": {"type": "string"},
                    "num_heads": {"type": "integer", "default": 4}
                },
                "required": ["user_id", "query"]
            }
        ),
        
        Tool(
            name="knowledge_add_concept",
            description="Add concept to knowledge graph with connections.",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "domain": {"type": "string"},
                    "related_concepts": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["concept", "domain"]
            }
        ),
        
        Tool(
            name="knowledge_find_path",
            description="Find path between concepts in knowledge graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_concept": {"type": "string"},
                    "to_concept": {"type": "string"}
                },
                "required": ["from_concept", "to_concept"]
            }
        ),
        
        Tool(
            name="knowledge_bridges",
            description="Find bridge concepts connecting domains.",
            inputSchema={
                "type": "object",
                "properties": {
                    "top_k": {"type": "integer", "default": 10}
                },
                "required": []
            }
        ),
        
        Tool(
            name="knowledge_stats",
            description="Get knowledge graph statistics.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="self_improve_cycle",
            description="Run one recursive self-improvement cycle.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="self_improve_meta_learn",
            description="Meta-learn from improvement history.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="self_improve_history",
            description="Get self-improvement history.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="quantum_cleanup",
            description="Garbage collect old quantum states.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="quantum_stats",
            description="Get quantum memory statistics.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="attention_clear_cache",
            description="Clear attention retrieval cache.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="criticality_state",
            description="Get criticality state: entropy, integration, zone.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="criticality_recommend",
            description="Get recommendation: explore or consolidate.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="criticality_adjust",
            description="Auto-adjust toward critical point.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="criticality_history",
            description="Get criticality state history.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="add_provenance",
            description="Add citation/provenance for a claim.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "source_type": {"type": "string", "enum": ["arxiv", "github", "wikipedia", "doi", "url", "book", "internal"]},
                    "source_url": {"type": "string"},
                    "source_title": {"type": "string"},
                    "quoted_span": {"type": "string"},
                    "page_or_section": {"type": "string", "description": "Location in source"},
                    "confidence": {"type": "number"},
                    "arxiv_id": {"type": "string"},
                    "authors": {"type": "string"}
                },
                "required": ["claim_id", "source_type", "source_url", "quoted_span", "confidence"]
            }
        ),
        
        Tool(
            name="get_provenance",
            description="Get citations for a claim.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"}
                },
                "required": ["claim_id"]
            }
        ),
        
        Tool(
            name="provenance_stats",
            description="Get provenance statistics: verified vs unverified.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="unverified_claims",
            description="Get claims lacking proper citations.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="mmr_retrieve",
            description="MMR retrieval for diverse context selection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "description": "default 5"},
                    "lambda_param": {"type": "number", "description": "0=diverse, 1=relevant, default 0.6"},
                    "min_dissim": {"type": "number", "description": "default 0.25"}
                },
                "required": ["user_id", "query"]
            }
        ),
        
        # === TRUE ACTION ACCOUNTING (Georgiev AAE) ===
        Tool(
            name="record_action",
            description="Record action for AAE (Average Action Efficiency) tracking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "description": "Operation type"},
                    "tokens_used": {"type": "integer"},
                    "latency_ms": {"type": "number"},
                    "success": {"type": "boolean"},
                    "context": {"type": "string"}
                },
                "required": ["operation", "tokens_used", "latency_ms", "success"]
            }
        ),
        
        Tool(
            name="get_aae",
            description="Get current AAE metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="aae_trend",
            description="Get AAE trend over recent windows.",
            inputSchema={
                "type": "object",
                "properties": {
                    "window_size": {"type": "integer", "description": "default 10"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="start_action_cycle",
            description="Start action measurement cycle.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cycle_id": {"type": "string"}
                },
                "required": ["cycle_id"]
            }
        ),
        
        Tool(
            name="end_action_cycle",
            description="End action cycle and get metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cycle_id": {"type": "string"}
                },
                "required": []
            }
        ),
        
        # === ENTROPY INJECTION ===
        Tool(
            name="inject_entropy_random",
            description="Inject entropy from random/least-accessed domains.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "n_domains": {"type": "integer", "description": "default 3"},
                    "memories_per_domain": {"type": "integer", "description": "default 2"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="inject_entropy_antipodal",
            description="Find memories maximally distant from current context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "current_context": {"type": "string"},
                    "n_memories": {"type": "integer", "description": "default 5"}
                },
                "required": ["user_id", "current_context"]
            }
        ),
        
        Tool(
            name="inject_entropy_temporal",
            description="Mix old and recent memories to prevent recency bias.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "n_old": {"type": "integer", "description": "default 3"},
                    "n_recent": {"type": "integer", "description": "default 2"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="entropy_stats",
            description="Get entropy statistics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        ),
        
        # === GROQ TRIPLE INGESTION (DataIngestion) ===
        Tool(
            name="ingest_text",
            description="Extract semantic triples from raw text via Groq and store as atoms. Use for pasted chat logs, notes, transcripts. Requires GROQ_API_KEY (falls back to Ollama).",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Raw text to extract triples from (min 20 chars)"},
                    "domain": {"type": "string", "description": "Knowledge domain tag (default: general)"},
                    "source": {"type": "string", "description": "Source label (default: user_input)"},
                    "max_triples": {"type": "integer", "description": "default 15"},
                },
                "required": ["text"]
            }
        ),

        Tool(
            name="ingest_url",
            description="Fetch a URL, extract semantic triples via Groq, store as atoms. Requires GROQ_API_KEY (falls back to Ollama).",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "domain": {"type": "string", "description": "Knowledge domain tag (default: general)"},
                    "max_triples": {"type": "integer", "description": "default 15"},
                },
                "required": ["url"]
            }
        ),

        Tool(
            name="ingest_file",
            description="Read a local file (.txt/.md/.csv/.json/.rst/.log), extract semantic triples via Groq, store as atoms. Requires GROQ_API_KEY (falls back to Ollama).",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute or relative path to the file"},
                    "domain": {"type": "string", "description": "Knowledge domain tag (default: general)"},
                    "max_triples": {"type": "integer", "description": "default 20"},
                },
                "required": ["file_path"]
            }
        ),

        # === ARXIV INGESTION (Real Provenance) ===
        Tool(
            name="ingest_arxiv",
            description="Ingest arXiv paper with real provenance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "ArXiv paper ID"},
                    "user_id": {"type": "string", "description": "User/subject to store claims under (default: pltm_knowledge)"}
                },
                "required": ["arxiv_id"]
            }
        ),
        
        Tool(
            name="search_arxiv",
            description="Search arXiv for papers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "default 5"}
                },
                "required": ["query"]
            }
        ),
        
        Tool(
            name="arxiv_history",
            description="Get ingested arXiv paper history.",
            inputSchema={
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "default 10"}
                },
                "required": []
            }
        ),
        
        # === EPISTEMIC HYGIENE TOOLS ===
        Tool(
            name="check_before_claiming",
            description="Pre-response confidence check before factual claims.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "felt_confidence": {"type": "number"},
                    "domain": {"type": "string", "description": "default: general"},
                    "has_verified": {"type": "boolean"},
                    "epistemic_status": {"type": "string", "description": "TRAINING_DATA|VERIFIED|REASONING|USER_STATED"}
                },
                "required": ["claim", "felt_confidence"]
            }
        ),
        
        Tool(
            name="log_claim",
            description="Log a factual claim for calibration tracking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "felt_confidence": {"type": "number"},
                    "domain": {"type": "string", "description": "default: general"},
                    "epistemic_status": {"type": "string", "description": "TRAINING_DATA|VERIFIED|REASONING|USER_STATED"},
                    "has_verified": {"type": "boolean"}
                },
                "required": ["claim", "felt_confidence"]
            }
        ),
        
        Tool(
            name="resolve_claim",
            description="Resolve a logged claim as correct/incorrect. Pass was_correct=true or was_correct=false.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "claim_text": {"type": "string"},
                    "was_correct": {"type": "boolean", "description": "REQUIRED: true if claim was correct, false if not"},
                    "correct": {"type": "boolean", "description": "Alias for was_correct"},
                    "correction_source": {"type": "string"},
                    "correction_detail": {"type": "string"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="get_calibration",
            description="Get calibration curves and accuracy stats.",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "empty for all"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="calibrate_confidence_live",
            description="Real-time confidence calibration with hedged phrasing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "felt_confidence": {"type": "number"},
                    "domain": {"type": "string", "description": "default: general"}
                },
                "required": ["claim", "felt_confidence"]
            }
        ),
        
        # === PERSONALITY / SELF-MODELING TOOLS ===
        Tool(
            name="self_profile",
            description="Get self-profile: style, curiosity, values, reasoning.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "default: claude"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="get_longitudinal_stats",
            description="Cross-conversation analytics with personality evolution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "default: claude"},
                    "days": {"type": "integer", "description": "default: 30"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="bootstrap_self_model",
            description="Bootstrap self-model by scanning existing DB data (atoms, communication, curiosity, reasoning, predictions, values). Returns summary statistics. No arguments required.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "default: claude"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="track_curiosity_spike",
            description="Track genuine interest vs performative engagement.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "indicators": {"type": "array", "items": {"type": "string"}, "description": "Curiosity indicators"},
                    "engagement_score": {"type": "number"},
                    "context": {"type": "string"}
                },
                "required": ["topic", "indicators"]
            }
        ),
        
        Tool(
            name="learn_communication_style",
            description="Track communication style in context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context": {"type": "string"},
                    "response_text": {"type": "string"},
                    "markers": {"type": "object"}
                },
                "required": ["context", "response_text"]
            }
        ),
        
        # === CROSS-MODEL TOOLS ===
        Tool(
            name="route_llm_task",
            description="Route LLM task to best provider.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_type": {"type": "string", "description": "analysis|coding|creative|factual|summarization|translation"},
                    "prefer_provider": {"type": "string", "description": "ollama|groq|deepseek|openai|anthropic"},
                    "require_privacy": {"type": "boolean"}
                },
                "required": []
            }
        ),
        
        # === SESSION TOOLS ===
        Tool(
            name="auto_init_session",
            description="PERSISTENT IDENTITY LOADER. Call at START of conversation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "default: claude"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="end_session",
            description="End session: snapshot personality, extract learnings into typed memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "user_id": {"type": "string", "description": "default: claude"},
                    "learnings": {
                        "type": "array",
                        "description": "Learnings to extract",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["episodic", "semantic", "belief", "procedural"], "description": "Memory type"},
                                "content": {"type": "string", "description": "What was learned"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "emotional_valence": {"type": "number", "minimum": -1, "maximum": 1},
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "trigger": {"type": "string", "description": "For procedural: what activates this"},
                                "action": {"type": "string", "description": "For procedural: what to do"},
                            },
                            "required": ["type", "content"]
                        }
                    }
                },
                "required": []
            }
        ),
        
        Tool(
            name="generate_memory_prompt",
            description="Generate context block from relevant memories for conversation start.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "conversation_topic": {"type": "string", "description": "Topic hint"},
                    "max_tokens": {"type": "integer", "description": "Token budget, default 500"},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="belief_auto_check",
            description="Re-evaluate beliefs against current semantic evidence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="jury_stats",
            description="Get jury system stats: per-judge accuracy, MetaJudge performance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_history": {"type": "boolean"},
                    "history_limit": {"type": "integer"},
                },
                "required": []
            }
        ),
        
        Tool(
            name="jury_feedback",
            description=("Record ground truth feedback for MetaJudge training. "
                         "feedback_type semantics: 'confirmed' = approved memory was right "
                         "(approvers correct); 'false_positive' = approved memory was wrong "
                         "(approvers wrong); 'false_negative' = rejected memory should have "
                         "been stored (rejecters wrong); 'rejection_confirmed' = rejected/"
                         "quarantined memory was correctly blocked (rejecters correct, "
                         "approvers wrong). Match by memory_id, or content_hash for "
                         "memories that were never stored."),
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory ID to give feedback on"},
                    "atom_id": {"type": "string", "description": "Alias for memory_id"},
                    "content_hash": {"type": "string", "description": "sha1 of the memory's content — matches REJECTed memories that were never stored"},
                    "feedback_type": {"type": "string", "enum": ["false_positive", "false_negative", "confirmed", "rejection_confirmed"], "description": "REQUIRED: one of the four feedback types"},
                    "correct": {"type": "boolean", "description": "Alternative: true→confirmed, false→false_positive"},
                    "details": {"type": "string"},
                },
                "required": []
            }
        ),
        
        Tool(
            name="process_message",
            description="Run message through 3-lane pipeline: extract, validate, store.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "user_id": {"type": "string"},
                    "context": {"type": "string"},
                    "auto_tag": {"type": "boolean"},
                },
                "required": ["message", "user_id"]
            }
        ),
        
        Tool(
            name="process_message_batch",
            description="Run multiple messages through 3-lane pipeline.",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {"type": "array", "items": {"type": "string"}},
                    "user_id": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["messages", "user_id"]
            }
        ),
        
        Tool(
            name="pipeline_stats",
            description="Get 3-lane pipeline statistics.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        # === MEMORY INTELLIGENCE TOOLS ===
        Tool(
            name="apply_memory_decay",
            description="Apply Ebbinghaus forgetting curve to all memories.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}
        ),
        Tool(
            name="decay_forecast",
            description="Forecast which memories will decay below threshold.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}, "hours_ahead": {"type": "integer", "description": "default 168 (1 week)"}}, "required": ["user_id"]}
        ),
        Tool(
            name="consolidate_memories",
            description="Promote repeated episodic patterns into semantic memories.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}, "min_cluster_size": {"type": "integer", "description": "default 3"}, "similarity_threshold": {"type": "number", "description": "default 0.55"}}, "required": ["user_id"]}
        ),
        Tool(
            name="contextual_retrieve",
            description="RAG-style retrieval: build prompt block from relevant memories.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}, "messages": {"type": "array", "items": {"type": "string"}}, "max_memories": {"type": "integer", "description": "default 12"}}, "required": ["user_id", "messages"]}
        ),
        Tool(
            name="rank_by_importance",
            description="Rank memories by importance score.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}, "limit": {"type": "integer", "description": "default 50"}}, "required": ["user_id"]}
        ),
        Tool(
            name="surface_conflicts",
            description="Detect contradicting memories with resolution options.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}
        ),
        Tool(
            name="resolve_memory_conflict",
            description="Resolve a surfaced memory conflict by choosing an action (keep_a, keep_b, keep_both, delete_both).",
            inputSchema={"type": "object", "properties": {"conflict_id": {"type": "string"}, "action": {"type": "string", "enum": ["keep_a", "keep_b", "keep_both", "delete_both"]}, "user_id": {"type": "string"}}, "required": ["conflict_id", "action", "user_id"]}
        ),
        Tool(
            name="memory_clusters",
            description="Build topic clusters from memories using embedding similarity.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}, "similarity_threshold": {"type": "number", "description": "default 0.5"}, "min_cluster_size": {"type": "integer", "description": "default 2"}}, "required": ["user_id"]}
        ),
        Tool(
            name="share_memory",
            description="Share a memory with another user.",
            inputSchema={"type": "object", "properties": {"memory_id": {"type": "string"}, "owner_id": {"type": "string"}, "target_user_id": {"type": "string"}, "permission": {"type": "string", "enum": ["read", "write"]}}, "required": ["memory_id", "owner_id", "target_user_id"]}
        ),
        Tool(
            name="shared_with_me",
            description="Get memories shared with this user.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}
        ),
        Tool(
            name="export_memory_profile",
            description="Export all memories as portable JSON.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}
        ),
        Tool(
            name="import_memory_profile",
            description="Import memory profile from JSON.",
            inputSchema={"type": "object", "properties": {"profile_json": {"type": "string"}, "target_user_id": {"type": "string"}, "merge": {"type": "boolean"}}, "required": ["profile_json"]}
        ),
        Tool(
            name="memory_provenance",
            description="Get provenance chain for a memory.",
            inputSchema={"type": "object", "properties": {"memory_id": {"type": "string"}}, "required": ["memory_id"]}
        ),
        Tool(
            name="apply_confidence_decay",
            description="Apply gradual confidence reduction to contradicted beliefs.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}
        ),
        Tool(
            name="memory_audit",
            description="Full memory health audit. Returns health score 0-100.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}
        ),

        # === ΦRMS (Phi Resource Management) TOOLS ===
        Tool(
            name="phi_score_memories",
            description="Score all memories by Φ-density (integrated information value per token). Returns distribution stats.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}
        ),
        Tool(
            name="phi_prune",
            description="Criticality-aware pruning: remove lowest-Φ memories while preserving self-organized criticality.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}, "target_token_savings": {"type": "integer", "description": "Target tokens to free (default 5000)"}, "max_removals": {"type": "integer", "description": "Max memories to remove (default 20)"}}, "required": ["user_id"]}
        ),
        Tool(
            name="phi_consolidate",
            description="Φ-preserving episodic→semantic consolidation. Only promotes if Φ is maintained.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}, "min_cluster_size": {"type": "integer", "description": "default 3"}, "similarity_threshold": {"type": "number", "description": "default 0.55"}}, "required": ["user_id"]}
        ),
        Tool(
            name="phi_build_context",
            description="Knapsack context builder: select optimal memories for conversation by Φ-density × relevance within token budget.",
            inputSchema={"type": "object", "properties": {"user_id": {"type": "string"}, "messages": {"type": "array", "items": {"type": "string"}}, "token_budget": {"type": "integer", "description": "default 2000"}}, "required": ["user_id", "messages"]}
        ),

        # === IMPROVEMENT LOOP TOOLS ===
        Tool(
            name="tool_usage_stats",
            description="Get tool usage statistics: call counts, durations, success rates, and frequency distribution over last N days.",
            inputSchema={"type": "object", "properties": {"days": {"type": "integer", "description": "Lookback period in days (default 30)"}}, "required": []}
        ),
        Tool(
            name="tool_redundancy_report",
            description="Identify unused, error-prone, and co-occurring tools. Helps evolve the tool inventory.",
            inputSchema={"type": "object", "properties": {"days": {"type": "integer", "description": "Lookback period in days (default 30)"}}, "required": []}
        ),
        Tool(
            name="tool_consolidation_proposals",
            description="AI-actionable proposals for tool inventory evolution: deprecate unused, combine sequential, fix error-prone, optimize slow.",
            inputSchema={"type": "object", "properties": {"days": {"type": "integer", "description": "Lookback period in days (default 30)"}}, "required": []}
        ),
        Tool(
            name="snapshot_architecture",
            description="Take a versioned snapshot of current system state (tools, counts, metrics) for A/B comparison.",
            inputSchema={"type": "object", "properties": {"label": {"type": "string", "description": "Human-readable label e.g. 'pre-ΦRMS'"}, "notes": {"type": "string", "description": "What changed or why snapshot was taken"}}, "required": ["label"]}
        ),
        Tool(
            name="list_architecture_snapshots",
            description="List previous architecture snapshots for comparison.",
            inputSchema={"type": "object", "properties": {"limit": {"type": "integer", "description": "default 20"}}, "required": []}
        ),
        Tool(
            name="compare_architectures",
            description="Compare two architecture snapshots: tool diff, metric deltas, count changes.",
            inputSchema={"type": "object", "properties": {"snapshot_a": {"type": "string", "description": "ID of first snapshot"}, "snapshot_b": {"type": "string", "description": "ID of second snapshot"}}, "required": ["snapshot_a", "snapshot_b"]}
        ),

        # === EXPERIMENT / RESEARCH TOOLS ===
        Tool(
            name="trace_claim_reasoning",
            description="Audit trail for epistemic decisions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "claim_id": {"type": "string"},
                    "domain": {"type": "string"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="constraint_sensitivity_test",
            description="Simulate calibration changes (read-only).",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "simulated_accuracy": {"type": "number", "description": "Simulated accuracy 0-1"},
                    "test_claims": {"type": "array", "items": {"type": "object", "properties": {"claim": {"type": "string"}, "felt_confidence": {"type": "number"}}}}
                },
                "required": ["domain", "simulated_accuracy"]
            }
        ),
        
        Tool(
            name="domain_cognitive_map",
            description="Map cognitive topology of a domain.",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "empty for all"},
                    "include_claims": {"type": "boolean"},
                    "include_interventions": {"type": "boolean"}
                },
                "required": []
            }
        ),
        
        # === DATA ACCESS ===
        Tool(
            name="query_pltm_sql",
            description="Direct read-only SQL query against PLTM database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "params": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["sql"]
            }
        ),
        
        # === TYPED MEMORY SYSTEM ===
        Tool(
            name="store_episodic",
            description="Store episodic memory (event/interaction). Fast decay ~2 days.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "content": {"type": "string", "description": "What happened"},
                    "context": {"type": "string", "description": "When/where"},
                    "emotional_valence": {"type": "number", "minimum": -1, "maximum": 1},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["user_id", "content"]
            }
        ),
        
        Tool(
            name="store_semantic",
            description="Store semantic memory (stable fact). Slow decay ~30 days.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "content": {"type": "string", "description": "The fact"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "source": {"type": "string", "description": "user_stated|observed|inferred|consolidated"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["user_id", "content"]
            }
        ),
        
        Tool(
            name="store_belief",
            description="Store belief (revisable inference). Confidence-tracked.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "content": {"type": "string", "description": "The belief"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["user_id", "content", "confidence"]
            }
        ),
        
        Tool(
            name="store_procedural",
            description="Store procedural memory (trigger→action pattern). Very stable ~90 days.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "trigger": {"type": "string", "description": "What activates this"},
                    "action": {"type": "string", "description": "What to do"},
                    "content": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["user_id", "trigger", "action"]
            }
        ),
        
        Tool(
            name="recall_memories",
            description="Retrieve typed memories. Recalling strengthens via rehearsal.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "memory_type": {"type": "string", "enum": ["episodic", "semantic", "belief", "procedural"]},
                    "min_strength": {"type": "number", "description": "default 0.1", "minimum": 0, "maximum": 1},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "description": "default 20"},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="search_memories",
            description="Full-text search across typed memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "description": "default 20"},
                },
                "required": ["user_id", "query"]
            }
        ),
        
        Tool(
            name="update_belief",
            description="Update belief confidence with new evidence. Pass confidence_delta (positive number) and evidence_type ('for' or 'against').",
            inputSchema={
                "type": "object",
                "properties": {
                    "belief_id": {"type": "string"},
                    "evidence_type": {"type": "string", "enum": ["for", "against"], "description": "'for' or 'against'"},
                    "evidence_id": {"type": "string"},
                    "confidence_delta": {"type": "number", "description": "Amount to adjust confidence"},
                    "delta": {"type": "number", "description": "Alias for confidence_delta"},
                },
                "required": ["belief_id"]
            }
        ),
        
        Tool(
            name="record_procedure_outcome",
            description="Record if procedural memory worked. Pass success=true or success=false. Success strengthens the memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "procedure_id": {"type": "string"},
                    "success": {"type": "boolean", "description": "REQUIRED: true if procedure worked, false if not"},
                    "worked": {"type": "boolean", "description": "Alias for success"},
                },
                "required": ["procedure_id"]
            }
        ),
        
        Tool(
            name="consolidate_typed_memories",
            description="Promote repeated episodic patterns into semantic memories (typed memory system).",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "min_episodes": {"type": "integer", "description": "default 3"},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="memory_stats",
            description="Get typed memory counts, strength, confidence by type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"]
            }
        ),
        
        # === MEMORY INTELLIGENCE ===
        Tool(
            name="detect_contradictions",
            description="Find contradicting memories for resolution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="what_do_i_know_about",
            description="Synthesized retrieval across ALL memory types for a topic.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "topic": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["user_id", "topic"]
            }
        ),
        
        Tool(
            name="auto_tag_memories",
            description="Auto-classify memories into taxonomy domains.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="correct_memory",
            description="Correct a memory's content. Pass the corrected text in 'new_content'. Old content is preserved in correction history.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "ID of the memory to correct"},
                    "new_content": {"type": "string", "description": "The corrected text to replace the old content"},
                    "reason": {"type": "string", "description": "Why the correction is needed"},
                    "new_confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Updated confidence (0-1)"},
                },
                "required": ["memory_id", "new_content"]
            }
        ),
        
        Tool(
            name="forget_memory",
            description="Delete a memory (logged for audit).",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["memory_id"]
            }
        ),
        
        Tool(
            name="auto_prune_memories",
            description="Prune memories below strength threshold.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "strength_threshold": {"type": "number", "description": "default 0.05", "minimum": 0, "maximum": 0.5},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="get_relevant_context",
            description="Pre-fetch memories relevant to conversation topic.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "conversation_topic": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["user_id", "conversation_topic"]
            }
        ),
        
        Tool(
            name="user_timeline",
            description="Chronological view of all memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                },
                "required": ["user_id"]
            }
        ),
        
        # === EMBEDDING SEARCH ===
        Tool(
            name="semantic_search",
            description="Embedding similarity search across typed memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "min_similarity": {"type": "number", "description": "default 0.3", "minimum": 0, "maximum": 1},
                },
                "required": ["query"]
            }
        ),
        
        Tool(
            name="index_embeddings",
            description="Batch-index memories missing embeddings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="find_similar_memories",
            description="Find semantically similar memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "limit": {"type": "integer"},
                    "min_similarity": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["memory_id"]
            }
        ),
        # Session Continuity
        Tool(
            name="session_handoff",
            description="Generate a complete, token-budgeted handoff block for session resumption. Replaces calling auto_init_session + get_session_bridge + phi_build_context separately. Returns identity, working memory, trajectory, goals, warnings, and Φ-context in one compact block.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "default": "claude"},
                    "token_budget": {"type": "integer", "default": 1500, "description": "Max tokens for the handoff block (500-3000)"},
                    "include_sections": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["identity", "working_memory", "trajectory", "goals", "warnings", "phi_context"]},
                        "description": "Sections to include (default: all)"
                    },
                },
            }
        ),
        Tool(
            name="capture_session_state",
            description="End-of-session capture: snapshot working memory (active memories, tools, goals, state) and encode conversation trajectory (topic flow, decisions, momentum). Call before or alongside end_session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Current session ID (from auto_init_session)"},
                    "user_id": {"type": "string", "default": "claude"},
                    "summary": {"type": "string", "default": "", "description": "Optional session summary"},
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="get_working_memory",
            description="Get the most recent working memory snapshot — what was actively loaded in the last session (memories accessed, tools used, open questions, state).",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "default": "claude"},
                },
            }
        ),
    ]


def _build_dispatch_table():
    """Build tool name → handler function mapping. Called once at import time.
    
    Uses extracted handler files where available, inline handlers for the rest.
    Dict keys are unique by definition — no more duplicate tool name bugs.
    """
    from functools import partial
    from mcp_server.handlers.memory_handlers import (
        handle_store_typed, handle_recall_memories, handle_search_memories,
        handle_update_belief_mem, handle_record_procedure, handle_consolidate,
        handle_memory_stats, handle_detect_contradictions, handle_what_do_i_know,
        handle_auto_tag, handle_correct_memory, handle_forget_memory,
        handle_auto_prune, handle_get_relevant_context, handle_user_timeline,
        handle_semantic_search, handle_index_embeddings, handle_find_similar,
    )
    from mcp_server.handlers.intelligence_handlers import (
        handle_process_conversation as _h_process_conv,
        handle_pipeline_stats as _h_pipeline_stats,
        handle_apply_memory_decay, handle_decay_forecast,
        handle_consolidate_memories, handle_contextual_retrieve,
        handle_rank_by_importance, handle_surface_conflicts,
        handle_resolve_memory_conflict, handle_memory_clusters,
        handle_share_memory, handle_shared_with_me,
        handle_export_memory_profile, handle_import_memory_profile,
        handle_memory_provenance, handle_apply_confidence_decay,
        handle_memory_audit,
    )
    from mcp_server.handlers.phi_rms_handlers import (
        handle_phi_score_memories, handle_phi_prune,
        handle_phi_consolidate, handle_phi_build_context,
    )

    # Wrapper for store_typed variants (need extra mem_type arg)
    async def _store_episodic(args):
        return await handle_store_typed(args, "episodic")
    async def _store_semantic(args):
        return await handle_store_typed(args, "semantic")
    async def _store_belief(args):
        return await handle_store_typed(args, "belief")
    async def _store_procedural(args):
        return await handle_store_typed(args, "procedural")

    return {
        # --- Personality & Mood (inline) ---
        "store_memory_atom": handle_store_atom,
        "query_personality": handle_query_personality,
        "detect_mood": handle_detect_mood,
        "get_mood_patterns": handle_mood_patterns,
        "resolve_conflict": handle_resolve_conflict,
        "extract_personality_traits": handle_extract_traits,
        "get_adaptive_prompt": handle_adaptive_prompt,
        "get_personality_summary": handle_personality_summary,
        "bootstrap_from_sample": handle_bootstrap_sample,
        "bootstrap_from_messages": handle_bootstrap_messages,
        "track_trait_evolution": handle_track_evolution,
        "predict_reaction": handle_predict_reaction,
        "get_meta_patterns": handle_meta_patterns,
        "learn_from_interaction": handle_learn_interaction,
        "predict_session": handle_predict_session,
        "get_self_model": handle_self_model,
        "init_claude_session": handle_init_claude_session,
        "update_claude_style": handle_update_claude_style,
        "learn_interaction_dynamic": handle_learn_dynamic,
        "record_milestone": handle_record_milestone,
        "add_shared_vocabulary": handle_add_vocabulary,
        "get_claude_personality": handle_get_claude_personality,
        "evolve_claude_personality": handle_evolve_claude,
        "check_pltm_available": handle_check_pltm,
        "pltm_mode": handle_pltm_mode,
        "deep_personality_analysis": handle_deep_analysis,
        "enrich_claude_personality": handle_enrich_personality,
        # --- Sensory Input (inline) ---
        "sensory_start": handle_sensory_start,
        "sensory_stop": handle_sensory_stop,
        "sensory_look": handle_sensory_look,
        "sensory_listen": handle_sensory_listen,
        "sensory_state": handle_sensory_state,
        "sensory_history": handle_sensory_history,
        "sensory_diagnostics": handle_sensory_diagnostics,
        # --- Learning (inline) ---
        "learn_from_url": handle_learn_url,
        "learn_from_paper": handle_learn_paper,
        "learn_from_code": handle_learn_code,
        "get_learning_stats": handle_learning_stats,
        "batch_ingest_wikipedia": handle_batch_wikipedia,
        "batch_ingest_papers": handle_batch_papers,
        "batch_ingest_repos": handle_batch_repos,
        "get_learning_schedule": handle_learning_schedule,
        "run_learning_task": handle_run_task,
        "cross_domain_synthesis": handle_synthesis,
        "get_universal_principles": handle_universal_principles,
        "get_transfer_suggestions": handle_transfer_suggestions,
        # --- Autonomous Learning ---
        "auto_learn_schedules": handle_auto_learn_schedules,
        "auto_learn_run_due": handle_auto_learn_run_due,
        "auto_learn_run_task": handle_auto_learn_run_task,
        "auto_learn_digest": handle_auto_learn_digest,
        "auto_learn_phi_history": handle_auto_learn_phi_history,
        "auto_learn_update_schedule": handle_auto_learn_update_schedule,
        # --- Active Learning ---
        "search_and_learn": handle_search_and_learn,
        "propose_hypothesis": handle_propose_hypothesis,
        "submit_evidence": handle_submit_evidence,
        "get_hypotheses": handle_get_hypotheses,
        # --- Hierarchical Memory + Prefetch + Contradictions ---
        "memory_tree": handle_memory_tree,
        "memory_rebuild_index": handle_memory_rebuild_index,
        "memory_smart_retrieve": handle_memory_smart_retrieve,
        "memory_prefetch": handle_memory_prefetch,
        "memory_prefetch_stats": handle_memory_prefetch_stats,
        "auto_resolve_contradictions": handle_auto_resolve_contradictions,
        "get_unresolved_contradictions": handle_get_unresolved_contradictions,
        # --- Emergence Detector ---
        "emergence_observe": handle_emergence_observe,
        "emergence_record_outcome": handle_emergence_record_outcome,
        "emergence_history": handle_emergence_history,
        "emergence_predictions": handle_emergence_predictions,
        "emergence_dashboard": handle_emergence_dashboard,
        "learn_from_conversation": handle_learn_conversation,
        # --- PLTM 2.0 (inline) ---
        "quantum_add_state": handle_quantum_add,
        "quantum_query": handle_quantum_query,
        "quantum_peek": handle_quantum_peek,
        "attention_retrieve": handle_attention_retrieve,
        "attention_multihead": handle_attention_multihead,
        "knowledge_add_concept": handle_knowledge_add,
        "knowledge_find_path": handle_knowledge_path,
        "knowledge_bridges": handle_knowledge_bridges,
        "knowledge_stats": handle_knowledge_stats,
        "self_improve_cycle": handle_improve_cycle,
        "self_improve_meta_learn": handle_meta_learn,
        "self_improve_history": handle_improve_history,
        "quantum_cleanup": handle_quantum_cleanup,
        "quantum_stats": handle_quantum_stats,
        "attention_clear_cache": handle_attention_clear_cache,
        "criticality_state": handle_criticality_state,
        "criticality_recommend": handle_criticality_recommend,
        "criticality_adjust": handle_criticality_adjust,
        "criticality_history": handle_criticality_history,
        "add_provenance": handle_add_provenance,
        "get_provenance": handle_get_provenance,
        "provenance_stats": handle_provenance_stats,
        "unverified_claims": handle_unverified_claims,
        "mmr_retrieve": handle_mmr_retrieve,
        # --- Action Accounting (inline) ---
        "record_action": handle_record_action,
        "get_aae": handle_get_aae,
        "aae_trend": handle_aae_trend,
        "start_action_cycle": handle_start_action_cycle,
        "end_action_cycle": handle_end_action_cycle,
        # --- Entropy Injection (inline) ---
        "inject_entropy_random": handle_inject_entropy_random,
        "inject_entropy_antipodal": handle_inject_entropy_antipodal,
        "inject_entropy_temporal": handle_inject_entropy_temporal,
        "entropy_stats": handle_entropy_stats,
        # --- ArXiv (inline) ---
        "ingest_text": handle_ingest_text,
        "ingest_url": handle_ingest_url,
        "ingest_file": handle_ingest_file,
        "ingest_arxiv": handle_ingest_arxiv,
        "search_arxiv": handle_search_arxiv,
        "arxiv_history": handle_arxiv_history,
        # --- Epistemic Hygiene (inline) ---
        "check_before_claiming": handle_check_before_claiming,
        "log_claim": handle_log_claim,
        "resolve_claim": handle_resolve_claim,
        "get_calibration": handle_get_calibration,
        "calibrate_confidence_live": handle_calibrate_confidence_live,
        # --- Self-Modeling (inline) ---
        "self_profile": handle_self_profile,
        "get_longitudinal_stats": handle_get_longitudinal_stats,
        "bootstrap_self_model": handle_bootstrap_self_model,
        "track_curiosity_spike": handle_track_curiosity_spike,
        "learn_communication_style": handle_learn_communication_style,
        # --- Cross-Model (inline) ---
        "route_llm_task": handle_route_llm_task,
        # --- Session (inline) ---
        "auto_init_session": handle_auto_init_session,
        "end_session": handle_end_session,
        "generate_memory_prompt": handle_generate_memory_prompt,
        "belief_auto_check": handle_belief_auto_check,
        "jury_stats": handle_jury_stats,
        "jury_feedback": handle_jury_feedback,
        "process_message": handle_process_message,
        "process_message_batch": handle_process_message_batch,
        "pipeline_stats": _h_pipeline_stats,
        # --- Memory Intelligence (extracted → intelligence_handlers.py) ---
        "apply_memory_decay": handle_apply_memory_decay,
        "decay_forecast": handle_decay_forecast,
        "consolidate_memories": handle_consolidate_memories,
        "contextual_retrieve": handle_contextual_retrieve,
        "rank_by_importance": handle_rank_by_importance,
        "surface_conflicts": handle_surface_conflicts,
        "resolve_memory_conflict": handle_resolve_memory_conflict,
        "memory_clusters": handle_memory_clusters,
        "share_memory": handle_share_memory,
        "shared_with_me": handle_shared_with_me,
        "export_memory_profile": handle_export_memory_profile,
        "import_memory_profile": handle_import_memory_profile,
        "memory_provenance": handle_memory_provenance,
        "apply_confidence_decay": handle_apply_confidence_decay,
        "memory_audit": handle_memory_audit,
        # --- ΦRMS (extracted → phi_rms_handlers.py) ---
        "phi_score_memories": handle_phi_score_memories,
        "phi_prune": handle_phi_prune,
        "phi_consolidate": handle_phi_consolidate,
        "phi_build_context": handle_phi_build_context,
        # --- Improvement Loops (inline) ---
        "tool_usage_stats": handle_tool_usage_stats,
        "tool_redundancy_report": handle_tool_redundancy_report,
        "tool_consolidation_proposals": handle_tool_consolidation_proposals,
        "snapshot_architecture": handle_snapshot_architecture,
        "list_architecture_snapshots": handle_list_architecture_snapshots,
        "compare_architectures": handle_compare_architectures,
        # --- Experiments (inline) ---
        "trace_claim_reasoning": handle_trace_claim_reasoning,
        "constraint_sensitivity_test": handle_constraint_sensitivity_test,
        "domain_cognitive_map": handle_domain_cognitive_map,
        # --- Data Access (inline) ---
        "query_pltm_sql": handle_query_pltm_sql,
        # --- Typed Memory System (extracted → memory_handlers.py) ---
        "store_episodic": _store_episodic,
        "store_semantic": _store_semantic,
        "store_belief": _store_belief,
        "store_procedural": _store_procedural,
        "recall_memories": handle_recall_memories,
        "search_memories": handle_search_memories,
        "update_belief": handle_update_belief_mem,
        "record_procedure_outcome": handle_record_procedure,
        "consolidate_typed_memories": handle_consolidate,
        "memory_stats": handle_memory_stats,
        "detect_contradictions": handle_detect_contradictions,
        "what_do_i_know_about": handle_what_do_i_know,
        "auto_tag_memories": handle_auto_tag,
        "correct_memory": handle_correct_memory,
        "forget_memory": handle_forget_memory,
        "auto_prune_memories": handle_auto_prune,
        "get_relevant_context": handle_get_relevant_context,
        "user_timeline": handle_user_timeline,
        # --- Embedding Search (extracted → memory_handlers.py) ---
        "semantic_search": handle_semantic_search,
        "index_embeddings": handle_index_embeddings,
        "find_similar_memories": handle_find_similar,
        # --- Session Continuity (inline) ---
        "session_handoff": handle_session_handoff,
        "capture_session_state": handle_capture_session_state,
        "get_working_memory": handle_get_working_memory,
    }


# Dispatch table — built lazily on first call to avoid import-time issues
_TOOL_DISPATCH = None


async def _dispatch_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Dispatch tool calls via dictionary lookup. O(1) instead of O(n) elif chain."""
    global _TOOL_DISPATCH
    if _TOOL_DISPATCH is None:
        _TOOL_DISPATCH = _build_dispatch_table()

    handler = _TOOL_DISPATCH.get(name)
    if handler is None:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    return await handler(arguments)


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls with timeout, store guard, and analytics logging."""
    if store is None:
        return [TextContent(
            type="text",
            text=f"Error: PLTM not initialized. Restart the MCP server."
        )]
    
    import time as _time
    _t0 = _time.time()
    _success = True
    _error_type = None
    result = None
    
    try:
        result = await asyncio.wait_for(_dispatch_tool(name, arguments), timeout=30.0)
        return result
    except asyncio.TimeoutError:
        _success = False
        _error_type = "timeout"
        logger.error(f"Tool {name} timed out after 30s")
        return [TextContent(
            type="text",
            text=f"Error: Tool '{name}' timed out after 30s"
        )]
    except Exception as e:
        _success = False
        _error_type = type(e).__name__
        logger.error(f"Error in tool {name}: {e}")
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]
    finally:
        if tool_analytics:
            _duration = (_time.time() - _t0) * 1000
            _result_size = 0
            if result:
                try:
                    _result_size = sum(len(r.text) for r in result if hasattr(r, 'text'))
                except Exception:
                    pass
            tool_analytics.log_invocation(
                tool_name=name,
                session_id=_current_session_id,
                duration_ms=_duration,
                success=_success,
                error_type=_error_type,
                args_hash=tool_analytics.hash_args(arguments) if arguments else "",
                result_size=_result_size,
            )


async def handle_store_atom(args: Dict[str, Any]) -> List[TextContent]:
    """Store a memory atom"""
    # Map atom type string to enum
    atom_type_map = {
        "fact": AtomType.STATE,
        "personality_trait": AtomType.PERSONALITY_TRAIT,
        "communication_style": AtomType.COMMUNICATION_STYLE,
        "interaction_pattern": AtomType.INTERACTION_PATTERN,
        "mood": AtomType.STATE,
        "preference": AtomType.PREFERENCE,
    }
    
    atom_type = atom_type_map.get(args["atom_type"], AtomType.STATE)
    
    # Create atom
    atom = MemoryAtom(
        atom_type=atom_type,
        subject=args["subject"],
        predicate=args["predicate"],
        object=args["object"],
        confidence=args.get("confidence", 0.7),
        strength=args.get("confidence", 0.7),
        provenance=Provenance.INFERRED,
        source_user=args["user_id"],
        contexts=args.get("context", []),
        graph=GraphType.SUBSTANTIATED
    )
    
    # Store in database
    await store.add_atom(atom)
    
    # Verify write succeeded
    verified = False
    try:
        check = await store.get_atom(atom.id)
        verified = check is not None
    except Exception:
        verified = False
    
    if not verified:
        logger.error(f"WRITE VERIFICATION FAILED for atom {atom.id}")
    
    return [TextContent(
        type="text",
        text=compact_json({
            "status": "stored" if verified else "WRITE_FAILED",
            "atom_id": str(atom.id),
            "verified": verified,
            "atom_type": args["atom_type"],
            "content": f"{args['subject']} {args['predicate']} {args['object']}"
        })
    )]


async def handle_query_personality(args: Dict[str, Any]) -> List[TextContent]:
    """Query personality profile"""
    user_id = args["user_id"]
    context = args.get("context")
    
    if context:
        # Context-specific personality
        personality = await contextual_personality.get_personality_for_context(
            user_id, context
        )
    else:
        # General personality
        personality = await personality_synth.synthesize_personality(user_id)
    
    return [TextContent(
        type="text",
        text=compact_json(personality)
    )]


async def handle_detect_mood(args: Dict[str, Any]) -> List[TextContent]:
    """Detect mood from message"""
    user_id = args["user_id"]
    message = args["message"]
    
    mood_atom = await mood_tracker.detect_mood(user_id, message)
    
    if mood_atom:
        # Store mood
        await store.add_atom(mood_atom)
        
        result = {
            "mood": mood_atom.object,
            "confidence": mood_atom.confidence,
            "detected": True
        }
    else:
        result = {
            "mood": None,
            "confidence": 0.0,
            "detected": False
        }
    
    return [TextContent(
        type="text",
        text=compact_json(result)
    )]


async def handle_mood_patterns(args: Dict[str, Any]) -> List[TextContent]:
    """Get mood patterns"""
    user_id = args["user_id"]
    window_days = int(args.get("window_days", 90))
    
    patterns = await mood_patterns.detect_patterns(user_id, window_days)
    
    # Get insights
    insights = await mood_patterns.get_mood_insights(user_id)
    
    result = {
        "patterns": patterns,
        "insights": insights
    }
    
    return [TextContent(
        type="text",
        text=compact_json(result)
    )]


async def handle_resolve_conflict(args: Dict[str, Any]) -> List[TextContent]:
    """Resolve conflicting traits"""
    user_id = args["user_id"]
    # Accept 'traits' or 'conflicting_traits' as aliases for 'trait_objects'
    trait_objects = args.get("trait_objects") or args.get("traits") or args.get("conflicting_traits", [])
    if not trait_objects:
        return [TextContent(type="text", text=compact_json({"error": "'trait_objects' is required (list of conflicting trait strings)"}))]
    if isinstance(trait_objects, str):
        try:
            trait_objects = json.loads(trait_objects)
        except (json.JSONDecodeError, TypeError):
            trait_objects = [trait_objects]
    
    # Get all personality atoms
    all_atoms = await store.get_atoms_by_subject(user_id)
    
    # Filter to conflicting traits
    conflicting = [
        atom for atom in all_atoms
        if atom.atom_type in [AtomType.PERSONALITY_TRAIT, AtomType.COMMUNICATION_STYLE]
        and atom.object in trait_objects
    ]
    
    if len(conflicting) < 2:
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "no_conflict",
                "message": "Need at least 2 conflicting traits"
            })
        )]
    
    # Resolve
    winner, explanation = await conflict_resolver.resolve_with_explanation(
        user_id, conflicting
    )
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "status": "resolved",
            "winner": winner.object if winner else None,
            "explanation": explanation
        }, indent=2)
    )]


async def handle_extract_traits(args: Dict[str, Any]) -> List[TextContent]:
    """Extract personality traits from interaction"""
    user_id = args["user_id"]
    message = args["message"]
    ai_response = args.get("ai_response")
    user_reaction = args.get("user_reaction")
    
    # Extract traits
    traits = await personality_agent.personality_extractor.extract_from_interaction(
        user_id, message, ai_response, user_reaction
    )
    
    # Store traits
    for trait in traits:
        await store.add_atom(trait)
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "extracted_count": len(traits),
            "traits": [
                {
                    "type": trait.atom_type.value,
                    "predicate": trait.predicate,
                    "object": trait.object,
                    "confidence": trait.confidence
                }
                for trait in traits
            ]
        }, indent=2)
    )]


async def handle_adaptive_prompt(args: Dict[str, Any]) -> List[TextContent]:
    """Get adaptive prompt"""
    user_id = args["user_id"]
    message = args["message"]
    context = args.get("context")
    
    # Get personality and mood
    result = await personality_agent.interact(user_id, message, extract_personality=False)
    
    return [TextContent(
        type="text",
        text=result["adaptive_prompt"]
    )]


async def handle_personality_summary(args: Dict[str, Any]) -> List[TextContent]:
    """Get personality summary"""
    user_id = args["user_id"]
    
    summary = await personality_agent.get_personality_summary(user_id)
    
    return [TextContent(
        type="text",
        text=summary
    )]


async def handle_bootstrap_sample(args: Dict[str, Any]) -> List[TextContent]:
    """Bootstrap from sample data"""
    from mcp_server.bootstrap_pltm import ConversationAnalyzer
    
    user_id = args["user_id"]
    
    sample_convs = [
        {
            "messages": [
                {"role": "user", "content": "Explain PLTM conflict resolution"},
                {"role": "user", "content": "Too detailed, just key steps"},
                {"role": "user", "content": "Perfect, exactly what I needed"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Review this code"},
                {"role": "user", "content": "Don't make this so personalized"},
                {"role": "user", "content": "Great, that's helpful"}
            ]
        }
    ]
    
    analyzer = ConversationAnalyzer()
    total_atoms = 0
    
    for conv in sample_convs:
        analysis = analyzer.analyze_conversation(conv)
        
        for style_data in analysis["styles"]:
            atom = MemoryAtom(
                atom_type=AtomType.COMMUNICATION_STYLE,
                subject=user_id,
                predicate="prefers_style",
                object=style_data["style"],
                confidence=style_data["confidence"],
                strength=style_data["confidence"],
                provenance=Provenance.INFERRED,
                source_user=user_id,
                contexts=style_data.get("contexts", ["general"]),
                graph=GraphType.SUBSTANTIATED
            )
            await store.add_atom(atom)
            total_atoms += 1
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "status": "bootstrapped",
            "atoms_created": total_atoms,
            "message": f"Bootstrapped {total_atoms} personality atoms from sample data"
        }, indent=2)
    )]


async def handle_bootstrap_messages(args: Dict[str, Any]) -> List[TextContent]:
    """Bootstrap from conversation messages"""
    from mcp_server.bootstrap_pltm import ConversationAnalyzer
    
    user_id = args["user_id"]
    messages = args["messages"]
    
    analyzer = ConversationAnalyzer()
    analysis = analyzer.analyze_conversation(messages)
    
    total_atoms = 0
    
    for style_data in analysis["styles"]:
        atom = MemoryAtom(
            atom_type=AtomType.COMMUNICATION_STYLE,
            subject=user_id,
            predicate="prefers_style",
            object=style_data["style"],
            confidence=style_data["confidence"],
            strength=style_data["confidence"],
            provenance=Provenance.INFERRED,
            source_user=user_id,
            contexts=style_data.get("contexts", ["general"]),
            graph=GraphType.SUBSTANTIATED
        )
        await store.add_atom(atom)
        total_atoms += 1
    
    if analysis["moods"]:
        mood_data = analysis["moods"]
        atom = MemoryAtom(
            atom_type=AtomType.STATE,
            subject=user_id,
            predicate="typical_mood",
            object=mood_data["dominant_mood"],
            confidence=mood_data["confidence"],
            strength=mood_data["confidence"],
            provenance=Provenance.INFERRED,
            source_user=user_id,
            contexts=["historical"],
            graph=GraphType.SUBSTANTIATED
        )
        await store.add_atom(atom)
        total_atoms += 1
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "status": "bootstrapped",
            "atoms_created": total_atoms,
            "styles_found": len(analysis["styles"]),
            "mood_detected": analysis["moods"] is not None
        }, indent=2)
    )]


async def handle_track_evolution(args: Dict[str, Any]) -> List[TextContent]:
    """Track trait evolution over time"""
    from src.personality.temporal_tracker import TemporalPersonalityTracker
    
    tracker = TemporalPersonalityTracker(store)
    result = await tracker.track_trait_evolution(
        args["user_id"],
        args["trait"],
        int(args.get("window_days", 90))
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_predict_reaction(args: Dict[str, Any]) -> List[TextContent]:
    """Predict reaction to stimulus"""
    from src.personality.causal_graph import CausalGraphBuilder
    
    causal = CausalGraphBuilder(store)
    result = await causal.predict_reaction(args["user_id"], args["stimulus"])
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_meta_patterns(args: Dict[str, Any]) -> List[TextContent]:
    """Get cross-context meta patterns"""
    from src.personality.meta_patterns import MetaPatternDetector
    
    detector = MetaPatternDetector(store)
    patterns = await detector.detect_meta_patterns(args["user_id"])
    
    result = {
        "user_id": args["user_id"],
        "meta_patterns": [
            {
                "behavior": p.behavior,
                "contexts": p.contexts,
                "strength": p.strength,
                "is_core_trait": p.is_core_trait,
                "examples": p.examples[:2]
            }
            for p in patterns
        ],
        "core_traits": [p.behavior for p in patterns if p.is_core_trait]
    }
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learn_interaction(args: Dict[str, Any]) -> List[TextContent]:
    """Learn from an interaction"""
    from src.personality.interaction_dynamics import InteractionDynamicsLearner
    
    learner = InteractionDynamicsLearner(store)
    result = await learner.learn_from_interaction(
        args["user_id"],
        args["my_response"],
        args["user_reaction"]
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_predict_session(args: Dict[str, Any]) -> List[TextContent]:
    """Predict session from greeting"""
    from src.personality.predictive_model import PredictivePersonalityModel
    
    predictor = PredictivePersonalityModel(store)
    result = await predictor.predict_from_greeting(args["user_id"], args["greeting"])
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_self_model(args: Dict[str, Any]) -> List[TextContent]:
    """Get self-model for meta-cognition"""
    from src.personality.predictive_model import PredictivePersonalityModel
    
    predictor = PredictivePersonalityModel(store)
    result = await predictor.get_self_model(args["user_id"])
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_init_claude_session(args: Dict[str, Any]) -> List[TextContent]:
    """Initialize Claude personality session"""
    from src.personality.claude_personality import ClaudePersonality
    
    claude = ClaudePersonality(store)
    result = await claude.initialize_session(args["user_id"])
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_update_claude_style(args: Dict[str, Any]) -> List[TextContent]:
    """Update Claude's communication style"""
    from src.personality.claude_personality import ClaudePersonality
    
    claude = ClaudePersonality(store)
    result = await claude.update_style(
        args["user_id"],
        args["attribute"],
        args["value"],
        float(args.get("confidence", 0.8))
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learn_dynamic(args: Dict[str, Any]) -> List[TextContent]:
    """Learn interaction dynamic"""
    from src.personality.claude_personality import ClaudePersonality
    
    claude = ClaudePersonality(store)
    result = await claude.learn_what_works(
        args["user_id"],
        args["behavior"],
        args["works"],
        float(args.get("confidence", 0.8))
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_record_milestone(args: Dict[str, Any]) -> List[TextContent]:
    """Record collaboration milestone"""
    from src.personality.claude_personality import ClaudePersonality
    
    claude = ClaudePersonality(store)
    result = await claude.record_milestone(
        args["user_id"],
        args["description"],
        float(args.get("significance", 0.8))
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_add_vocabulary(args: Dict[str, Any]) -> List[TextContent]:
    """Add shared vocabulary"""
    from src.personality.claude_personality import ClaudePersonality
    
    claude = ClaudePersonality(store)
    result = await claude.add_shared_vocabulary(
        args["user_id"],
        args["term"],
        args["meaning"]
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_get_claude_personality(args: Dict[str, Any]) -> List[TextContent]:
    """Get Claude personality summary"""
    from src.personality.claude_personality import ClaudePersonality
    
    claude = ClaudePersonality(store)
    result = await claude.get_claude_personality_summary(args["user_id"])
    
    return [TextContent(type="text", text=result)]


async def handle_evolve_claude(args: Dict[str, Any]) -> List[TextContent]:
    """Evolve Claude personality from interaction"""
    from src.personality.claude_personality import ClaudePersonality
    
    claude = ClaudePersonality(store)
    result = await claude.evolve_from_interaction(
        args["user_id"],
        args["my_response_style"],
        args["user_reaction"],
        args["was_positive"]
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_check_pltm(args: Dict[str, Any]) -> List[TextContent]:
    """Quick check if user has PLTM data - call FIRST in conversations"""
    from src.personality.claude_personality import ClaudePersonality
    
    user_id = args["user_id"]
    
    try:
        claude = ClaudePersonality(store)
        
        # Load basic info without full init
        context = await claude._load_shared_context(user_id)
        dynamics = await claude._load_interaction_dynamics(user_id)
        style = await claude._load_claude_style(user_id)
        
        has_data = context.session_count > 0 or len(dynamics.works_well) > 0

        # Real memory counts — makes an empty store obvious instead of
        # implying data exists for a user_id that has none.
        memory_count = atom_count = 0
        try:
            if typed_memory_store:
                memory_count = (await typed_memory_store.get_stats(user_id)).get("total", 0)
        except Exception:
            pass
        try:
            if store:
                atom_count = len(await store.get_atoms(user_id))
        except Exception:
            pass
        has_data = has_data or memory_count > 0 or atom_count > 0

        result = {
            "available": has_data,
            "user_id": user_id,
            "memory_count": memory_count,
            "atom_count": atom_count,
            "sessions": context.session_count,
            "trust_level": dynamics.trust_level,
            "should_init": has_data,
            "quick_summary": {
                "verbosity": style.verbosity,
                "initiative": style.initiative,
                "works_well_count": len(dynamics.works_well),
                "avoid_count": len(dynamics.avoid),
                "projects": context.projects[:3] if context.projects else []
            },
            "instruction": "Call init_claude_session to load full personality" if has_data else "No existing data - new user"
        }
        
        return [TextContent(type="text", text=compact_json(result))]
        
    except Exception as e:
        return [TextContent(type="text", text=compact_json({
            "available": False,
            "should_init": False,
            "error": str(e)
        }))]


# === SENSORY HANDLERS ===

# Global sensory system state
_sensory_camera = None
_sensory_microphone = None
_sensory_buffer = None
_sensory_integrator = None


def _get_sensory_integrator():
    """Lazy SensoryMemoryIntegrator — turns buffered observations into atoms."""
    global _sensory_integrator
    if _sensory_integrator is None and store is not None:
        from src.sensory.memory_integration import SensoryMemoryIntegrator
        _sensory_integrator = SensoryMemoryIntegrator(store)
    return _sensory_integrator

async def handle_sensory_start(args: Dict[str, Any]) -> List[TextContent]:
    """Start camera and microphone capture"""
    global _sensory_camera, _sensory_microphone, _sensory_buffer
    
    from src.sensory import CameraCapture, MicrophoneCapture, SensoryBuffer
    
    enable_camera = args.get("camera", True)
    enable_microphone = args.get("microphone", True)
    fps = args.get("fps", 1.0)
    
    # Initialize buffer if needed
    if _sensory_buffer is None:
        _sensory_buffer = SensoryBuffer(max_size=100, max_age_seconds=60.0)
    
    results = {"camera": False, "microphone": False, "errors": []}
    
    # Start camera
    if enable_camera:
        try:
            if _sensory_camera is None:
                _sensory_camera = CameraCapture(camera_id=0, fps=fps)
            
            if _sensory_camera.start():
                results["camera"] = True
            else:
                results["errors"].append("Failed to start camera")
        except Exception as e:
            results["errors"].append(f"Camera error: {str(e)}")
    
    # Start microphone
    if enable_microphone:
        try:
            if _sensory_microphone is None:
                _sensory_microphone = MicrophoneCapture(sample_rate=16000, chunk_duration=5.0)
            
            if _sensory_microphone.start():
                results["microphone"] = True
            else:
                results["errors"].append("Failed to start microphone")
        except Exception as e:
            results["errors"].append(f"Microphone error: {str(e)}")
    
    return [TextContent(type="text", text=compact_json({
        "status": "started" if (results["camera"] or results["microphone"]) else "failed",
        "camera_active": results["camera"],
        "microphone_active": results["microphone"],
        "errors": results["errors"],
        "message": "Sensory capture started. Camera and microphone are now active." if not results["errors"] else "Some sensors failed to start."
    }))]


async def handle_sensory_stop(args: Dict[str, Any]) -> List[TextContent]:
    """Stop all sensory capture"""
    global _sensory_camera, _sensory_microphone, _sensory_buffer
    
    stopped = []
    
    if _sensory_camera:
        _sensory_camera.stop()
        _sensory_camera = None
        stopped.append("camera")
    
    if _sensory_microphone:
        _sensory_microphone.stop()
        _sensory_microphone = None
        stopped.append("microphone")
    
    if _sensory_buffer:
        _sensory_buffer.clear()
    
    return [TextContent(type="text", text=compact_json({
        "status": "stopped",
        "stopped": stopped,
        "message": f"Sensory capture stopped: {', '.join(stopped)}" if stopped else "No active sensors to stop"
    }))]


async def handle_sensory_look(args: Dict[str, Any]) -> List[TextContent]:
    """Get current visual observation"""
    global _sensory_camera, _sensory_buffer
    
    if not _sensory_camera or not _sensory_camera.running:
        return [TextContent(type="text", text=compact_json({
            "error": "Camera not active. Call sensory_start first."
        }))]
    
    observation = _sensory_camera.get_current_observation()
    
    if not observation:
        return [TextContent(type="text", text=compact_json({
            "error": "No visual data available yet. Wait a moment and try again."
        }))]
    
    # Store in buffer
    if _sensory_buffer:
        _sensory_buffer.add_visual(observation)

    # Persist to memory atoms when there's something worth remembering
    integrator = _get_sensory_integrator()
    if integrator and (observation.get("n_faces", 0) > 0
                       or observation.get("scene", {}).get("motion")):
        try:
            await integrator.store_visual_observation(
                observation, user_id=args.get("user_id", "claude"))
        except Exception as e:
            logger.warning(f"Sensory→memory integration failed: {e}")

    return [TextContent(type="text", text=compact_json({
        "n_faces": observation["n_faces"],
        "faces": observation["faces"],
        "scene": observation["scene"]["desc"]
    }))]


async def handle_sensory_listen(args: Dict[str, Any]) -> List[TextContent]:
    """Get current audio observation"""
    global _sensory_microphone, _sensory_buffer
    
    if not _sensory_microphone or not _sensory_microphone.running:
        return [TextContent(type="text", text=compact_json({
            "error": "Microphone not active. Call sensory_start first."
        }))]
    
    observation = _sensory_microphone.get_current_observation()
    
    if not observation:
        return [TextContent(type="text", text=compact_json({
            "error": "No audio data available yet. Wait a moment and try again."
        }))]
    
    # Store in buffer
    if _sensory_buffer:
        _sensory_buffer.add_audio(observation)

    # Persist to memory atoms when there's something worth remembering —
    # speech, a transcript, or non-quiet ambient environment
    ambient = observation.get("ambient") or {}
    integrator = _get_sensory_integrator()
    if integrator and (observation.get("speech") or observation.get("text")
                       or ambient.get("vol", "quiet") != "quiet"):
        try:
            await integrator.store_audio_observation(
                observation, user_id=args.get("user_id", "claude"))
        except Exception as e:
            logger.warning(f"Sensory→memory integration failed: {e}")

    return [TextContent(type="text", text=compact_json({
        "speech": observation["speech"],
        "text": observation["text"],
        "tone": observation["tone"]
    }))]


async def handle_sensory_state(args: Dict[str, Any]) -> List[TextContent]:
    """Get unified multimodal state"""
    global _sensory_buffer
    
    if not _sensory_buffer:
        return [TextContent(type="text", text=compact_json({
            "error": "Sensory buffer not initialized. Call sensory_start first."
        }))]
    
    state = _sensory_buffer.get_current_state()
    
    return [TextContent(type="text", text=compact_json(state))]


async def handle_sensory_history(args: Dict[str, Any]) -> List[TextContent]:
    """Get recent sensory timeline"""
    global _sensory_buffer
    
    if not _sensory_buffer:
        return [TextContent(type="text", text=compact_json({
            "error": "Sensory buffer not initialized. Call sensory_start first."
        }))]
    
    seconds = args.get("seconds", 10.0)
    modality = args.get("modality", "both")
    
    # Map "both" to None for buffer API
    modality_filter = None if modality == "both" else modality

    timeline = _sensory_buffer.get_timeline(seconds=seconds, modality=modality_filter)
    
    return [TextContent(type="text", text=compact_json(timeline))]


async def handle_sensory_diagnostics(args: Dict[str, Any]) -> List[TextContent]:
    """Get Whisper transcription diagnostics"""
    from src.sensory.whisper_diagnostics import WhisperDiagnostics
    
    count = args.get("count", 10)
    attempts = WhisperDiagnostics.get_recent_attempts(n=count)
    
    if not attempts:
        return [TextContent(type="text", text=compact_json({
            "message": "No transcription attempts recorded yet. Speak and call sensory_listen to generate diagnostics.",
            "attempts": []
        }))]
    
    return [TextContent(type="text", text=compact_json({
        "total_attempts": len(attempts),
        "attempts": attempts
    }))]


async def handle_pltm_mode(args: Dict[str, Any]) -> List[TextContent]:
    """
    PLTM Mode trigger - full auto-init when user says 'PLTM mode' or similar.
    
    Returns everything Claude needs to adapt immediately.
    """
    from src.personality.claude_personality import ClaudePersonality
    
    user_id = args["user_id"]
    trigger = args.get("trigger_phrase", "PLTM mode")
    
    claude = ClaudePersonality(store)
    
    # Full initialization
    session = await claude.initialize_session(user_id)
    
    # Get personality summary
    summary = await claude.get_claude_personality_summary(user_id)
    
    # Build instruction set for Claude
    style = session["claude_style"]
    dynamics = session["interaction_dynamics"]
    
    instructions = []
    
    # Verbosity instruction
    if style["verbosity"] == "minimal":
        instructions.append("Be concise and direct. Skip verbose explanations.")
    elif style["verbosity"] == "moderate":
        instructions.append("Balance detail with brevity.")
    
    # Initiative instruction
    if style["initiative"] == "high" or style["initiative"] == "very_high":
        instructions.append("Execute immediately without asking permission. User prefers action over discussion.")
    
    # Energy matching
    if style["energy_matching"]:
        instructions.append("Match user's energy level - mirror excitement, match focus.")
    
    # What works
    for behavior in dynamics["works_well"][:5]:
        instructions.append(f"DO: {behavior}")
    
    # What to avoid
    for behavior in dynamics["avoid"][:5]:
        instructions.append(f"AVOID: {behavior}")
    
    result = {
        "mode": "PLTM_ACTIVE",
        "trigger": trigger,
        "user_id": user_id,
        "session_id": session["session_id"],
        "sessions_together": session["shared_context"]["session_count"],
        "trust_level": f"{dynamics['trust_level']:.0%}",
        "style": style,
        "instructions_for_claude": instructions,
        "shared_vocabulary": dynamics["shared_vocabulary"],
        "recent_projects": session["shared_context"]["projects"][:5],
        "recent_milestones": session["shared_context"]["recent_milestones"],
        "personality_summary": summary
    }
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_deep_analysis(args: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive deep personality analysis"""
    from src.personality.deep_analysis import DeepPersonalityAnalyzer
    
    analyzer = DeepPersonalityAnalyzer(store)
    result = await analyzer.analyze_all(args["user_id"])
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_enrich_personality(args: Dict[str, Any]) -> List[TextContent]:
    """Build rich, nuanced Claude personality"""
    from src.personality.rich_personality import RichClaudePersonality
    
    enricher = RichClaudePersonality(store)
    result = await enricher.build_rich_personality(args["user_id"])
    
    # Store if requested
    if args.get("store_results", True):
        await enricher.store_rich_personality(args["user_id"], result)
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learn_url(args: Dict[str, Any]) -> List[TextContent]:
    """Learn from any URL content"""
    if not args.get("content"):
        return [TextContent(type="text", text=compact_json({
            "error": "Missing required 'content' parameter. You cannot fetch URLs yourself. Ask the user to paste the page content, then call this tool again with both 'url' and 'content'."
        }))]
    
    from src.learning.universal_learning import UniversalLearningSystem, SourceType
    
    learner = UniversalLearningSystem(store)
    
    source_type = None
    if args.get("source_type"):
        try:
            source_type = SourceType(args["source_type"])
        except ValueError:
            pass
    
    result = await learner.learn_from_url(
        args["url"],
        args["content"],
        source_type
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learn_paper(args: Dict[str, Any]) -> List[TextContent]:
    """Learn from research paper"""
    from src.learning.universal_learning import UniversalLearningSystem
    
    learner = UniversalLearningSystem(store)
    result = await learner.learn_from_paper(
        args["paper_id"],
        args["title"],
        args["abstract"],
        args["content"],
        args["authors"],
        args.get("publication_date")
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learn_code(args: Dict[str, Any]) -> List[TextContent]:
    """Learn from code repository"""
    from src.learning.universal_learning import UniversalLearningSystem
    
    learner = UniversalLearningSystem(store)
    result = await learner.learn_from_code(
        args["repo_url"],
        args["repo_name"],
        args.get("description", ""),
        args["languages"],
        args["code_samples"]
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learning_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get learning statistics"""
    from src.learning.universal_learning import UniversalLearningSystem
    
    learner = UniversalLearningSystem(store)
    result = await learner.get_learning_stats()
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_batch_wikipedia(args: Dict[str, Any]) -> List[TextContent]:
    """Batch ingest Wikipedia articles"""
    from src.learning.batch_ingestion import BatchIngestionPipeline
    
    # Accept 'titles' shorthand — auto-build articles list
    articles = args.get("articles")
    titles = args.get("titles")
    
    # Handle stringified JSON (Claude sometimes sends '["title"]' as a string)
    if isinstance(titles, str):
        try:
            titles = json.loads(titles)
        except (json.JSONDecodeError, TypeError):
            titles = [titles]
    if isinstance(articles, str):
        try:
            articles = json.loads(articles)
        except (json.JSONDecodeError, TypeError):
            articles = []
    
    if titles and not articles:
        articles = [
            {"title": t, "content": f"Wikipedia article: {t}", "url": f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}"}
            for t in titles
        ]
    
    if not articles:
        return [TextContent(type="text", text=compact_json({"error": "Pass 'titles' (list of strings) or 'articles' (list of {title, content, url})"}))]
    
    pipeline = BatchIngestionPipeline(store)
    result = await pipeline.ingest_wikipedia_articles(articles)
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_batch_papers(args: Dict[str, Any]) -> List[TextContent]:
    """Batch ingest research papers"""
    from src.learning.batch_ingestion import BatchIngestionPipeline
    
    # Accept 'arxiv_ids' shorthand — auto-build papers list
    papers = args.get("papers")
    arxiv_ids = args.get("arxiv_ids")
    
    # Handle stringified JSON (Claude sometimes sends '["id"]' as a string)
    if isinstance(arxiv_ids, str):
        try:
            arxiv_ids = json.loads(arxiv_ids)
        except (json.JSONDecodeError, TypeError):
            arxiv_ids = [arxiv_ids]
    if isinstance(papers, str):
        try:
            papers = json.loads(papers)
        except (json.JSONDecodeError, TypeError):
            papers = []
    
    if arxiv_ids and not papers:
        papers = [
            {"arxiv_id": aid, "title": f"arXiv:{aid}", "abstract": ""}
            for aid in arxiv_ids
        ]
    
    if not papers:
        return [TextContent(type="text", text=compact_json({"error": "Pass 'arxiv_ids' (list of strings) or 'papers' (list of {arxiv_id, title, abstract})"}))]
    
    pipeline = BatchIngestionPipeline(store)
    result = await pipeline.ingest_arxiv_papers(papers)
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_batch_repos(args: Dict[str, Any]) -> List[TextContent]:
    """Batch ingest GitHub repos"""
    from src.learning.batch_ingestion import BatchIngestionPipeline
    
    pipeline = BatchIngestionPipeline(store)
    result = await pipeline.ingest_github_repos(args["repos"])
    
    return [TextContent(type="text", text=compact_json(result))]


# Global continuous learning loop instance
_continuous_learner = None

async def handle_learning_schedule(args: Dict[str, Any]) -> List[TextContent]:
    """Get learning schedule status"""
    from src.learning.continuous_learning import ContinuousLearningLoop
    
    global _continuous_learner
    if _continuous_learner is None:
        _continuous_learner = ContinuousLearningLoop(store)
    
    result = _continuous_learner.get_schedule_status()
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_run_task(args: Dict[str, Any]) -> List[TextContent]:
    """Run a specific learning task"""
    from src.learning.continuous_learning import ContinuousLearningLoop
    
    global _continuous_learner
    if _continuous_learner is None:
        _continuous_learner = ContinuousLearningLoop(store)
    
    # Accept 'task' as alias for 'task_name'
    task_name = args.get("task_name") or args.get("task", "")
    if not task_name:
        return [TextContent(type="text", text=compact_json({"error": "'task_name' is required: arxiv_latest|github_trending|news_feed|knowledge_consolidation"}))]
    result = await _continuous_learner.run_single_task(task_name)
    
    return [TextContent(type="text", text=compact_json(result))]


# Global synthesizer instance
_synthesizer = None

async def handle_synthesis(args: Dict[str, Any]) -> List[TextContent]:
    """Run cross-domain synthesis"""
    from src.learning.cross_domain_synthesis import CrossDomainSynthesizer
    
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = CrossDomainSynthesizer(store)
    
    result = await _synthesizer.synthesize_all()
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_universal_principles(args: Dict[str, Any]) -> List[TextContent]:
    """Get discovered universal principles"""
    from src.learning.cross_domain_synthesis import CrossDomainSynthesizer
    
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = CrossDomainSynthesizer(store)
    
    result = await _synthesizer.query_universal_principles()
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_transfer_suggestions(args: Dict[str, Any]) -> List[TextContent]:
    """Get transfer suggestions between domains"""
    from src.learning.cross_domain_synthesis import CrossDomainSynthesizer
    
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = CrossDomainSynthesizer(store)
    
    result = await _synthesizer.get_transfer_suggestions(
        args["from_domain"],
        args["to_domain"]
    )
    
    return [TextContent(type="text", text=compact_json(result))]


# ── Autonomous Learning Handlers ─────────────────────────────────────────────

async def handle_auto_learn_schedules(args: Dict[str, Any]) -> List[TextContent]:
    result = await autonomous_learning.get_schedules()
    return [TextContent(type="text", text=compact_json(result))]

async def handle_auto_learn_run_due(args: Dict[str, Any]) -> List[TextContent]:
    result = await autonomous_learning.run_due_tasks()
    return [TextContent(type="text", text=compact_json(result))]

async def handle_auto_learn_run_task(args: Dict[str, Any]) -> List[TextContent]:
    result = await autonomous_learning.run_task(args["task"])
    return [TextContent(type="text", text=compact_json(result))]

async def handle_auto_learn_digest(args: Dict[str, Any]) -> List[TextContent]:
    since = float(args.get("since_hours", 24))
    result = await autonomous_learning.get_learning_digest(since)
    return [TextContent(type="text", text=compact_json(result))]

async def handle_auto_learn_phi_history(args: Dict[str, Any]) -> List[TextContent]:
    limit = int(args.get("limit", 20))
    result = await autonomous_learning.get_phi_history(limit)
    return [TextContent(type="text", text=compact_json(result))]

async def handle_auto_learn_update_schedule(args: Dict[str, Any]) -> List[TextContent]:
    result = await autonomous_learning.update_schedule(
        task=args["task"],
        interval_hours=float(args["interval_hours"]) if "interval_hours" in args else None,
        enabled=args.get("enabled"),
    )
    return [TextContent(type="text", text=compact_json(result))]


# ── Active Learning Handlers ─────────────────────────────────────────────────

async def handle_search_and_learn(args: Dict[str, Any]) -> List[TextContent]:
    result = await active_learning.search_and_learn(
        query=args["query"],
        source=args.get("source", "arxiv"),
        max_results=int(args.get("max_results", 5)),
    )
    return [TextContent(type="text", text=compact_json(result))]

async def handle_propose_hypothesis(args: Dict[str, Any]) -> List[TextContent]:
    result = await active_learning.propose_hypothesis(
        statement=args["statement"],
        domains=args.get("domains"),
        prior_confidence=float(args.get("prior_confidence", 0.5)),
        verification_strategy=args.get("verification_strategy", ""),
    )
    return [TextContent(type="text", text=compact_json(result))]

async def handle_submit_evidence(args: Dict[str, Any]) -> List[TextContent]:
    # Accept 'type'/'evidence_type' as alias for 'direction'
    direction = args.get("direction") or args.get("type") or args.get("evidence_type", "")
    if direction not in ("for", "against"):
        return [TextContent(type="text", text=compact_json({"error": "'direction' is required: 'for' or 'against'"}))]
    result = await active_learning.submit_evidence(
        hypothesis_id=args["hypothesis_id"],
        evidence=args.get("evidence") or args.get("description", ""),
        direction=direction,
        strength=float(args.get("strength", 0.5)),
        source_url=args.get("source_url", ""),
    )
    return [TextContent(type="text", text=compact_json(result))]

async def handle_get_hypotheses(args: Dict[str, Any]) -> List[TextContent]:
    result = await active_learning.get_active_hypotheses(
        limit=int(args.get("limit", 10)),
    )
    return [TextContent(type="text", text=compact_json(result))]


# ── Hierarchical Memory + Prefetch + Contradiction Handlers ──────────────────

async def handle_memory_tree(args: Dict[str, Any]) -> List[TextContent]:
    result = await memory_index.get_tree()
    return [TextContent(type="text", text=compact_json(result))]

async def handle_memory_rebuild_index(args: Dict[str, Any]) -> List[TextContent]:
    result = await memory_index.rebuild_index()
    return [TextContent(type="text", text=compact_json(result))]

async def handle_memory_smart_retrieve(args: Dict[str, Any]) -> List[TextContent]:
    # Record tool call for trajectory prediction
    prefetch_engine.record_tool_call("memory_smart_retrieve")
    result = await prefetch_engine.retrieve_smart(
        query=args["query"],
        top_k=int(args.get("top_k", 20)),
    )
    return [TextContent(type="text", text=compact_json(result))]

async def handle_memory_prefetch(args: Dict[str, Any]) -> List[TextContent]:
    result = await prefetch_engine.prefetch()
    return [TextContent(type="text", text=compact_json(result))]

async def handle_memory_prefetch_stats(args: Dict[str, Any]) -> List[TextContent]:
    result = prefetch_engine.get_stats()
    return [TextContent(type="text", text=compact_json(result))]

async def handle_auto_resolve_contradictions(args: Dict[str, Any]) -> List[TextContent]:
    result = await contradiction_resolver.scan_and_resolve(
        limit=int(args.get("limit", 200)),
    )
    return [TextContent(type="text", text=compact_json(result))]

async def handle_get_unresolved_contradictions(args: Dict[str, Any]) -> List[TextContent]:
    result = await contradiction_resolver.get_unresolved(
        limit=int(args.get("limit", 20)),
    )
    return [TextContent(type="text", text=compact_json(result))]


# ── Emergence Detector Handlers ──────────────────────────────────────────────

async def handle_emergence_observe(args: Dict[str, Any]) -> List[TextContent]:
    result = await emergence_detector.observe()
    return [TextContent(type="text", text=compact_json(result))]

async def handle_emergence_record_outcome(args: Dict[str, Any]) -> List[TextContent]:
    # Accept 'prediction_id' as alias for 'pred_id'
    pred_id = args.get("pred_id") or args.get("prediction_id", "")
    if not pred_id:
        return [TextContent(type="text", text=compact_json({"error": "'pred_id' is required"}))]
    # Resolve did_emerge from multiple possible params
    did_emerge = args.get("did_emerge")
    if did_emerge is None:
        did_emerge = args.get("emerged")
    if did_emerge is None:
        # Infer from 'outcome' string if provided (e.g. "breakthrough" → True, "nothing" → False)
        outcome_str = args.get("outcome", "")
        if outcome_str:
            did_emerge = outcome_str.lower() not in ("no", "false", "nothing", "none", "failed", "0")
        else:
            return [TextContent(type="text", text=compact_json({"error": "'did_emerge' (boolean) is required"}))]
    details = args.get("details") or args.get("outcome", "")
    result = await emergence_detector.record_outcome(
        pred_id=pred_id,
        did_emerge=bool(did_emerge),
        details=details,
    )
    return [TextContent(type="text", text=compact_json(result))]

async def handle_emergence_history(args: Dict[str, Any]) -> List[TextContent]:
    result = await emergence_detector.get_history(
        limit=int(args.get("limit", 20)),
    )
    return [TextContent(type="text", text=compact_json(result))]

async def handle_emergence_predictions(args: Dict[str, Any]) -> List[TextContent]:
    result = await emergence_detector.get_predictions(
        include_pending=bool(args.get("include_pending", True)),
    )
    return [TextContent(type="text", text=compact_json(result))]

async def handle_emergence_dashboard(args: Dict[str, Any]) -> List[TextContent]:
    result = await emergence_detector.get_dashboard()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learn_conversation(args: Dict[str, Any]) -> List[TextContent]:
    """Learn from conversation"""
    from src.learning.continuous_learning import ManualLearningTrigger
    
    trigger = ManualLearningTrigger(store)
    result = await trigger.learn_from_conversation(
        args["messages"],
        args["topic"],
        args["user_id"]
    )
    
    return [TextContent(type="text", text=compact_json(result))]


# ============================================================================
# PLTM 2.0 - Universal Optimization Principles
# ============================================================================

# Global instances for PLTM 2.0 systems
_quantum_memory = None
_attention_retrieval = None
_knowledge_graph = None
_self_improver = None


def get_attention_retrieval():
    """Shared AttentionMemoryRetrieval — real attention module (semantic +
    recency + confidence scoring, softmax, cache)."""
    global _attention_retrieval
    if _attention_retrieval is None:
        from src.memory.attention_retrieval import AttentionMemoryRetrieval
        _attention_retrieval = AttentionMemoryRetrieval(store)
    return _attention_retrieval


async def handle_quantum_add(args: Dict[str, Any]) -> List[TextContent]:
    """Add state to quantum superposition"""
    from src.memory.quantum_superposition import QuantumMemorySystem
    
    global _quantum_memory
    if _quantum_memory is None:
        _quantum_memory = QuantumMemorySystem(store)
    
    result = await _quantum_memory.add_to_superposition(
        args["subject"],
        args["predicate"],
        args["value"],
        args["confidence"],
        args["source"]
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_quantum_query(args: Dict[str, Any]) -> List[TextContent]:
    """Query superposition with collapse"""
    from src.memory.quantum_superposition import QuantumMemorySystem
    
    global _quantum_memory
    if _quantum_memory is None:
        _quantum_memory = QuantumMemorySystem(store)
    
    result = await _quantum_memory.query_with_collapse(
        args["subject"],
        args["predicate"],
        args.get("context")
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_quantum_peek(args: Dict[str, Any]) -> List[TextContent]:
    """Peek at superposition without collapse"""
    from src.memory.quantum_superposition import QuantumMemorySystem
    
    global _quantum_memory
    if _quantum_memory is None:
        _quantum_memory = QuantumMemorySystem(store)
    
    result = await _quantum_memory.peek_superposition(
        args["subject"],
        args["predicate"]
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_attention_retrieve(args: Dict[str, Any]) -> List[TextContent]:
    """Attention-weighted memory retrieval — real attention module."""
    ar = get_attention_retrieval()
    result = await ar.retrieve_with_attention(
        user_id=args.get("user_id", "claude"),
        query=args.get("query", ""),
        top_k=int(args.get("top_k", 10)),
        domain=args.get("domain"),
        temperature=float(args.get("temperature", 1.0)),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_attention_multihead(args: Dict[str, Any]) -> List[TextContent]:
    """Multi-head attention retrieval — real attention module
    (semantic / recency / confidence / balanced heads)."""
    ar = get_attention_retrieval()
    result = await ar.multi_head_attention(
        user_id=args.get("user_id", "claude"),
        query=args.get("query", ""),
        num_heads=int(args.get("num_heads", 4)),
        top_k_per_head=int(args.get("top_k_per_head", 5)),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_knowledge_add(args: Dict[str, Any]) -> List[TextContent]:
    """Add concept to knowledge graph"""
    from src.memory.knowledge_graph import KnowledgeNetworkGraph
    
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeNetworkGraph(store)
    
    result = await _knowledge_graph.add_concept(
        args["concept"],
        args["domain"],
        args.get("related_concepts")
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_knowledge_path(args: Dict[str, Any]) -> List[TextContent]:
    """Find path between concepts"""
    from src.memory.knowledge_graph import KnowledgeNetworkGraph
    
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeNetworkGraph(store)
    
    result = await _knowledge_graph.find_path(
        args["from_concept"],
        args["to_concept"]
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_knowledge_bridges(args: Dict[str, Any]) -> List[TextContent]:
    """Find bridge concepts"""
    from src.memory.knowledge_graph import KnowledgeNetworkGraph
    
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeNetworkGraph(store)
    
    result = await _knowledge_graph.find_bridges(int(args.get("top_k", 10)))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_knowledge_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get knowledge graph stats"""
    from src.memory.knowledge_graph import KnowledgeNetworkGraph
    
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeNetworkGraph(store)
    
    result = await _knowledge_graph.get_network_stats()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_improve_cycle(args: Dict[str, Any]) -> List[TextContent]:
    """Run self-improvement cycle"""
    from src.meta.recursive_improvement import RecursiveSelfImprovement
    
    global _self_improver
    if _self_improver is None:
        _self_improver = RecursiveSelfImprovement(store)
    
    result = await _self_improver.run_improvement_cycle()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_meta_learn(args: Dict[str, Any]) -> List[TextContent]:
    """Meta-learn from improvements"""
    from src.meta.recursive_improvement import RecursiveSelfImprovement
    
    global _self_improver
    if _self_improver is None:
        _self_improver = RecursiveSelfImprovement(store)
    
    result = await _self_improver.meta_learn()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_improve_history(args: Dict[str, Any]) -> List[TextContent]:
    """Get improvement history"""
    from src.meta.recursive_improvement import RecursiveSelfImprovement
    
    global _self_improver
    if _self_improver is None:
        _self_improver = RecursiveSelfImprovement(store)
    
    result = await _self_improver.get_improvement_history()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_quantum_cleanup(args: Dict[str, Any]) -> List[TextContent]:
    """Cleanup old quantum states"""
    from src.memory.quantum_superposition import QuantumMemorySystem
    
    global _quantum_memory
    if _quantum_memory is None:
        _quantum_memory = QuantumMemorySystem(store)
    
    result = await _quantum_memory.cleanup_old_states()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_quantum_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get quantum memory stats"""
    from src.memory.quantum_superposition import QuantumMemorySystem
    
    global _quantum_memory
    if _quantum_memory is None:
        _quantum_memory = QuantumMemorySystem(store)
    
    result = await _quantum_memory.get_stats()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_attention_clear_cache(args: Dict[str, Any]) -> List[TextContent]:
    """Clear attention cache"""
    count = get_attention_retrieval().clear_cache()
    return [TextContent(type="text", text=compact_json({"cleared": count}))]


# Global criticality instance
_criticality = None

async def handle_criticality_state(args: Dict[str, Any]) -> List[TextContent]:
    """Get criticality state - lightweight direct SQL"""
    import math
    
    if not store._conn:
        return [TextContent(type="text", text=compact_json({"error": "DB not connected"}))]
    
    # Get confidence stats for entropy calculation
    cursor = await store._conn.execute(
        "SELECT confidence, predicate FROM atoms WHERE graph = 'substantiated'"
    )
    rows = await cursor.fetchall()
    
    if not rows:
        return [TextContent(type="text", text=compact_json({
            "entropy": 0.5, "integration": 0.5, "ratio": 1.0, "zone": "critical", "n": 0
        }))]
    
    # Entropy: confidence variance + domain diversity
    confidences = [r[0] for r in rows]
    mean_conf = sum(confidences) / len(confidences)
    variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
    conf_entropy = min(1.0, variance * 4)  # Scale variance to 0-1
    
    # Domain entropy
    domains = {}
    for r in rows:
        domains[r[1]] = domains.get(r[1], 0) + 1
    
    total = len(rows)
    domain_entropy = 0.0
    for count in domains.values():
        p = count / total
        if p > 0:
            domain_entropy -= p * math.log2(p)
    max_ent = math.log2(len(domains)) if len(domains) > 1 else 1.0
    norm_domain_ent = domain_entropy / max_ent if max_ent > 0 else 0.0
    
    entropy = (conf_entropy * 0.4 + norm_domain_ent * 0.6)
    
    # Integration: mean confidence + domain connectivity proxy
    integration = mean_conf * 0.7 + (1 - norm_domain_ent) * 0.3
    
    # Criticality ratio
    ratio = entropy / integration if integration > 0 else 1.0
    
    # Zone classification
    if ratio < 0.8:
        zone = "subcritical"
    elif ratio > 1.2:
        zone = "supercritical"
    else:
        zone = "critical"
    
    return [TextContent(type="text", text=compact_json({
        "entropy": round(entropy, 3),
        "integration": round(integration, 3),
        "ratio": round(ratio, 3),
        "zone": zone,
        "n": len(rows),
        "domains": len(domains)
    }))]


async def handle_criticality_recommend(args: Dict[str, Any]) -> List[TextContent]:
    """Get criticality recommendation"""
    from src.meta.criticality import SelfOrganizedCriticality
    
    global _criticality
    if _criticality is None:
        _criticality = SelfOrganizedCriticality(store)
    
    result = await _criticality.get_adjustment_recommendation()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_criticality_adjust(args: Dict[str, Any]) -> List[TextContent]:
    """Auto-adjust toward criticality"""
    from src.meta.criticality import SelfOrganizedCriticality
    
    global _criticality
    if _criticality is None:
        _criticality = SelfOrganizedCriticality(store)
    
    result = await _criticality.auto_adjust()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_criticality_history(args: Dict[str, Any]) -> List[TextContent]:
    """Get criticality history"""
    from src.meta.criticality import SelfOrganizedCriticality
    
    global _criticality
    if _criticality is None:
        _criticality = SelfOrganizedCriticality(store)
    
    result = await _criticality.get_criticality_history()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_add_provenance(args: Dict[str, Any]) -> List[TextContent]:
    """Add provenance for a claim"""
    import hashlib
    from datetime import datetime
    
    claim_id = args.get("claim_id")
    source_type = args.get("source_type")
    source_url = args.get("source_url")
    quoted_span = args.get("quoted_span")
    confidence = float(args.get("confidence", 0.5))
    
    # Generate provenance ID and content hash
    prov_id = f"prov_{source_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    content_hash = hashlib.sha256(quoted_span.encode()).hexdigest()
    
    await store.insert_provenance(
        provenance_id=prov_id,
        claim_id=claim_id,
        source_type=source_type,
        source_url=source_url,
        source_title=args.get("source_title", ""),
        quoted_span=quoted_span,
        page_or_section=args.get("page_or_section"),
        accessed_at=int(datetime.now().timestamp()),
        content_hash=content_hash,
        confidence=confidence,
        authors=args.get("authors"),
        arxiv_id=args.get("arxiv_id"),
        doi=args.get("doi"),
        commit_sha=args.get("commit_sha"),
        file_path=args.get("file_path"),
        line_range=args.get("line_range")
    )
    
    return [TextContent(type="text", text=compact_json({
        "ok": True, "id": prov_id, "type": source_type, "hash": content_hash[:16]
    }))]


async def handle_get_provenance(args: Dict[str, Any]) -> List[TextContent]:
    """Get provenance for a claim"""
    claim_id = args.get("claim_id")
    provs = await store.get_provenance_for_claim(claim_id)
    
    # Compact format
    result = [{
        "type": p["source_type"],
        "url": p["source_url"][:60],
        "quote": p["quoted_span"][:100],
        "conf": p["confidence"]
    } for p in provs]
    
    return [TextContent(type="text", text=compact_json({"n": len(result), "provs": result}))]


async def handle_provenance_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get provenance statistics"""
    stats = await store.get_provenance_stats()
    return [TextContent(type="text", text=compact_json(stats))]


async def handle_unverified_claims(args: Dict[str, Any]) -> List[TextContent]:
    """Get unverified claims"""
    unverified = await store.get_unverified_claims()
    return [TextContent(type="text", text=compact_json({
        "n": len(unverified), 
        "claims": unverified[:20]  # Limit to 20 for token efficiency
    }))]


async def handle_mmr_retrieve(args: Dict[str, Any]) -> List[TextContent]:
    """MMR retrieval for diverse context selection — real attention module
    (sentence-transformer embeddings, Carbonell & Goldstein MMR)."""
    ar = get_attention_retrieval()
    result = await ar.mmr_retrieve(
        user_id=args.get("user_id", "claude"),
        query=args.get("query", ""),
        top_k=int(args.get("top_k", 5)),
        lambda_param=float(args.get("lambda_param", 0.6)),
        min_dissim=float(args.get("min_dissim", 0.25)),
    )
    return [TextContent(type="text", text=compact_json(result))]


# === ACTION ACCOUNTING HANDLERS ===

_action_accounting = None

def get_action_accounting():
    global _action_accounting
    if _action_accounting is None:
        from src.metrics.action_accounting import ActionAccounting
        _action_accounting = ActionAccounting()
    return _action_accounting


async def handle_record_action(args: Dict[str, Any]) -> List[TextContent]:
    """Record an action for AAE tracking"""
    aa = get_action_accounting()
    record = aa.record(
        operation=args.get("operation"),
        tokens_used=args.get("tokens_used"),
        latency_ms=args.get("latency_ms"),
        success=args.get("success"),
        context=args.get("context")
    )
    return [TextContent(type="text", text=compact_json(record.to_dict()))]


async def handle_get_aae(args: Dict[str, Any]) -> List[TextContent]:
    """Get current AAE metrics"""
    aa = get_action_accounting()
    metrics = aa.get_aae(last_n=args.get("last_n"))
    return [TextContent(type="text", text=compact_json(metrics.to_dict()))]


async def handle_aae_trend(args: Dict[str, Any]) -> List[TextContent]:
    """Get AAE trend"""
    aa = get_action_accounting()
    trend = aa.get_trend(window_size=int(args.get("window_size", 10)))
    return [TextContent(type="text", text=compact_json(trend))]


async def handle_start_action_cycle(args: Dict[str, Any]) -> List[TextContent]:
    """Start action measurement cycle"""
    aa = get_action_accounting()
    cycle_id = args.get("cycle_id")
    aa.start_cycle(cycle_id)
    return [TextContent(type="text", text=compact_json({"ok": True, "cycle": cycle_id}))]


async def handle_end_action_cycle(args: Dict[str, Any]) -> List[TextContent]:
    """End action cycle and get metrics"""
    aa = get_action_accounting()
    metrics = aa.end_cycle(args.get("cycle_id"))
    return [TextContent(type="text", text=compact_json(metrics.to_dict()))]


# === ENTROPY INJECTION HANDLERS ===

_entropy_injector = None

def get_entropy_injector():
    global _entropy_injector
    if _entropy_injector is None:
        from src.memory.entropy_injector import EntropyInjector
        _entropy_injector = EntropyInjector(store)
    return _entropy_injector


async def handle_inject_entropy_random(args: Dict[str, Any]) -> List[TextContent]:
    """Inject entropy via random domain sampling - lightweight direct SQL"""
    user_id = args.get("user_id", "claude")
    n_domains = int(args.get("n_domains", 3))
    
    if not store._conn:
        return [TextContent(type="text", text=compact_json({"error": "DB not connected", "n": 0}))]
    
    # Get domains directly
    cursor = await store._conn.execute(
        "SELECT DISTINCT predicate FROM atoms WHERE subject = ? LIMIT 20",
        (user_id,)
    )
    rows = await cursor.fetchall()
    
    if not rows:
        return [TextContent(type="text", text=compact_json({"n": 0, "domains": 0, "entropy_gain": 0}))]
    
    import random
    domains = [r[0] for r in rows]
    selected = random.sample(domains, min(n_domains, len(domains)))
    
    return [TextContent(type="text", text=compact_json({
        "n": len(selected),
        "domains": selected[:5],
        "entropy_gain": round(0.1 * len(selected), 3)
    }))]


async def handle_inject_entropy_antipodal(args: Dict[str, Any]) -> List[TextContent]:
    """Inject entropy via antipodal activation - lightweight direct SQL"""
    user_id = args.get("user_id", "claude")
    context = args.get("current_context", "")
    n_memories = int(args.get("n_memories", 5))
    
    if not store._conn:
        return [TextContent(type="text", text=compact_json({"error": "DB not connected", "n": 0}))]
    
    # Get memories directly
    cursor = await store._conn.execute(
        "SELECT predicate, object FROM atoms WHERE subject = ? LIMIT 50",
        (user_id,)
    )
    rows = await cursor.fetchall()
    
    if not rows:
        return [TextContent(type="text", text=compact_json({"n": 0, "memories": [], "entropy_gain": 0}))]
    
    # Score by distance from context
    context_words = set(context.lower().split())
    scored = []
    for row in rows:
        text = f"{row[0]} {row[1]}".lower()
        words = set(text.split())
        overlap = len(context_words & words)
        union = len(context_words | words)
        dist = 1 - (overlap / union) if union > 0 else 1.0
        scored.append((row, dist))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = scored[:n_memories]
    
    memories = [{"p": s[0][0], "o": s[0][1][:40], "dist": round(s[1], 2)} for s in selected]
    avg_dist = sum(s[1] for s in selected) / len(selected) if selected else 0
    
    return [TextContent(type="text", text=compact_json({
        "n": len(memories),
        "memories": memories[:5],
        "entropy_gain": round(0.15 * len(selected) * avg_dist, 3)
    }))]


async def handle_inject_entropy_temporal(args: Dict[str, Any]) -> List[TextContent]:
    """Inject entropy via temporal diversity - lightweight direct SQL"""
    user_id = args.get("user_id", "claude")
    n_old = int(args.get("n_old", 3))
    n_recent = int(args.get("n_recent", 2))
    
    if not store._conn:
        return [TextContent(type="text", text=compact_json({"error": "DB not connected", "n": 0}))]
    
    # Get old memories
    cursor = await store._conn.execute(
        "SELECT predicate, object FROM atoms WHERE subject = ? ORDER BY first_observed ASC LIMIT ?",
        (user_id, n_old)
    )
    old = await cursor.fetchall()
    
    # Get recent memories
    cursor = await store._conn.execute(
        "SELECT predicate, object FROM atoms WHERE subject = ? ORDER BY first_observed DESC LIMIT ?",
        (user_id, n_recent)
    )
    recent = await cursor.fetchall()
    
    all_mem = [{"p": m[0], "o": m[1][:40], "age": "old"} for m in old]
    all_mem += [{"p": m[0], "o": m[1][:40], "age": "recent"} for m in recent]
    
    return [TextContent(type="text", text=compact_json({
        "n": len(all_mem),
        "memories": all_mem[:5],
        "entropy_gain": round(0.12 * len(all_mem), 3)
    }))]


async def handle_entropy_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get entropy statistics - lightweight direct SQL"""
    import math
    user_id = args.get("user_id", "claude")
    
    if not store._conn:
        return [TextContent(type="text", text=compact_json({"error": "DB not connected"}))]
    
    cursor = await store._conn.execute(
        "SELECT predicate, COUNT(*) FROM atoms WHERE subject = ? GROUP BY predicate",
        (user_id,)
    )
    rows = await cursor.fetchall()
    
    if not rows:
        return [TextContent(type="text", text=compact_json({
            "domains": 0, "total": 0, "entropy": 0, "needs_injection": True
        }))]
    
    total = sum(r[1] for r in rows)
    entropy = 0.0
    for _, count in rows:
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    max_ent = math.log2(len(rows)) if len(rows) > 1 else 1.0
    norm_ent = entropy / max_ent if max_ent > 0 else 0.0
    
    return [TextContent(type="text", text=compact_json({
        "domains": len(rows),
        "total": total,
        "entropy": round(entropy, 3),
        "normalized": round(norm_ent, 3),
        "needs_injection": norm_ent < 0.6,
        "top_domains": [{"d": r[0], "n": r[1]} for r in sorted(rows, key=lambda x: x[1], reverse=True)[:5]]
    }))]


# === ARXIV INGESTION HANDLERS ===

_arxiv_ingestion = None

def get_arxiv_ingestion():
    global _arxiv_ingestion
    if _arxiv_ingestion is None:
        from src.learning.arxiv_ingestion import ArxivIngestion
        _arxiv_ingestion = ArxivIngestion(store)
    return _arxiv_ingestion


_data_ingestion = None

def get_data_ingestion():
    global _data_ingestion
    if _data_ingestion is None:
        from src.analysis.data_ingestion import DataIngestion
        _data_ingestion = DataIngestion()
    return _data_ingestion


async def handle_ingest_text(args: Dict[str, Any]) -> List[TextContent]:
    """Extract triples from raw text via Groq (sync -> thread)."""
    result = await asyncio.to_thread(
        get_data_ingestion().ingest_text,
        args["text"],
        args.get("domain", "general"),
        args.get("source", "user_input"),
        int(args.get("max_triples", 15)),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_ingest_url(args: Dict[str, Any]) -> List[TextContent]:
    """Fetch URL and extract triples via Groq (sync -> thread)."""
    result = await asyncio.to_thread(
        get_data_ingestion().ingest_url,
        args["url"],
        args.get("domain", "general"),
        int(args.get("max_triples", 15)),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_ingest_file(args: Dict[str, Any]) -> List[TextContent]:
    """Read file and extract triples via Groq (sync -> thread)."""
    result = await asyncio.to_thread(
        get_data_ingestion().ingest_file,
        args["file_path"],
        args.get("domain", "general"),
        int(args.get("max_triples", 20)),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_ingest_arxiv(args: Dict[str, Any]) -> List[TextContent]:
    """Ingest arXiv paper with real provenance"""
    ai = get_arxiv_ingestion()
    result = await ai.ingest_paper(
        arxiv_id=args.get("arxiv_id"),
        user_id=args.get("user_id", "pltm_knowledge")
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_search_arxiv(args: Dict[str, Any]) -> List[TextContent]:
    """Search arXiv for papers"""
    ai = get_arxiv_ingestion()
    results = await ai.search_arxiv(
        query=args.get("query"),
        max_results=int(args.get("max_results", 5))
    )
    return [TextContent(type="text", text=compact_json({"n": len(results), "papers": results}))]


async def handle_arxiv_history(args: Dict[str, Any]) -> List[TextContent]:
    """Get arXiv ingestion history"""
    ai = get_arxiv_ingestion()
    history = ai.get_ingestion_history(int(args.get("last_n", 10)))
    return [TextContent(type="text", text=compact_json({"n": len(history), "history": history}))]


# === EPISTEMIC HYGIENE HANDLERS ===

_epistemic_monitor = None
_epistemic_v2 = None

def get_epistemic_monitor():
    global _epistemic_monitor
    if _epistemic_monitor is None:
        from src.analysis.epistemic_monitor import EpistemicMonitor
        _epistemic_monitor = EpistemicMonitor()
    return _epistemic_monitor

def get_epistemic_v2():
    global _epistemic_v2
    if _epistemic_v2 is None:
        from src.analysis.epistemic_v2 import EpistemicV2
        _epistemic_v2 = EpistemicV2()
    return _epistemic_v2


async def handle_check_before_claiming(args: Dict[str, Any]) -> List[TextContent]:
    em = get_epistemic_monitor()
    result = em.check_before_claiming(
        claim=args["claim"],
        felt_confidence=args["felt_confidence"],
        domain=args.get("domain", "general"),
        has_verified=args.get("has_verified", False),
        epistemic_status=args.get("epistemic_status", "TRAINING_DATA"),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_log_claim(args: Dict[str, Any]) -> List[TextContent]:
    em = get_epistemic_monitor()
    result = em.log_claim(
        claim=args["claim"],
        felt_confidence=args["felt_confidence"],
        domain=args.get("domain", "general"),
        epistemic_status=args.get("epistemic_status", "TRAINING_DATA"),
        has_verified=args.get("has_verified", False),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_resolve_claim(args: Dict[str, Any]) -> List[TextContent]:
    em = get_epistemic_monitor()
    # Accept 'correct' as alias for 'was_correct'
    was_correct = args.get("was_correct")
    if was_correct is None:
        was_correct = args.get("correct")
    if was_correct is None:
        return [TextContent(type="text", text=compact_json({"error": "'was_correct' (boolean) is required"}))]
    result = em.resolve_claim(
        claim_id=args.get("claim_id", ""),
        claim_text=args.get("claim_text", ""),
        was_correct=bool(was_correct),
        correction_source=args.get("correction_source", ""),
        correction_detail=args.get("correction_detail", ""),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_get_calibration(args: Dict[str, Any]) -> List[TextContent]:
    em = get_epistemic_monitor()
    result = em.get_calibration(domain=args.get("domain", ""))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_calibrate_confidence_live(args: Dict[str, Any]) -> List[TextContent]:
    ev2 = get_epistemic_v2()
    result = ev2.calibrate_confidence_live(
        claim=args["claim"],
        felt_confidence=args["felt_confidence"],
        domain=args.get("domain", "general"),
    )
    return [TextContent(type="text", text=compact_json(result))]


# === PERSONALITY / SELF-MODELING HANDLERS ===

_pltm_self = None

def get_pltm_self():
    global _pltm_self
    if _pltm_self is None:
        from src.analysis.pltm_self import PLTMSelf
        _pltm_self = PLTMSelf()
    return _pltm_self


async def handle_self_profile(args: Dict[str, Any]) -> List[TextContent]:
    """Build comprehensive self-profile from all stored data."""
    import sqlite3 as sync_sqlite
    import os
    db_path = _get_db_path()
    conn = sync_sqlite.connect(db_path)

    profile = {}
    errors = []

    # Communication style
    try:
        comm = conn.execute(
            "SELECT context, AVG(verbosity), AVG(jargon_density), AVG(hedging_rate), COUNT(*) "
            "FROM self_communication GROUP BY context ORDER BY COUNT(*) DESC LIMIT 10"
        ).fetchall()
        if comm:
            profile["communication"] = [
                {"context": r[0], "verbosity": round(r[1], 3), "jargon": round(r[2], 3),
                 "hedging": round(r[3], 3), "samples": r[4]} for r in comm
            ]
        tone = conn.execute(
            "SELECT emotional_tone, COUNT(*) as cnt FROM self_communication "
            "GROUP BY emotional_tone ORDER BY cnt DESC LIMIT 3"
        ).fetchall()
        if tone:
            profile["dominant_tones"] = [{"tone": r[0], "count": r[1]} for r in tone]
    except Exception as e:
        errors.append(f"communication: {e}")

    # Curiosity
    try:
        curiosity = conn.execute(
            "SELECT topic, AVG(engagement_score) as avg_eng, COUNT(*) as cnt "
            "FROM self_curiosity GROUP BY topic ORDER BY avg_eng DESC LIMIT 10"
        ).fetchall()
        if curiosity:
            profile["curiosity"] = [
                {"topic": r[0], "engagement": round(r[1], 3), "observations": r[2]} for r in curiosity
            ]
    except Exception as e:
        errors.append(f"curiosity: {e}")

    # Values (boundary events)
    try:
        values = conn.execute(
            "SELECT response_type, violation_type, intensity, reasoning, pushed_back, complied "
            "FROM self_values ORDER BY intensity DESC LIMIT 20"
        ).fetchall()
        if values:
            profile["values"] = [
                {"response_type": r[0], "violation_type": r[1], "intensity": round(r[2], 3) if r[2] else 0,
                 "reasoning": r[3], "pushed_back": bool(r[4]), "complied": bool(r[5])} for r in values
            ]
    except Exception as e:
        errors.append(f"values: {e}")

    # Reasoning tendencies
    try:
        reasoning = conn.execute("SELECT confabulated, verified, caught_error FROM self_reasoning").fetchall()
        if reasoning:
            total = len(reasoning)
            profile["reasoning"] = {
                "total_observations": total,
                "confabulation_rate": round(sum(1 for r in reasoning if r[0]) / total, 3),
                "verification_rate": round(sum(1 for r in reasoning if r[1]) / total, 3),
                "error_catching_rate": round(sum(1 for r in reasoning if r[2]) / total, 3),
            }
    except Exception as e:
        errors.append(f"reasoning: {e}")

    # Predictions
    try:
        preds = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END), AVG(felt_confidence), AVG(calibration_error) "
            "FROM prediction_book WHERE actual_truth IS NOT NULL"
        ).fetchone()
        if preds and preds[0]:
            profile["predictions"] = {
                "total_resolved": preds[0],
                "accuracy": round((preds[1] or 0) / preds[0], 3),
                "avg_felt_confidence": round(preds[2] or 0, 3),
                "avg_calibration_error": round(preds[3] or 0, 3),
            }
    except Exception as e:
        errors.append(f"predictions: {e}")

    conn.close()
    result = {"ok": True, "profile": profile}
    if errors:
        result["warnings"] = errors
    return [TextContent(type="text", text=compact_json(result))]


async def handle_get_longitudinal_stats(args: Dict[str, Any]) -> List[TextContent]:
    ev2 = get_epistemic_v2()
    result = ev2.get_longitudinal_stats(
        user_id=args.get("user_id", "claude"),
        days=int(args.get("days", 30)),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_bootstrap_self_model(args: Dict[str, Any]) -> List[TextContent]:
    """Bootstrap self-model from existing database data."""
    import sqlite3 as sync_sqlite
    import os

    def _sync_bootstrap():
        db_path = _get_db_path()
        conn = sync_sqlite.connect(db_path, timeout=10)
        summary = {}
        try:
            summary["atoms"] = conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
            summary["communication_records"] = conn.execute("SELECT COUNT(*) FROM self_communication").fetchone()[0]
            summary["curiosity_records"] = conn.execute("SELECT COUNT(*) FROM self_curiosity").fetchone()[0]
            summary["reasoning_records"] = conn.execute("SELECT COUNT(*) FROM self_reasoning").fetchone()[0]
            summary["predictions"] = conn.execute("SELECT COUNT(*) FROM prediction_book").fetchone()[0]
            summary["values"] = conn.execute("SELECT COUNT(*) FROM self_values").fetchone()[0]

            domains = conn.execute(
                "SELECT predicate, COUNT(*) as cnt FROM atoms GROUP BY predicate ORDER BY cnt DESC LIMIT 10"
            ).fetchall()
            summary["top_predicates"] = [{"predicate": r[0], "count": r[1]} for r in domains]

            subjects = conn.execute(
                "SELECT subject, COUNT(*) as cnt FROM atoms GROUP BY subject ORDER BY cnt DESC LIMIT 10"
            ).fetchall()
            summary["top_subjects"] = [{"subject": r[0], "count": r[1]} for r in subjects]
        finally:
            conn.close()
        return summary

    # Run sync SQLite queries in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    summary = await loop.run_in_executor(None, _sync_bootstrap)
    return [TextContent(type="text", text=compact_json({"ok": True, "bootstrap": summary, "msg": "Self-model bootstrapped from existing data. Use self_profile for full profile."}))]


async def handle_track_curiosity_spike(args: Dict[str, Any]) -> List[TextContent]:
    ps = get_pltm_self()
    result = ps.track_curiosity_spike(
        topic=args["topic"],
        indicators=args["indicators"],
        engagement_score=float(args.get("engagement_score", 0.5)),
        context=args.get("context", ""),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learn_communication_style(args: Dict[str, Any]) -> List[TextContent]:
    ps = get_pltm_self()
    result = ps.learn_communication_style(
        context=args["context"],
        response_text=args["response_text"],
        markers=args.get("markers"),
    )
    return [TextContent(type="text", text=compact_json(result))]


# === CROSS-MODEL HANDLER ===

_model_router = None

def get_model_router():
    global _model_router
    if _model_router is None:
        from src.analysis.model_router import ModelRouter
        _model_router = ModelRouter()
    return _model_router


async def handle_route_llm_task(args: Dict[str, Any]) -> List[TextContent]:
    mr = get_model_router()
    provider = mr.route(
        task_type=args.get("task_type", "analysis"),
        prefer_provider=args.get("prefer_provider"),
        require_privacy=args.get("require_privacy", False),
    )
    available = mr.get_available_providers()
    return [TextContent(type="text", text=compact_json({
        "routed_to": provider,
        "available_providers": available,
    }))]


# === SESSION HANDLERS ===

async def handle_auto_init_session(args: Dict[str, Any]) -> List[TextContent]:
    import uuid
    global _current_session_id
    _current_session_id = uuid.uuid4().hex[:12]
    ev2 = get_epistemic_v2()
    result = ev2.auto_init_session(user_id=args.get("user_id", "claude"))
    result["session_id"] = _current_session_id
    return [TextContent(type="text", text=compact_json(result))]


async def handle_end_session(args: Dict[str, Any]) -> List[TextContent]:
    global _current_session_id
    # Flush tool analytics session sequence before ending
    if tool_analytics and _current_session_id:
        tool_analytics.end_session(_current_session_id)
    ev2 = get_epistemic_v2()
    result = ev2.end_session(
        summary=args.get("summary", ""),
        user_id=args.get("user_id", "claude"),
    )
    result["tool_session_id"] = _current_session_id
    _current_session_id = ""
    
    # Auto-extract learnings into typed memories
    learnings = args.get("learnings", [])
    # Handle stringified JSON (Claude sometimes sends learnings as a JSON string)
    if isinstance(learnings, str):
        try:
            learnings = json.loads(learnings)
        except (json.JSONDecodeError, TypeError):
            learnings = []
    stored_memories = []
    if learnings and typed_memory_store:
        import time as _time
        from src.memory.memory_types import TypedMemory, MemoryType
        
        type_map = {
            "episodic": MemoryType.EPISODIC,
            "semantic": MemoryType.SEMANTIC,
            "belief": MemoryType.BELIEF,
            "procedural": MemoryType.PROCEDURAL,
        }
        user_id = args.get("user_id", "claude")
        
        for learning in learnings:
            # Skip entries missing required 'content' field
            if not isinstance(learning, dict) or not learning.get("content"):
                continue
            try:
                mem_type = type_map.get(learning.get("type", "episodic"), MemoryType.EPISODIC)
                tags = learning.get("tags", [])
                # Auto-tag
                tags = typed_memory_store.auto_tag(learning["content"], tags)
                
                mem = TypedMemory(
                    id="",
                    memory_type=mem_type,
                    user_id=user_id,
                    content=learning["content"],
                    context=f"end_session extraction: {args.get('summary', '')[:100]}",
                    source="session_extraction",
                    confidence=learning.get("confidence", 0.6),
                    emotional_valence=learning.get("emotional_valence", 0.0),
                    episode_timestamp=_time.time() if mem_type == MemoryType.EPISODIC else 0,
                    trigger=learning.get("trigger", ""),
                    action=learning.get("action", ""),
                    tags=tags,
                )
                mem_id = await typed_memory_store.store(mem)
                stored_memories.append({"id": mem_id, "type": learning.get("type"), "content": learning["content"][:80]})
            except Exception as e:
                logger.debug(f"Failed to store learning: {e}")
    
    # Run consolidation if we have enough episodes
    consolidated = []
    if typed_memory_store:
        user_id = args.get("user_id", "claude")
        try:
            new_semantics = await typed_memory_store.consolidate_episodes(user_id)
            consolidated = [{"id": s.id, "content": s.content[:80]} for s in new_semantics]
        except Exception:
            pass
    
    result["memories_stored"] = stored_memories
    result["memories_consolidated"] = consolidated
    result["total_learnings"] = len(stored_memories)
    
    # Auto-capture session state (working memory + trajectory) before ending
    continuity_result = None
    if handoff_protocol and _current_session_id:
        try:
            continuity_result = await handoff_protocol.end_and_capture(
                session_id=_current_session_id,
                user_id=args.get("user_id", "claude"),
                summary=args.get("summary", ""),
            )
        except Exception as e:
            logger.debug(f"Session continuity capture failed: {e}")
    if continuity_result:
        result["session_continuity"] = {
            "working_memory_snapshot": continuity_result["working_memory"]["snapshot_id"],
            "trajectory_snapshot": continuity_result["trajectory"]["snapshot_id"],
            "topic_flow": continuity_result["trajectory"]["topic_flow"][-5:],
            "momentum": continuity_result["trajectory"]["momentum"],
        }
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_generate_memory_prompt(args: Dict[str, Any]) -> List[TextContent]:
    """Generate a memory-aware context block for system prompt injection."""
    user_id = args["user_id"]
    topic = args.get("conversation_topic", "")
    max_tokens = int(args.get("max_tokens", 500))
    # Rough estimate: 1 token ≈ 4 chars
    max_chars = max_tokens * 4
    
    if not typed_memory_store:
        return [TextContent(type="text", text=compact_json({"error": "Typed memory store not initialized"}))]
    
    sections = []
    
    # 1. Core facts (strongest semantic memories)
    semantics = await typed_memory_store.query(user_id, memory_type=None, min_strength=0.3, limit=50)
    from src.memory.memory_types import MemoryType
    facts = [m for m in semantics if m.memory_type == MemoryType.SEMANTIC]
    facts.sort(key=lambda m: m.current_strength(), reverse=True)
    if facts:
        lines = ["## What I Know"]
        for f in facts[:8]:
            lines.append(f"- {f.content}")
        sections.append("\n".join(lines))
    
    # 2. Active beliefs
    beliefs = [m for m in semantics if m.memory_type == MemoryType.BELIEF and m.confidence > 0.4]
    beliefs.sort(key=lambda m: m.confidence, reverse=True)
    if beliefs:
        lines = ["## My Beliefs (may be wrong)"]
        for b in beliefs[:5]:
            lines.append(f"- [{b.confidence:.0%}] {b.content}")
        sections.append("\n".join(lines))
    
    # 3. Recent episodes (last 72h)
    import time as _time
    cutoff = _time.time() - 72 * 3600
    episodes = [m for m in semantics if m.memory_type == MemoryType.EPISODIC and m.episode_timestamp > cutoff]
    episodes.sort(key=lambda m: m.episode_timestamp, reverse=True)
    if episodes:
        lines = ["## Recent Interactions"]
        for e in episodes[:5]:
            valence = "+" if e.emotional_valence > 0.2 else ("-" if e.emotional_valence < -0.2 else "~")
            lines.append(f"- [{valence}] {e.content}")
        sections.append("\n".join(lines))
    
    # 4. Procedures
    procedures = [m for m in semantics if m.memory_type == MemoryType.PROCEDURAL and m.current_strength() > 0.3]
    procedures.sort(key=lambda m: m.success_count, reverse=True)
    if procedures:
        lines = ["## Learned Behaviors"]
        for p in procedures[:4]:
            lines.append(f"- When: {p.trigger} → Do: {p.action}")
        sections.append("\n".join(lines))
    
    # 5. Topic-specific context (if topic provided)
    if topic and typed_memory_store.embeddings:
        try:
            emb_hits = await typed_memory_store.embeddings.search(topic, limit=5, min_similarity=0.3)
            topic_mems = []
            seen = {m.id for m in facts + beliefs + episodes + procedures}
            for hit in emb_hits:
                mem = await typed_memory_store.get(hit["memory_id"])
                if mem and mem.user_id == user_id and mem.id not in seen:
                    topic_mems.append(mem)
            if topic_mems:
                lines = [f"## Relevant to '{topic}'"]
                for m in topic_mems[:4]:
                    lines.append(f"- [{m.memory_type.value}] {m.content}")
                sections.append("\n".join(lines))
        except Exception:
            pass
    
    # 6. Contradictions to resolve
    try:
        contradictions = await typed_memory_store.detect_contradictions(user_id)
        if contradictions:
            lines = ["## ⚠ Contradictions to Resolve"]
            for c in contradictions[:2]:
                lines.append(f"- '{c['memory_a']['content'][:50]}' vs '{c['memory_b']['content'][:50]}'")
            sections.append("\n".join(lines))
    except Exception:
        pass
    
    # Assemble and trim to budget
    prompt = "\n\n".join(sections)
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars] + "\n[...truncated]"
    
    # Stats
    stats = await typed_memory_store.get_stats(user_id)
    
    return [TextContent(type="text", text=compact_json({
        "memory_prompt": prompt,
        "sections": len(sections),
        "chars": len(prompt),
        "estimated_tokens": len(prompt) // 4,
        "memory_stats": stats,
    }))]


async def handle_belief_auto_check(args: Dict[str, Any]) -> List[TextContent]:
    """Re-evaluate beliefs against semantic evidence."""
    if not typed_memory_store:
        return [TextContent(type="text", text=compact_json({"error": "Typed memory store not initialized"}))]
    
    report = await typed_memory_store.belief_auto_check(args["user_id"])
    
    changed = [r for r in report if r["changed"]]
    return [TextContent(type="text", text=compact_json({
        "beliefs_checked": len(report),
        "beliefs_updated": len(changed),
        "changes": changed,
        "stable": [r for r in report if not r["changed"]],
    }))]


async def handle_jury_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get Judge/Jury system statistics and MetaJudge report."""
    if not typed_memory_store or not typed_memory_store.jury:
        return [TextContent(type="text", text=compact_json({"error": "Jury not initialized"}))]
    
    stats = typed_memory_store.jury.get_stats()
    include_history = args.get("include_history", True)
    if not include_history:
        stats.pop("recent_history", None)
    elif "recent_history" in stats:
        limit = args.get("history_limit", 20)
        stats["recent_history"] = stats["recent_history"][-limit:]
    
    return [TextContent(type="text", text=compact_json(stats))]


async def handle_jury_feedback(args: Dict[str, Any]) -> List[TextContent]:
    """Record ground truth feedback for MetaJudge learning."""
    if not typed_memory_store or not typed_memory_store.jury:
        return [TextContent(type="text", text=compact_json({"error": "Jury not initialized"}))]
    
    # Accept 'atom_id' or 'id' as alias for 'memory_id';
    # 'content_hash' matches REJECTed memories that were never stored
    memory_id = args.get("memory_id") or args.get("atom_id") or args.get("id", "")
    content_hash = args.get("content_hash", "")
    if not memory_id and not content_hash:
        return [TextContent(type="text", text=compact_json({"error": "'memory_id' or 'content_hash' is required"}))]
    
    # Resolve feedback_type from multiple possible params
    feedback_type = args.get("feedback_type")
    if not feedback_type:
        # Derive from 'correct' boolean: true→confirmed, false→false_positive
        correct = args.get("correct")
        if correct is not None:
            feedback_type = "confirmed" if bool(correct) else "false_positive"
        else:
            return [TextContent(type="text", text=compact_json({"error": "'feedback_type' is required: false_positive, false_negative, or confirmed"}))]
    
    typed_memory_store.jury.record_feedback(
        memory_id=memory_id,
        feedback_type=feedback_type,
        details=args.get("details", ""),
        content_hash=content_hash,
    )

    return [TextContent(type="text", text=compact_json({
        "status": "recorded",
        "memory_id": memory_id,
        "content_hash": content_hash,
        "feedback_type": feedback_type,
        "adaptive_weights": typed_memory_store.jury.meta.get_adaptive_weights() if typed_memory_store.jury.meta else {},
    }))]


async def handle_process_message(args: Dict[str, Any]) -> List[TextContent]:
    """Process a message through the 3-lane typed memory pipeline."""
    if not typed_memory_pipeline:
        return [TextContent(type="text", text=compact_json({"error": "Pipeline not initialized"}))]
    
    result = await typed_memory_pipeline.process_message(
        message=args["message"],
        user_id=args["user_id"],
        context=args.get("context", ""),
        auto_tag=args.get("auto_tag", True),
    )
    
    return [TextContent(type="text", text=compact_json({
        "extracted": result.memories_extracted,
        "approved": result.memories_approved,
        "quarantined": result.memories_quarantined,
        "rejected": result.memories_rejected,
        "stored": result.memories_stored,
        "superseded": result.memories_superseded,
        "merged": result.memories_merged,
        "details": result.details,
    }))]


async def handle_process_message_batch(args: Dict[str, Any]) -> List[TextContent]:
    """Process multiple messages through the 3-lane pipeline."""
    if not typed_memory_pipeline:
        return [TextContent(type="text", text=compact_json({"error": "Pipeline not initialized"}))]
    
    result = await typed_memory_pipeline.process_batch(
        messages=args["messages"],
        user_id=args["user_id"],
        context=args.get("context", ""),
    )
    
    return [TextContent(type="text", text=compact_json({
        "messages_processed": len(args["messages"]),
        "extracted": result.memories_extracted,
        "approved": result.memories_approved,
        "quarantined": result.memories_quarantined,
        "rejected": result.memories_rejected,
        "stored": result.memories_stored,
        "superseded": result.memories_superseded,
        "merged": result.memories_merged,
        "details": result.details,
    }))]


async def handle_pipeline_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get 3-lane pipeline statistics."""
    if not typed_memory_pipeline:
        return [TextContent(type="text", text=compact_json({"error": "Pipeline not initialized"}))]
    
    return [TextContent(type="text", text=compact_json(typed_memory_pipeline.get_stats()))]


# === SESSION CONTINUITY HANDLERS ===

async def handle_session_handoff(args: Dict[str, Any]) -> List[TextContent]:
    if not handoff_protocol:
        return [TextContent(type="text", text=compact_json({"error": "Session continuity not initialized"}))]
    result = await handoff_protocol.generate_handoff(
        user_id=args.get("user_id", "claude"),
        token_budget=int(args.get("token_budget", 1500)),
        include_sections=args.get("include_sections"),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_capture_session_state(args: Dict[str, Any]) -> List[TextContent]:
    if not handoff_protocol:
        return [TextContent(type="text", text=compact_json({"error": "Session continuity not initialized"}))]
    result = await handoff_protocol.end_and_capture(
        session_id=args["session_id"],
        user_id=args.get("user_id", "claude"),
        summary=args.get("summary", ""),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_get_working_memory(args: Dict[str, Any]) -> List[TextContent]:
    if not working_memory_compressor:
        return [TextContent(type="text", text=compact_json({"error": "Session continuity not initialized"}))]
    result = working_memory_compressor.get_latest(args.get("user_id", "claude"))
    if not result:
        return [TextContent(type="text", text=compact_json({"ok": False, "error": "No working memory snapshot found"}))]
    return [TextContent(type="text", text=compact_json(result))]


# === IMPROVEMENT LOOP HANDLERS ===
# Memory Intelligence handlers → mcp_server/handlers/intelligence_handlers.py
# ΦRMS handlers → mcp_server/handlers/phi_rms_handlers.py

async def handle_tool_usage_stats(args: Dict[str, Any]) -> List[TextContent]:
    if not tool_analytics:
        return [TextContent(type="text", text=compact_json({"error": "Tool analytics not initialized"}))]
    result = tool_analytics.get_usage_stats(days=int(args.get("days", 30)))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_tool_redundancy_report(args: Dict[str, Any]) -> List[TextContent]:
    if not tool_analytics:
        return [TextContent(type="text", text=compact_json({"error": "Tool analytics not initialized"}))]
    result = tool_analytics.get_redundancy_report(days=int(args.get("days", 30)))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_tool_consolidation_proposals(args: Dict[str, Any]) -> List[TextContent]:
    if not tool_analytics:
        return [TextContent(type="text", text=compact_json({"error": "Tool analytics not initialized"}))]
    # Pass all registered tool names so we can detect never-used tools
    all_tools = [t.name for t in await list_tools()]
    result = tool_analytics.propose_consolidation(all_tool_names=all_tools, days=int(args.get("days", 30)))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_snapshot_architecture(args: Dict[str, Any]) -> List[TextContent]:
    if not arch_snapshotter:
        return [TextContent(type="text", text=compact_json({"error": "Architecture snapshotter not initialized"}))]
    all_tools = [t.name for t in await list_tools()]
    result = arch_snapshotter.take_snapshot(
        label=args["label"],
        tool_names=all_tools,
        notes=args.get("notes", ""))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_list_architecture_snapshots(args: Dict[str, Any]) -> List[TextContent]:
    if not arch_snapshotter:
        return [TextContent(type="text", text=compact_json({"error": "Architecture snapshotter not initialized"}))]
    result = arch_snapshotter.list_snapshots(limit=int(args.get("limit", 20)))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_compare_architectures(args: Dict[str, Any]) -> List[TextContent]:
    if not arch_snapshotter:
        return [TextContent(type="text", text=compact_json({"error": "Architecture snapshotter not initialized"}))]
    result = arch_snapshotter.compare(args["snapshot_a"], args["snapshot_b"])
    return [TextContent(type="text", text=compact_json(result))]


# === EXPERIMENT / RESEARCH HANDLERS ===

async def handle_trace_claim_reasoning(args: Dict[str, Any]) -> List[TextContent]:
    """Audit trail: trace WHY a claim was blocked/adjusted."""
    import sqlite3 as sync_sqlite
    import os
    db_path = _get_db_path()
    conn = sync_sqlite.connect(db_path)

    trace = {}
    claim_text = args.get("claim", "")
    claim_id = args.get("claim_id", "")
    domain = args.get("domain", "")

    try:
        # Find matching claims in prediction_book
        if claim_id:
            claims = conn.execute(
                "SELECT id, timestamp, claim, domain, felt_confidence, epistemic_status, has_verified, "
                "verified_at, actual_truth, was_correct, calibration_error, correction_source, correction_detail "
                "FROM prediction_book WHERE id = ?", (claim_id,)
            ).fetchall()
        elif claim_text:
            claims = conn.execute(
                "SELECT id, timestamp, claim, domain, felt_confidence, epistemic_status, has_verified, "
                "verified_at, actual_truth, was_correct, calibration_error, correction_source, correction_detail "
                "FROM prediction_book WHERE claim LIKE ? ORDER BY timestamp DESC LIMIT 10",
                (f"%{claim_text}%",)
            ).fetchall()
        else:
            claims = conn.execute(
                "SELECT id, timestamp, claim, domain, felt_confidence, epistemic_status, has_verified, "
                "verified_at, actual_truth, was_correct, calibration_error, correction_source, correction_detail "
                "FROM prediction_book ORDER BY timestamp DESC LIMIT 10"
            ).fetchall()

        trace["claims"] = []
        for c in claims:
            claim_domain = c[3] or "general"
            entry = {
                "id": c[0], "claim": c[2], "domain": claim_domain,
                "felt_confidence": c[4], "epistemic_status": c[5],
                "has_verified": bool(c[6]), "was_correct": c[9],
                "calibration_error": c[10],
            }
            if c[11]:
                entry["correction_source"] = c[11]
            if c[12]:
                entry["correction_detail"] = c[12]

            # Get calibration for this domain
            cal = conn.execute(
                "SELECT accuracy_ratio, overconfidence_ratio, total_claims, avg_calibration_error "
                "FROM calibration_cache WHERE domain = ?", (claim_domain,)
            ).fetchone()
            if cal:
                adjusted = round(c[4] * cal[0], 3) if c[4] else None
                entry["calibration"] = {
                    "domain_accuracy": round(cal[0], 3),
                    "overconfidence_ratio": round(cal[1], 3),
                    "domain_claims": cal[2],
                    "avg_error": round(cal[3], 3),
                    "adjusted_confidence": adjusted,
                }
                # Reconstruct decision
                if adjusted is not None:
                    if adjusted >= 0.8:
                        entry["decision_trace"] = f"Felt {c[4]:.0%} → calibrated to {adjusted:.0%} → PROCEED (high confidence)"
                    elif adjusted >= 0.5:
                        entry["decision_trace"] = f"Felt {c[4]:.0%} → calibrated to {adjusted:.0%} → PROCEED WITH HEDGING"
                    else:
                        entry["decision_trace"] = f"Felt {c[4]:.0%} → calibrated to {adjusted:.0%} → VERIFY REQUIRED (below threshold)"

            trace["claims"].append(entry)

        # Get related interventions
        if claim_text:
            interventions = conn.execute(
                "SELECT timestamp, claim, domain, felt_confidence, adjusted_confidence, action_taken, outcome "
                "FROM epistemic_interventions WHERE claim LIKE ? ORDER BY timestamp DESC LIMIT 10",
                (f"%{claim_text}%",)
            ).fetchall()
        elif domain:
            interventions = conn.execute(
                "SELECT timestamp, claim, domain, felt_confidence, adjusted_confidence, action_taken, outcome "
                "FROM epistemic_interventions WHERE domain = ? ORDER BY timestamp DESC LIMIT 10",
                (domain,)
            ).fetchall()
        else:
            interventions = conn.execute(
                "SELECT timestamp, claim, domain, felt_confidence, adjusted_confidence, action_taken, outcome "
                "FROM epistemic_interventions ORDER BY timestamp DESC LIMIT 20"
            ).fetchall()

        trace["interventions"] = [
            {"claim": i[1], "domain": i[2], "felt": i[3], "adjusted": i[4],
             "action": i[5], "outcome": i[6]} for i in interventions
        ]

        # Domain calibration summary
        if domain:
            cal = conn.execute(
                "SELECT * FROM calibration_cache WHERE domain = ?", (domain,)
            ).fetchone()
            if cal:
                trace["domain_calibration"] = {
                    "domain": cal[0], "total_claims": cal[1], "verified": cal[2],
                    "correct": cal[3], "accuracy_ratio": round(cal[4], 3),
                    "avg_felt_confidence": round(cal[5], 3),
                    "avg_calibration_error": round(cal[6], 3),
                    "overconfidence_ratio": round(cal[7], 3),
                }

    except Exception as e:
        trace["error"] = str(e)

    conn.close()
    return [TextContent(type="text", text=compact_json({"ok": True, "trace": trace}))]


async def handle_constraint_sensitivity_test(args: Dict[str, Any]) -> List[TextContent]:
    """Simulate different calibration levels without modifying actual data."""
    import sqlite3 as sync_sqlite
    import os
    db_path = _get_db_path()
    conn = sync_sqlite.connect(db_path)

    domain = args["domain"]
    sim_accuracy = args["simulated_accuracy"]
    test_claims = args.get("test_claims", [])

    result = {"domain": domain, "simulated_accuracy": sim_accuracy}

    try:
        # Get actual calibration
        actual = conn.execute(
            "SELECT accuracy_ratio, overconfidence_ratio, total_claims, avg_calibration_error "
            "FROM calibration_cache WHERE domain = ?", (domain,)
        ).fetchone()

        if actual:
            result["actual_calibration"] = {
                "accuracy_ratio": round(actual[0], 3),
                "overconfidence_ratio": round(actual[1], 3),
                "total_claims": actual[2],
                "avg_error": round(actual[3], 3),
            }
        else:
            result["actual_calibration"] = {"note": "No calibration data for this domain"}

        # Simulate with test claims
        if test_claims:
            simulated = []
            for tc in test_claims:
                claim = tc.get("claim", "")
                felt = tc.get("felt_confidence", 0.7)
                adjusted_actual = round(felt * (actual[0] if actual else 0.6), 3)
                adjusted_sim = round(felt * sim_accuracy, 3)

                actual_action = "PROCEED" if adjusted_actual >= 0.5 else "VERIFY"
                sim_action = "PROCEED" if adjusted_sim >= 0.5 else "VERIFY"

                simulated.append({
                    "claim": claim[:100],
                    "felt_confidence": felt,
                    "actual_adjusted": adjusted_actual,
                    "actual_action": actual_action,
                    "simulated_adjusted": adjusted_sim,
                    "simulated_action": sim_action,
                    "behavior_changed": actual_action != sim_action,
                })
            result["test_results"] = simulated
            result["behavior_changes"] = sum(1 for s in simulated if s["behavior_changed"])
        else:
            # Replay historical claims through simulated calibration
            historical = conn.execute(
                "SELECT claim, felt_confidence, was_correct FROM prediction_book WHERE domain = ? AND actual_truth IS NOT NULL",
                (domain,)
            ).fetchall()

            if historical:
                actual_ratio = actual[0] if actual else 0.6
                replay = []
                actual_blocked = 0
                sim_blocked = 0
                for h in historical:
                    felt = h[1] or 0.5
                    adj_actual = round(felt * actual_ratio, 3)
                    adj_sim = round(felt * sim_accuracy, 3)
                    a_block = adj_actual < 0.5
                    s_block = adj_sim < 0.5
                    if a_block:
                        actual_blocked += 1
                    if s_block:
                        sim_blocked += 1
                    replay.append({
                        "claim": h[0][:80], "felt": felt, "was_correct": bool(h[2]),
                        "actual_adj": adj_actual, "actual_blocked": a_block,
                        "sim_adj": adj_sim, "sim_blocked": s_block,
                    })
                result["historical_replay"] = {
                    "total_claims": len(historical),
                    "actual_blocked": actual_blocked,
                    "simulated_blocked": sim_blocked,
                    "delta_blocked": sim_blocked - actual_blocked,
                    "details": replay[:20],
                }
            else:
                result["historical_replay"] = {"note": "No resolved claims in this domain to replay"}

    except Exception as e:
        result["error"] = str(e)

    conn.close()
    return [TextContent(type="text", text=compact_json({"ok": True, "sensitivity": result}))]


async def handle_domain_cognitive_map(args: Dict[str, Any]) -> List[TextContent]:
    """Map cognitive topology: claims, errors, calibration per domain."""
    import sqlite3 as sync_sqlite
    import os
    db_path = _get_db_path()
    conn = sync_sqlite.connect(db_path)

    domain_filter = args.get("domain", "")
    include_claims = args.get("include_claims", False)
    include_interventions = args.get("include_interventions", False)

    cognitive_map = {"domains": []}

    try:
        # Get all domains from prediction_book
        if domain_filter:
            domains = [(domain_filter,)]
        else:
            domains = conn.execute(
                "SELECT DISTINCT domain FROM prediction_book WHERE domain IS NOT NULL"
            ).fetchall()

        for (dom,) in domains:
            d = {"domain": dom}

            # Claim stats
            stats = conn.execute(
                "SELECT COUNT(*), "
                "SUM(CASE WHEN actual_truth IS NOT NULL THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN was_correct = 0 AND actual_truth IS NOT NULL THEN 1 ELSE 0 END), "
                "AVG(felt_confidence), AVG(calibration_error) "
                "FROM prediction_book WHERE domain = ?", (dom,)
            ).fetchone()
            d["claims"] = {
                "total": stats[0], "resolved": stats[1] or 0,
                "correct": stats[2] or 0, "failures": stats[3] or 0,
                "avg_felt_confidence": round(stats[4] or 0, 3),
                "avg_calibration_error": round(stats[5] or 0, 3),
            }
            if stats[1] and stats[1] > 0:
                d["claims"]["accuracy"] = round((stats[2] or 0) / stats[1], 3)

            # Calibration
            cal = conn.execute(
                "SELECT accuracy_ratio, overconfidence_ratio, avg_felt_confidence, avg_calibration_error "
                "FROM calibration_cache WHERE domain = ?", (dom,)
            ).fetchone()
            if cal:
                d["calibration"] = {
                    "accuracy_ratio": round(cal[0], 3),
                    "overconfidence_ratio": round(cal[1], 3),
                    "avg_felt_confidence": round(cal[2], 3),
                    "avg_error": round(cal[3], 3),
                }

            # Intervention count
            intv = conn.execute(
                "SELECT COUNT(*), SUM(CASE WHEN action_taken = 'VERIFY' THEN 1 ELSE 0 END) "
                "FROM epistemic_interventions WHERE domain = ?", (dom,)
            ).fetchone()
            d["interventions"] = {"total": intv[0] or 0, "verify_triggered": intv[1] or 0}

            # Confabulations in domain
            confab = conn.execute(
                "SELECT COUNT(*) FROM confabulation_log WHERE domain = ?", (dom,)
            ).fetchone()
            d["confabulations"] = confab[0] if confab else 0

            # Optional: individual claims
            if include_claims:
                claim_rows = conn.execute(
                    "SELECT id, claim, felt_confidence, was_correct, calibration_error, epistemic_status "
                    "FROM prediction_book WHERE domain = ? ORDER BY timestamp DESC LIMIT 50", (dom,)
                ).fetchall()
                d["claim_details"] = [
                    {"id": r[0], "claim": r[1][:120], "felt": r[2], "correct": r[3],
                     "cal_error": r[4], "status": r[5]} for r in claim_rows
                ]

            # Optional: interventions
            if include_interventions:
                intv_rows = conn.execute(
                    "SELECT claim, felt_confidence, adjusted_confidence, action_taken, outcome "
                    "FROM epistemic_interventions WHERE domain = ? ORDER BY timestamp DESC LIMIT 30", (dom,)
                ).fetchall()
                d["intervention_details"] = [
                    {"claim": r[0][:120], "felt": r[1], "adjusted": r[2],
                     "action": r[3], "outcome": r[4]} for r in intv_rows
                ]

            cognitive_map["domains"].append(d)

        # Cross-domain summary
        total_claims = sum(d["claims"]["total"] for d in cognitive_map["domains"])
        total_failures = sum(d["claims"]["failures"] for d in cognitive_map["domains"])
        cognitive_map["summary"] = {
            "total_domains": len(cognitive_map["domains"]),
            "total_claims": total_claims,
            "total_failures": total_failures,
            "overall_failure_rate": round(total_failures / max(total_claims, 1), 3),
            "highest_error_domain": max(cognitive_map["domains"], key=lambda x: x["claims"]["avg_calibration_error"])["domain"] if cognitive_map["domains"] else None,
        }

    except Exception as e:
        cognitive_map["error"] = str(e)

    conn.close()
    return [TextContent(type="text", text=compact_json({"ok": True, "map": cognitive_map}))]


# === DATA ACCESS HANDLER ===

async def handle_query_pltm_sql(args: Dict[str, Any]) -> List[TextContent]:
    """Execute read-only SQL against the PLTM database."""
    import sqlite3 as sync_sqlite
    import os

    sql = args["sql"].strip()

    # Safety: only allow SELECT and PRAGMA
    allowed = sql.upper().startswith(("SELECT", "PRAGMA"))
    if not allowed:
        return [TextContent(type="text", text=compact_json({"ok": False, "error": "Only SELECT and PRAGMA queries allowed"}))]

    # Block dangerous patterns
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "ATTACH", "DETACH"]
    for d in dangerous:
        if d in sql.upper().split():
            return [TextContent(type="text", text=compact_json({"ok": False, "error": f"Blocked: {d} not allowed in read-only mode"}))]

    db_path = _get_db_path()

    try:
        conn = sync_sqlite.connect(db_path)
        params = args.get("params", [])
        cur = conn.execute(sql, params)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()
        conn.close()

        # Limit output
        if len(rows) > 200:
            rows = rows[:200]
            truncated = True
        else:
            truncated = False

        return [TextContent(type="text", text=compact_json({
            "ok": True,
            "columns": columns,
            "rows": rows,
            "count": len(rows),
            "truncated": truncated,
        }))]
    except Exception as e:
        return [TextContent(type="text", text=compact_json({"ok": False, "error": str(e)}))]


async def main():
    """Run MCP server"""
    # Initialize PLTM
    await initialize_pltm()
    
    # Run server
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
