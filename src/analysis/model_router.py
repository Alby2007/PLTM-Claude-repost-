"""
Hybrid Model Router

Routes LLM tasks to the cheapest appropriate model:
- Layer 1: Claude (strategic reasoning, 1% of calls)
- Layer 2: GPT-4o / DeepSeek (routine operations, 9% of calls)
- Layer 3: Groq / Together.ai (bulk work, free tier, 40% of calls)
- Layer 4: Ollama (local, free, unlimited, 50% of calls)

All providers use OpenAI-compatible API format where possible.
Tracks costs, latency, and success rates per provider.
"""

import json
import os
import sqlite3
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class ModelRouter:
    """Route LLM tasks to appropriate model based on complexity, cost, privacy."""
    
    # Provider configurations
    PROVIDERS = {
        "ollama": {
            "name": "Ollama (Local)",
            "tier": 4,
            "cost_per_1k_tokens": 0.0,
            "base_url": "http://localhost:11434",
            "api_format": "ollama",
            "default_model": "llama3.1:8b",
            "models": ["llama3.1:70b", "llama3.1:8b", "qwen2.5:72b", "deepseek-coder:33b", "mistral:7b"],
            "env_key": None,
            "max_tokens": 4096,
        },
        "groq": {
            "name": "Groq (Free Tier)",
            "tier": 3,
            "cost_per_1k_tokens": 0.00027,
            "base_url": "https://api.groq.com/openai/v1",
            "api_format": "openai",
            "default_model": "llama-3.3-70b-versatile",
            "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            "env_key": "GROQ_API_KEY",
            "max_tokens": 4096,
        },
        "deepseek": {
            "name": "DeepSeek (Cheap)",
            "tier": 2,
            "cost_per_1k_tokens": 0.00027,
            "base_url": "https://api.deepseek.com/v1",
            "api_format": "openai",
            "default_model": "deepseek-chat",
            "models": ["deepseek-chat", "deepseek-coder"],
            "env_key": "DEEPSEEK_API_KEY",
            "max_tokens": 4096,
        },
        "together": {
            "name": "Together.ai",
            "tier": 3,
            "cost_per_1k_tokens": 0.0008,
            "base_url": "https://api.together.xyz/v1",
            "api_format": "openai",
            "default_model": "meta-llama/Llama-3.1-70B-Instruct-Turbo",
            "models": ["meta-llama/Llama-3.1-405B-Instruct-Turbo", "meta-llama/Llama-3.1-70B-Instruct-Turbo"],
            "env_key": "TOGETHER_API_KEY",
            "max_tokens": 4096,
        },
        "openrouter": {
            "name": "OpenRouter (Aggregator)",
            "tier": 2,
            "cost_per_1k_tokens": 0.001,
            "base_url": "https://openrouter.ai/api/v1",
            "api_format": "openai",
            "default_model": "meta-llama/llama-3.1-70b-instruct",
            "models": ["meta-llama/llama-3.1-70b-instruct", "anthropic/claude-3.5-sonnet", "openai/gpt-4o"],
            "env_key": "OPENROUTER_API_KEY",
            "max_tokens": 4096,
        },
        "openai": {
            "name": "OpenAI GPT-4o",
            "tier": 2,
            "cost_per_1k_tokens": 0.005,
            "base_url": "https://api.openai.com/v1",
            "api_format": "openai",
            "default_model": "gpt-4o",
            "models": ["gpt-4o", "gpt-4o-mini"],
            "env_key": "OPENAI_API_KEY",
            "max_tokens": 4096,
        },
    }
    
    # Task complexity routing rules
    ROUTING_RULES = {
        "monitoring": {"preferred_tier": 4, "fallback_tier": 3},
        "data_collection": {"preferred_tier": 3, "fallback_tier": 4},
        "classification": {"preferred_tier": 4, "fallback_tier": 3},
        "extraction": {"preferred_tier": 3, "fallback_tier": 2},
        "analysis": {"preferred_tier": 2, "fallback_tier": 3},
        "synthesis": {"preferred_tier": 1, "fallback_tier": 2},
        "strategic": {"preferred_tier": 1, "fallback_tier": 2},
        "coding": {"preferred_tier": 2, "fallback_tier": 3},
        "translation": {"preferred_tier": 3, "fallback_tier": 4},
        "summarization": {"preferred_tier": 3, "fallback_tier": 4},
    }
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            from src.db_config import get_db_path
            db_path = Path(get_db_path())
        self.db_path = db_path
        self.timeout = 120
        self._ensure_tables()
    
    def _ensure_tables(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                task_type TEXT,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                latency_ms INTEGER DEFAULT 0,
                success INTEGER DEFAULT 1,
                error TEXT,
                timestamp REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_ts ON llm_usage(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_provider ON llm_usage(provider)")
        conn.commit()
        conn.close()
    
    def get_available_providers(self) -> List[Dict]:
        """Check which providers are available (have API keys or are local)"""
        available = []
        for pid, pconfig in self.PROVIDERS.items():
            status = "unavailable"
            
            if pid == "ollama":
                # Check if Ollama is running
                try:
                    req = urllib.request.Request(
                        f"{pconfig['base_url']}/api/tags",
                        headers={"User-Agent": "PLTM/2.0"}
                    )
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        data = json.loads(resp.read())
                        local_models = [m["name"] for m in data.get("models", [])]
                        status = "available"
                        pconfig = {**pconfig, "local_models": local_models}
                except Exception:
                    status = "not_running"
            elif pconfig["env_key"]:
                if os.environ.get(pconfig["env_key"]):
                    status = "available"
                else:
                    status = "no_api_key"
            
            available.append({
                "id": pid,
                "name": pconfig["name"],
                "tier": pconfig["tier"],
                "cost": pconfig["cost_per_1k_tokens"],
                "status": status,
                "default_model": pconfig["default_model"],
                "models": pconfig.get("local_models", pconfig["models"]),
            })
        
        return sorted(available, key=lambda x: (x["status"] != "available", x["tier"]))
    
    def route(self, task_type: str = "analysis", prefer_provider: Optional[str] = None,
              require_privacy: bool = False) -> Optional[str]:
        """Determine best provider for a task"""
        if prefer_provider and prefer_provider in self.PROVIDERS:
            return prefer_provider
        
        if require_privacy:
            return "ollama"
        
        rule = self.ROUTING_RULES.get(task_type, {"preferred_tier": 3, "fallback_tier": 4})
        preferred_tier = rule["preferred_tier"]
        fallback_tier = rule["fallback_tier"]
        
        # Find available provider at preferred tier
        available = self.get_available_providers()
        
        for p in available:
            if p["status"] == "available" and p["tier"] == preferred_tier:
                return p["id"]
        
        # Fallback tier
        for p in available:
            if p["status"] == "available" and p["tier"] == fallback_tier:
                return p["id"]
        
        # Any available
        for p in available:
            if p["status"] == "available":
                return p["id"]
        
        return None
    
    def call(self, prompt: str, provider: Optional[str] = None,
             model: Optional[str] = None, task_type: str = "analysis",
             system_prompt: Optional[str] = None,
             temperature: float = 0.3, max_tokens: int = 2048,
             require_privacy: bool = False) -> Dict:
        """
        Call an LLM through the router.
        
        Auto-selects provider based on task_type if not specified.
        Returns response with cost tracking.
        """
        # Route to provider
        if not provider:
            provider = self.route(task_type, require_privacy=require_privacy)
        
        if not provider:
            return {"ok": False, "err": "No available provider. Set API keys or start Ollama.",
                    "available": self.get_available_providers()}
        
        pconfig = self.PROVIDERS.get(provider)
        if not pconfig:
            return {"ok": False, "err": f"Unknown provider: {provider}"}
        
        model = model or pconfig["default_model"]
        
        start = time.time()
        
        try:
            if pconfig["api_format"] == "ollama":
                result = self._call_ollama(pconfig, model, prompt, system_prompt, temperature, max_tokens)
            elif pconfig["api_format"] == "openai":
                result = self._call_openai_compat(pconfig, provider, model, prompt, system_prompt, temperature, max_tokens)
            else:
                return {"ok": False, "err": f"Unknown API format: {pconfig['api_format']}"}
            
            elapsed_ms = int((time.time() - start) * 1000)
            
            # Estimate tokens and cost
            input_tokens = len(prompt.split()) * 1.3  # rough estimate
            output_tokens = len(result.get("text", "").split()) * 1.3
            cost = (input_tokens + output_tokens) / 1000 * pconfig["cost_per_1k_tokens"]
            
            # Log usage
            self._log_usage(provider, model, task_type, int(input_tokens), int(output_tokens), cost, elapsed_ms, True)
            
            result["provider"] = provider
            result["model"] = model
            result["ms"] = elapsed_ms
            result["cost_usd"] = round(cost, 6)
            result["tier"] = pconfig["tier"]
            
            return result
            
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            self._log_usage(provider, model, task_type, 0, 0, 0, elapsed_ms, False, str(e))
            return {"ok": False, "err": str(e)[:200], "provider": provider, "model": model, "ms": elapsed_ms}
    
    def _call_ollama(self, pconfig: Dict, model: str, prompt: str,
                     system_prompt: Optional[str], temperature: float, max_tokens: int) -> Dict:
        """Call Ollama local API"""
        # Fall back to an actually-pulled model when the configured default
        # isn't installed — otherwise every call 404s and extraction
        # silently returns nothing.
        try:
            req = urllib.request.Request(
                f"{pconfig['base_url']}/api/tags",
                headers={"User-Agent": "PLTM/2.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                pulled = [m["name"] for m in
                          json.loads(resp.read()).get("models", [])]
            if pulled and model not in pulled:
                model = pulled[0]
                logger.info(f"Ollama model not pulled; using {model}")
        except Exception:
            pass  # proceed with configured model — call itself will error

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system_prompt:
            payload["system"] = system_prompt
        
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{pconfig['base_url']}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
        
        return {"ok": True, "text": result.get("response", "")}
    
    def _call_openai_compat(self, pconfig: Dict, provider: str, model: str,
                            prompt: str, system_prompt: Optional[str],
                            temperature: float, max_tokens: int) -> Dict:
        """Call OpenAI-compatible API (Groq, DeepSeek, Together, OpenRouter, OpenAI)"""
        api_key = os.environ.get(pconfig["env_key"], "")
        if not api_key:
            return {"ok": False, "err": f"No API key. Set {pconfig['env_key']} environment variable."}
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "PLTM/2.0",
            "Accept": "application/json",
        }
        
        # OpenRouter needs extra headers
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://pltm.local"
            headers["X-Title"] = "PLTM Intelligence System"
        
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{pconfig['base_url']}/chat/completions",
            data=data,
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
        
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = result.get("usage", {})
        
        return {
            "ok": True,
            "text": text,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
    
    def _log_usage(self, provider: str, model: str, task_type: str,
                   input_tokens: int, output_tokens: int, cost: float,
                   latency_ms: int, success: bool, error: str = ""):
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                "INSERT INTO llm_usage (provider, model, task_type, input_tokens, output_tokens, cost_usd, latency_ms, success, error, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (provider, model, task_type, input_tokens, output_tokens, cost, latency_ms, int(success), error[:200], time.time())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to log LLM usage: {e}")
    
    def get_usage_stats(self, days: int = 30) -> Dict:
        """Get usage statistics"""
        conn = sqlite3.connect(str(self.db_path))
        cutoff = time.time() - (days * 86400)
        
        # Per-provider stats
        cursor = conn.execute("""
            SELECT provider, COUNT(*) as calls, SUM(cost_usd) as total_cost,
                   AVG(latency_ms) as avg_latency, SUM(input_tokens + output_tokens) as total_tokens,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
            FROM llm_usage WHERE timestamp > ?
            GROUP BY provider ORDER BY total_cost DESC
        """, (cutoff,))
        
        providers = []
        total_cost = 0
        total_calls = 0
        for r in cursor.fetchall():
            providers.append({
                "provider": r[0], "calls": r[1], "cost": round(r[2] or 0, 4),
                "avg_ms": int(r[3] or 0), "tokens": r[4] or 0,
                "success_rate": round((r[5] or 0) / max(r[1], 1), 2),
            })
            total_cost += r[2] or 0
            total_calls += r[1]
        
        # Per-task-type stats
        cursor = conn.execute("""
            SELECT task_type, COUNT(*) as calls, SUM(cost_usd) as cost
            FROM llm_usage WHERE timestamp > ?
            GROUP BY task_type ORDER BY calls DESC
        """, (cutoff,))
        
        tasks = [{"type": r[0], "calls": r[1], "cost": round(r[2] or 0, 4)} for r in cursor.fetchall()]
        
        conn.close()
        
        return {
            "period_days": days,
            "total_calls": total_calls,
            "total_cost_usd": round(total_cost, 4),
            "providers": providers,
            "by_task_type": tasks,
        }
