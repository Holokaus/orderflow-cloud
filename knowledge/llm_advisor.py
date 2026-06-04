import json
import hashlib
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import os

from loguru import logger

from config.settings import LLMProvider, settings

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

__all__ = ['LLMAdvisor', 'LLMProvider', 'LLMResponse', 'ResponseCache']


@dataclass
class LLMResponse:
    content: str
    parsed: Optional[Dict] = None
    provider: str = ""
    model: str = ""
    tokens_used: int = 0
    cached: bool = False


class ResponseCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, tuple] = {}
        self.ttl = timedelta(seconds=ttl_seconds)

    def _hash_prompt(self, prompt: str, provider: str = "", model: str = "") -> str:
        combined = f"{provider}:{model}:{prompt}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get(self, prompt: str, provider: str = "", model: str = "") -> Optional[str]:
        key = self._hash_prompt(prompt, provider, model)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return response
            else:
                del self.cache[key]
        return None

    def set(self, prompt: str, response: str, provider: str = "", model: str = ""):
        key = self._hash_prompt(prompt, provider, model)
        self.cache[key] = (response, datetime.now())


class LLMAdvisor:
    _PROVIDER_CONFIG = {
        LLMProvider.OPENAI: {"type": "openai", "base_url": None, "default_model": "gpt-4o"},
        LLMProvider.OPENROUTER: {"type": "openai", "base_url": "https://openrouter.ai/api/v1", "default_model": "anthropic/claude-3.5-sonnet"},
        LLMProvider.GITHUB_MODELS: {"type": "openai", "base_url": "https://models.inference.ai.azure.com", "default_model": "gpt-4o"},
        LLMProvider.XAI: {"type": "openai", "base_url": "https://api.x.ai/v1", "default_model": "grok-2-latest"},
        LLMProvider.ZAI: {"type": "openai", "base_url": "https://api.z.ai/api/paas/v4", "default_model": "glm-5"},
        LLMProvider.GROQ: {"type": "openai", "base_url": "https://api.groq.com/openai/v1", "default_model": "llama-3.3-70b-versatile"},
        LLMProvider.OLLAMA: {"type": "openai", "base_url": "http://localhost:11434/v1", "default_model": "llama3"},
        LLMProvider.GEMINI: {"type": "gemini", "default_model": "gemini-2.0-flash"},
        LLMProvider.ANTHROPIC: {"type": "anthropic", "default_model": "claude-sonnet-4-20250514"},
    }

    def __init__(self, provider: Union[LLMProvider, str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, cache_responses: bool = True, base_url: Optional[str] = None):
        if provider is None:
            provider = settings.llm.provider
        elif isinstance(provider, str):
            provider = LLMProvider(provider)
        self.provider = provider
        self.cache = ResponseCache(settings.llm.cache_ttl_seconds) if cache_responses else None
        if provider not in self._PROVIDER_CONFIG:
            raise ValueError(f"Unknown provider: {provider}")
        config = self._PROVIDER_CONFIG[provider]
        self.model = model or settings.llm.model or config["default_model"]
        resolved_key = self._resolve_api_key(provider, api_key)
        if config["type"] == "openai":
            self._init_openai_compatible(provider, config, resolved_key, base_url)
        elif config["type"] == "gemini":
            self._init_gemini(resolved_key)
        elif config["type"] == "anthropic":
            self._init_anthropic(resolved_key)

    @classmethod
    def create_with_failover(cls, llm_config: 'LLMConfig') -> Optional['LLMAdvisor']:
        if not llm_config.enable_failover:
            try:
                api_key = llm_config.get_api_key(llm_config.provider)
                base_url = llm_config.get_base_url(llm_config.provider)
                return cls(provider=llm_config.provider, model=llm_config.model,
                          api_key=api_key, cache_responses=llm_config.cache_responses, base_url=base_url)
            except Exception as e:
                logger.error(f"LLM initialization failed: {e}")
                return None
        chain = llm_config.get_failover_chain()
        for provider, api_key, base_url in chain:
            try:
                logger.info(f"Attempting LLM provider: {provider.value}")
                advisor = cls(provider=provider, model=llm_config.model, api_key=api_key,
                             cache_responses=llm_config.cache_responses, base_url=base_url)
                test_response = advisor._call_llm("Test connection")
                if test_response and test_response.content:
                    logger.info(f"LLM initialized successfully: {provider.value}")
                    return advisor
                else:
                    raise ValueError("Empty response from test call")
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                logger.warning(f"{provider.value} failed: {error_msg}")
                continue
        logger.error("All LLM providers exhausted, no LLM available")
        return None

    def _resolve_api_key(self, provider: LLMProvider, explicit_key: Optional[str]) -> Optional[str]:
        if explicit_key:
            return explicit_key
        llm_config = settings.llm
        key_map = {
            LLMProvider.OPENAI: llm_config.openai_api_key,
            LLMProvider.GEMINI: llm_config.gemini_api_key,
            LLMProvider.ANTHROPIC: llm_config.anthropic_api_key,
            LLMProvider.OPENROUTER: llm_config.openrouter_api_key,
            LLMProvider.GITHUB_MODELS: llm_config.github_token,
            LLMProvider.XAI: llm_config.xai_api_key,
            LLMProvider.ZAI: llm_config.zai_api_key,
            LLMProvider.GROQ: llm_config.groq_api_key,
            LLMProvider.OLLAMA: "ollama",
        }
        if provider in key_map:
            key = key_map[provider]
            if key:
                return key
        env_map = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.GEMINI: "GEMINI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
            LLMProvider.GITHUB_MODELS: "GITHUB_TOKEN",
            LLMProvider.XAI: "XAI_API_KEY",
            LLMProvider.ZAI: "ZAI_API_KEY",
            LLMProvider.GROQ: "GROQ_API_KEY",
            LLMProvider.OLLAMA: None,
        }
        env_var = env_map.get(provider)
        if env_var:
            return os.getenv(env_var)
        if provider == LLMProvider.OLLAMA:
            return "ollama"
        return None

    def _init_openai_compatible(self, provider: LLMProvider, config: Dict, api_key: Optional[str], custom_base_url: Optional[str]):
        if not HAS_OPENAI:
            raise ImportError("openai package not installed. Install with: pip install openai")
        base_url = custom_base_url or config.get("base_url")
        init_kwargs = {"api_key": api_key or "dummy"}
        if base_url:
            init_kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**init_kwargs)
        self.provider_headers = {}
        if provider == LLMProvider.OPENROUTER:
            if settings.llm.openrouter_site_url:
                self.provider_headers["HTTP-Referer"] = settings.llm.openrouter_site_url
            if settings.llm.openrouter_app_name:
                self.provider_headers["X-Title"] = settings.llm.openrouter_app_name

    def _init_gemini(self, api_key: Optional[str]):
        if not HAS_GEMINI:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
        if not api_key:
            raise ValueError("Gemini API key not found.")
        genai.configure(api_key=api_key)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.client = genai.GenerativeModel(self.model)

    def _init_anthropic(self, api_key: Optional[str]):
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        if not api_key:
            raise ValueError("Anthropic API key not found.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        if self.cache:
            cached = self.cache.get(prompt, self.provider.value, self.model)
            if cached:
                return LLMResponse(content=cached, cached=True, provider=self.provider.value, model=self.model)
        try:
            config = self._PROVIDER_CONFIG[self.provider]
            if config["type"] == "openai":
                content, tokens = self._call_openai_compatible(prompt, system_prompt)
            elif config["type"] == "gemini":
                content, tokens = self._call_gemini(prompt, system_prompt)
            elif config["type"] == "anthropic":
                content, tokens = self._call_anthropic(prompt, system_prompt)
            else:
                raise ValueError(f"Unknown provider type: {config['type']}")
            if self.cache:
                self.cache.set(prompt, content, self.provider.value, self.model)
            return LLMResponse(content=content, provider=self.provider.value, model=self.model, tokens_used=tokens)
        except Exception as e:
            logger.error(f"LLM API call failed ({self.provider.value}): {e}")
            raise

    def _call_openai_compatible(self, prompt: str, system_prompt: Optional[str]) -> tuple:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        call_kwargs = {"model": self.model, "messages": messages, "temperature": 0.3}
        if self.provider == LLMProvider.OPENROUTER and self.provider_headers:
            call_kwargs["extra_headers"] = self.provider_headers
        response = self.client.chat.completions.create(**call_kwargs)
        content = response.choices[0].message.content
        tokens = getattr(response.usage, 'total_tokens', 0)
        return content, tokens

    def _call_gemini(self, prompt: str, system_prompt: Optional[str]) -> tuple:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self.client.generate_content(full_prompt, safety_settings=self.safety_settings)
        content = response.text
        return content, 0

    def _call_anthropic(self, prompt: str, system_prompt: Optional[str]) -> tuple:
        response = self.client.messages.create(
            model=self.model, max_tokens=4096,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return content, tokens
