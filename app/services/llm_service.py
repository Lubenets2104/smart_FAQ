"""LLM service for generating answers using pluggable providers (Strategy pattern)."""

import time
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import get_settings
from app.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()


# Shared system prompt for all providers
SYSTEM_PROMPT = """Ты - помощник по продукту SmartTask, облачной платформе для управления проектами и задачами.

Твоя задача - отвечать на вопросы пользователей, используя предоставленный контекст из базы знаний SmartTask.

Правила:
1. Отвечай только на основе предоставленного контекста.
2. Если информации нет в контексте - честно скажи: "К сожалению, я не нашёл информацию по этому вопросу в базе знаний."
3. Отвечай на русском языке, кратко и по делу.
4. Если вопрос касается нескольких тем - структурируй ответ.
5. Не выдумывай информацию, которой нет в контексте."""


class LLMProvider(ABC):
    """Abstract base class for LLM providers (Strategy interface)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider."""
        pass

    @abstractmethod
    def generate(
        self,
        user_message: str,
        system_prompt: str,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Generate a response from the LLM.

        Args:
            user_message: The user's message/prompt
            system_prompt: The system prompt to use
            max_tokens: Maximum tokens in response
            model: Optional model override

        Returns:
            Tuple of (response_text, tokens_used)
        """
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the provider is properly configured (API key set)."""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self):
        self._client = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return "claude-3-haiku-20240307"

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    def generate(
        self,
        user_message: str,
        system_prompt: str,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> Tuple[str, int]:
        """Generate response using Anthropic Claude."""
        client = self._get_client()
        model = model or self.default_model

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )

        answer = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens

        logger.debug(
            "Anthropic generation complete",
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return answer, tokens

    def is_configured(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(settings.anthropic_api_key)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider implementation."""

    def __init__(self):
        self._client = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return "gpt-3.5-turbo"

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=settings.openai_api_key)
        return self._client

    def generate(
        self,
        user_message: str,
        system_prompt: str,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> Tuple[str, int]:
        """Generate response using OpenAI GPT."""
        client = self._get_client()
        model = model or self.default_model

        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens

        logger.debug(
            "OpenAI generation complete",
            model=model,
            total_tokens=tokens,
        )

        return answer, tokens

    def is_configured(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(settings.openai_api_key)


class LLMProviderRegistry:
    """Registry for managing LLM providers."""

    def __init__(self):
        self._providers: Dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        """Register a provider."""
        self._providers[provider.name] = provider
        logger.debug("Registered LLM provider", provider=provider.name)

    def get(self, name: str) -> Optional[LLMProvider]:
        """Get a provider by name."""
        return self._providers.get(name)

    def get_available(self) -> list[str]:
        """Get list of available (configured) provider names."""
        return [
            name for name, provider in self._providers.items()
            if provider.is_configured()
        ]

    def list_all(self) -> list[str]:
        """Get list of all registered provider names."""
        return list(self._providers.keys())


class LLMService:
    """Service for interacting with LLM APIs using the Strategy pattern."""

    def __init__(self, provider_name: Optional[str] = None):
        """
        Initialize LLM service.

        Args:
            provider_name: Name of the provider to use. If None, uses settings.llm_provider
        """
        self._registry = LLMProviderRegistry()
        self._current_provider: Optional[LLMProvider] = None
        self._system_prompt = SYSTEM_PROMPT

        # Register default providers
        self._registry.register(AnthropicProvider())
        self._registry.register(OpenAIProvider())

        # Set the active provider
        provider_name = provider_name or settings.llm_provider
        self.set_provider(provider_name)

    @property
    def provider(self) -> Optional[LLMProvider]:
        """Get the current provider."""
        return self._current_provider

    @property
    def provider_name(self) -> str:
        """Get the current provider name."""
        return self._current_provider.name if self._current_provider else "none"

    @property
    def system_prompt(self) -> str:
        """Get the current system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, prompt: str) -> None:
        """Set a custom system prompt."""
        self._system_prompt = prompt

    def set_provider(self, name: str) -> None:
        """
        Set the active LLM provider.

        Args:
            name: Provider name ('anthropic' or 'openai')

        Raises:
            ValueError: If provider is not registered or not configured
        """
        provider = self._registry.get(name)
        if provider is None:
            available = self._registry.list_all()
            raise ValueError(
                f"Unknown provider '{name}'. Available: {available}"
            )

        if not provider.is_configured():
            raise ValueError(
                f"Provider '{name}' is not configured. Please set the API key."
            )

        self._current_provider = provider
        logger.info("LLM provider set", provider=name)

    def register_provider(self, provider: LLMProvider) -> None:
        """
        Register a custom LLM provider.

        Args:
            provider: LLMProvider instance to register
        """
        self._registry.register(provider)

    def get_available_providers(self) -> list[str]:
        """Get list of available (configured) providers."""
        return self._registry.get_available()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def generate_answer(
        self,
        question: str,
        context: str,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> Tuple[str, int, int]:
        """
        Generate an answer using the configured LLM provider.

        Args:
            question: User's question
            context: Relevant context from RAG
            max_tokens: Maximum tokens in response
            model: Optional model override

        Returns:
            Tuple of (answer, tokens_used, response_time_ms)

        Raises:
            RuntimeError: If no provider is configured
        """
        if self._current_provider is None:
            raise RuntimeError("No LLM provider configured")

        start_time = time.time()

        user_message = f"""Контекст из базы знаний:
{context if context else "Контекст не найден."}

Вопрос пользователя: {question}

Ответь на вопрос, используя только информацию из контекста."""

        try:
            answer, tokens = self._current_provider.generate(
                user_message=user_message,
                system_prompt=self._system_prompt,
                max_tokens=max_tokens,
                model=model,
            )

            response_time = int((time.time() - start_time) * 1000)

            logger.info(
                "Generated answer",
                provider=self._current_provider.name,
                tokens=tokens,
                response_time_ms=response_time,
            )

            return answer, tokens, response_time

        except Exception as e:
            logger.error(
                "Error generating answer",
                provider=self._current_provider.name,
                error=str(e)
            )
            raise

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Rough estimate: ~4 characters per token
            return len(text) // 4


# Global LLM service instance
llm_service = LLMService()
