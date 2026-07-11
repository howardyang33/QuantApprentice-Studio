from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..local_service import (
    describe_local_service_status,
    ensure_local_service_if_configured,
    local_api_url_if_enabled,
)


def _normalize_url(url: str) -> str:
    text = str(url or "").strip().rstrip("/")
    if not text:
        return ""
    if text.endswith("/chat/completions"):
        return text
    if text.endswith("/v1"):
        return f"{text}/chat/completions"
    return f"{text}/v1/chat/completions"


def _default_model() -> str:
    return (
        os.environ.get("QA_STUDIO_DEFAULT_MODEL", "")
        or os.environ.get("QA_STUDIO_LIVE_MODEL", "")
        or "gpt-oss-20b"
    ).strip()


@dataclass(frozen=True)
class StudioLLMConfig:
    api_url: str
    api_key: str
    default_model: str
    outer_loop_model: str
    apprentice_model: str
    timeout_seconds: float
    max_retries: int


class StudioLLMBackend:
    def __init__(self, config: Optional[StudioLLMConfig] = None) -> None:
        self.config = config or self.load_config()

    @staticmethod
    def load_config() -> StudioLLMConfig:
        raw_api_url = (
            os.environ.get("QA_STUDIO_API_URL", "")
            or os.environ.get("APPRENTICE_API_URL", "")
            or os.environ.get("TEACHER_LOOP_API_URL", "")
            or os.environ.get("LLM_API_URL", "")
        ).strip()
        api_url = _normalize_url(raw_api_url) or local_api_url_if_enabled()
        default_model = _default_model()
        return StudioLLMConfig(
            api_url=api_url,
            api_key=(
                os.environ.get("QA_STUDIO_API_KEY", "")
                or os.environ.get("APPRENTICE_API_KEY", "")
                or os.environ.get("TEACHER_LOOP_API_KEY", "")
                or os.environ.get("CHATANYWHERE_API_KEY", "")
            ).strip(),
            default_model=default_model,
            outer_loop_model=(os.environ.get("QA_STUDIO_OUTER_MODEL", "") or default_model).strip(),
            apprentice_model=(os.environ.get("QA_STUDIO_APPRENTICE_MODEL", "") or default_model).strip(),
            timeout_seconds=float(os.environ.get("QA_STUDIO_API_TIMEOUT_SECONDS", "180")),
            max_retries=int(os.environ.get("QA_STUDIO_API_MAX_RETRIES", "3")),
        )

    def describe(self) -> Dict[str, Any]:
        payload = asdict(self.config)
        try:
            payload["local_service"] = describe_local_service_status()
        except Exception as exc:
            payload["local_service"] = {"error": str(exc)}
        return payload

    def ensure_ready(self) -> Dict[str, Any]:
        if not self.config.api_url:
            raise RuntimeError(
                "No API endpoint configured for studio agents. "
                "Set QA_STUDIO_API_URL or enable the local GPT-OSS runtime."
            )
        return ensure_local_service_if_configured(self.config.api_url)

    def clean_env_overrides(self) -> Dict[str, str]:
        env = {
            "LLM_API_URL": self.config.api_url,
            "APPRENTICE_API_URL": self.config.api_url,
            "TEACHER_LOOP_API_URL": self.config.api_url,
            "APPRENTICE_API_MODEL": self.config.apprentice_model,
            "TEACHER_LOOP_API_MODEL": self.config.outer_loop_model,
        }
        if self.config.api_key:
            env["CHATANYWHERE_API_KEY"] = self.config.api_key
            env["APPRENTICE_API_KEY"] = self.config.api_key
            env["TEACHER_LOOP_API_KEY"] = self.config.api_key
        return env

    def chat_completion(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str = "",
        max_tokens: int = 1200,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        self.ensure_ready()
        model_name = str(model or self.config.default_model).strip()
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": list(messages),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if "gpt-oss" in model_name.lower():
            payload["reasoning_effort"] = "low"
        elif model_name.startswith("gpt-"):
            payload["reasoning_effort"] = "minimal"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        body = json.dumps(payload).encode("utf-8")
        last_error: Optional[Exception] = None
        for attempt in range(max(1, int(self.config.max_retries))):
            try:
                req = Request(self.config.api_url, data=body, headers=headers, method="POST")
                with urlopen(req, timeout=float(self.config.timeout_seconds)) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except HTTPError as exc:
                text = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
                last_error = RuntimeError(f"HTTPError {exc.code}: {text}")
                if attempt >= int(self.config.max_retries) - 1:
                    raise last_error
            except URLError as exc:
                last_error = exc
                if attempt >= int(self.config.max_retries) - 1:
                    raise
            except Exception as exc:
                last_error = exc
                if attempt >= int(self.config.max_retries) - 1:
                    raise
        raise RuntimeError(f"studio llm backend failed after retries: {last_error}") from last_error
