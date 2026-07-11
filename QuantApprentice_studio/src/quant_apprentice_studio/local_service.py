from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from .paths import studio_root


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _normalize_url(url: str) -> str:
    text = str(url or "").strip().rstrip("/")
    if not text:
        return ""
    if text.endswith("/chat/completions"):
        return text
    if text.endswith("/v1"):
        return f"{text}/chat/completions"
    return f"{text}/v1/chat/completions"


def _base_url_from_chat_url(url: str) -> str:
    text = str(url or "").strip()
    suffix = "/v1/chat/completions"
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text.rstrip("/")


@dataclass
class LocalVLLMConfig:
    enabled: bool
    prefer_local: bool
    autostart: bool
    host: str
    port: int
    model_path: str
    served_model_name: str
    conda_env: str
    cuda_visible_devices: str
    tensor_parallel_size: int
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    startup_timeout_seconds: float
    api_url: str
    pid_file: str
    log_file: str


def load_local_vllm_config() -> LocalVLLMConfig:
    host = os.environ.get("QA_STUDIO_LOCAL_HOST", "127.0.0.1").strip()
    port = int(os.environ.get("QA_STUDIO_LOCAL_PORT", "2310"))
    pid_file = studio_root() / "runs" / "local_model" / "gpt_oss_20b_vllm.pid"
    log_file = studio_root() / "runs" / "local_model" / "gpt_oss_20b_vllm.log"
    return LocalVLLMConfig(
        enabled=_env_flag("QA_STUDIO_LOCAL_ENABLED", True),
        prefer_local=_env_flag("QA_STUDIO_PREFER_LOCAL", True),
        autostart=_env_flag("QA_STUDIO_LOCAL_AUTOSTART", True),
        host=host,
        port=port,
        model_path=os.environ.get("QA_STUDIO_LOCAL_MODEL_PATH", "gpt-oss-20b").strip(),
        served_model_name=os.environ.get("QA_STUDIO_LOCAL_SERVED_MODEL_NAME", "gpt-oss-20b").strip(),
        conda_env=os.environ.get("QA_STUDIO_LOCAL_CONDA_ENV", "prrl").strip(),
        cuda_visible_devices=os.environ.get("QA_STUDIO_LOCAL_CUDA_VISIBLE_DEVICES", "0").strip(),
        tensor_parallel_size=int(os.environ.get("QA_STUDIO_LOCAL_TENSOR_PARALLEL_SIZE", "1")),
        max_model_len=int(os.environ.get("QA_STUDIO_LOCAL_MAX_MODEL_LEN", "32768")),
        max_num_seqs=int(os.environ.get("QA_STUDIO_LOCAL_MAX_NUM_SEQS", "128")),
        gpu_memory_utilization=float(os.environ.get("QA_STUDIO_LOCAL_GPU_MEMORY_UTILIZATION", "0.88")),
        startup_timeout_seconds=float(os.environ.get("QA_STUDIO_LOCAL_STARTUP_TIMEOUT_SECONDS", "240")),
        api_url=_normalize_url(f"http://{host}:{port}"),
        pid_file=str(pid_file),
        log_file=str(log_file),
    )


def local_api_url_if_enabled() -> str:
    config = load_local_vllm_config()
    if not (config.enabled and config.prefer_local):
        return ""
    return config.api_url


def _probe_url(url: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return {"status": resp.status, "body": body}
    except URLError:
        return None
    except Exception:
        return None


def _service_healthy(config: LocalVLLMConfig, timeout: float = 2.0) -> bool:
    base = _base_url_from_chat_url(config.api_url)
    health = _probe_url(f"{base}/health", timeout=timeout)
    if health and 200 <= int(health["status"]) < 300:
        return True
    models = _probe_url(f"{base}/v1/models", timeout=timeout)
    return bool(models and 200 <= int(models["status"]) < 300)


def _pid_from_file(path: str) -> Optional[int]:
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
        if not text:
            return None
        return int(text)
    except Exception:
        return None


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def describe_local_service_status() -> Dict[str, Any]:
    config = load_local_vllm_config()
    pid = _pid_from_file(config.pid_file)
    return {
        "enabled": config.enabled,
        "prefer_local": config.prefer_local,
        "autostart": config.autostart,
        "api_url": config.api_url,
        "base_url": _base_url_from_chat_url(config.api_url),
        "model_path": config.model_path,
        "model_path_exists": Path(config.model_path).exists(),
        "conda_env": config.conda_env,
        "cuda_visible_devices": config.cuda_visible_devices,
        "tensor_parallel_size": config.tensor_parallel_size,
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.max_num_seqs,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "pid_file": config.pid_file,
        "log_file": config.log_file,
        "pid": pid,
        "pid_running": bool(pid and _pid_running(pid)),
        "service_healthy": _service_healthy(config, timeout=1.0),
    }


def _build_launch_command(config: LocalVLLMConfig) -> str:
    source_conda = f"source {shlex.quote(os.environ.get('QA_STUDIO_CONDA_SH', '~/miniconda3/etc/profile.d/conda.sh'))}"
    args = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config.model_path,
        "--served-model-name",
        config.served_model_name,
        "--trust-remote-code",
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--max-model-len",
        str(config.max_model_len),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--disable-frontend-multiprocessing",
        "--enforce-eager",
    ]
    command = " ".join(shlex.quote(x) for x in args)
    return (
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        f"{source_conda}; "
        f"conda activate {shlex.quote(config.conda_env)}; "
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(config.cuda_visible_devices)}; "
        f"exec {command}"
    )


def _tail_log(path: str, max_lines: int = 40) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def start_local_service(force_restart: bool = False) -> Dict[str, Any]:
    config = load_local_vllm_config()
    if not config.enabled:
        raise RuntimeError("Local GPT-OSS service is disabled by QA_STUDIO_LOCAL_ENABLED=0.")
    if not Path(config.model_path).exists():
        raise FileNotFoundError(f"Local model path not found: {config.model_path}")

    status = describe_local_service_status()
    if status["service_healthy"] and not force_restart:
        return {**status, "action": "reused_existing"}

    if force_restart:
        stop_local_service()

    Path(config.pid_file).parent.mkdir(parents=True, exist_ok=True)
    Path(config.log_file).parent.mkdir(parents=True, exist_ok=True)
    launch_cmd = _build_launch_command(config)
    log_handle = open(config.log_file, "a", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", "-lc", launch_cmd],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(studio_root()),
        start_new_session=True,
    )
    Path(config.pid_file).write_text(str(process.pid), encoding="utf-8")

    deadline = time.time() + max(5.0, float(config.startup_timeout_seconds))
    while time.time() < deadline:
        if _service_healthy(config, timeout=2.0):
            return {**describe_local_service_status(), "action": "started"}
        if process.poll() is not None:
            break
        time.sleep(2.0)

    raise RuntimeError(
        "Local GPT-OSS vLLM service failed to become healthy. "
        f"pid={process.pid} log_tail=\n{_tail_log(config.log_file)}"
    )


def stop_local_service() -> Dict[str, Any]:
    config = load_local_vllm_config()
    pid = _pid_from_file(config.pid_file)
    if not pid:
        return {**describe_local_service_status(), "action": "no_pid_file"}
    if not _pid_running(pid):
        try:
            Path(config.pid_file).unlink(missing_ok=True)
        except Exception:
            pass
        return {**describe_local_service_status(), "action": "stale_pid_removed"}

    os.killpg(pid, signal.SIGTERM)
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if not _pid_running(pid):
            break
        time.sleep(0.5)
    if _pid_running(pid):
        os.killpg(pid, signal.SIGKILL)
    try:
        Path(config.pid_file).unlink(missing_ok=True)
    except Exception:
        pass
    return {**describe_local_service_status(), "action": "stopped"}


def ensure_local_service_if_configured(target_api_url: str) -> Dict[str, Any]:
    config = load_local_vllm_config()
    if not config.enabled:
        return describe_local_service_status()
    normalized_target = _normalize_url(target_api_url)
    if normalized_target != config.api_url:
        return describe_local_service_status()
    if _service_healthy(config, timeout=1.0):
        return {**describe_local_service_status(), "action": "healthy"}
    if not config.autostart:
        raise RuntimeError(
            "Local GPT-OSS service is preferred but not healthy, and autostart is disabled. "
            f"Expected local endpoint: {config.api_url}"
        )
    return start_local_service(force_restart=False)
