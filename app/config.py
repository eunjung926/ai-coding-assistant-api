import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    base_model_path: str = os.environ.get("BASE_MODEL_PATH", "Qwen/Qwen2-7B")
    vul_adapter_path: str = os.environ.get("VUL_ADAPTER_PATH", "adapters/vul")
    ske_adapter_path: str = os.environ.get("SKE_ADAPTER_PATH", "adapters/ske")
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "5059"))
    ngrok_authtoken: str = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    ngrok_domain: str = os.environ.get("NGROK_DOMAIN", "").strip()
    gpu_free_gb_threshold: float = float(
        os.environ.get("GPU_FREE_GB_THRESHOLD", "30")
    )
    eof_strings: tuple[str, ...] = ("<|endoftext|>", "</s>")


settings = Settings()
