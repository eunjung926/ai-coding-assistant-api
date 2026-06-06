import asyncio

from loguru import logger
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings
from app.gpu import configure_cuda_devices

GPU_PICK_REASON = configure_cuda_devices()


class ModelRegistry:
    model = None
    tokenizer = None
    adapter_lock = asyncio.Lock()
    gpu_pick_reason = GPU_PICK_REASON

    @classmethod
    def initialize(cls) -> None:
        if cls.model is not None:
            return

        logger.info(
            "Loading models: base={!r}, vul={!r}, ske={!r}",
            settings.base_model_path,
            settings.vul_adapter_path,
            settings.ske_adapter_path,
        )

        cls.tokenizer = AutoTokenizer.from_pretrained(settings.base_model_path)

        logger.info(
            f"Loading base model (device_map=auto, gpu_pick={cls.gpu_pick_reason})..."
        )
        cls.model = AutoModelForCausalLM.from_pretrained(
            settings.base_model_path,
            device_map="auto",
            torch_dtype=torch.float32,
        )
        cls.model.config.use_cache = True

        logger.info("Loading vul adapter...")
        cls.model = PeftModel.from_pretrained(cls.model, settings.vul_adapter_path)
        cls.model.print_trainable_parameters()

        logger.info("Loading ske adapter...")
        cls.model.load_adapter(settings.ske_adapter_path, adapter_name="ske")

        cls.model.eval()
        cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id
        cls.model.config.pad_token_id = cls.model.config.eos_token_id

        logger.info("Model initialization complete")
