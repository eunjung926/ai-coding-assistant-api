from dataclasses import dataclass, field

from peft import LoraConfig


@dataclass
class LoRASettings:
    """Matches the adapter_config.json used in adapters/vul and adapters/ske."""

    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_peft_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type,
        )


@dataclass
class TrainSettings:
    base_model: str = "Qwen/Qwen2-7B"
    max_seq_length: int = 2048
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100
    bf16: bool = True
    gradient_checkpointing: bool = True
    use_4bit: bool = False
