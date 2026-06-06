"""
PEFT LoRA fine-tuning for Qwen2-7B code generation adapters.

Usage (from project root):
  python -m training.train --adapter-type vul --data-path training/data/example_vul.jsonl
  python -m training.train --adapter-type ske --data-path training/data/example_ske.jsonl
"""

import argparse
from pathlib import Path

import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from training.config import LoRASettings, TrainSettings
from training.dataset import CausalLMCollator, InstructionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LoRA adapters with PEFT")
    parser.add_argument(
        "--adapter-type",
        choices=["vul", "ske"],
        required=True,
        help="Adapter to train: vul (vulnerable code) or ske (skeleton code)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to JSONL file with instruction/output pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save adapter (default: training/outputs/<adapter-type>)",
    )
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-7B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--use-4bit", action="store_true", help="Load base model in 4-bit")
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16 training")
    return parser.parse_args()


def load_base_model(base_model: str, use_4bit: bool):
    model_kwargs: dict = {"device_map": "auto"}

    if use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model


def main() -> None:
    args = parse_args()

    train_settings = TrainSettings(
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=not args.no_bf16,
        use_4bit=args.use_4bit,
    )
    lora_settings = LoRASettings()

    output_dir = args.output_dir or str(
        Path(__file__).parent / "outputs" / args.adapter_type
    )

    print(f"Training adapter : {args.adapter_type}")
    print(f"Base model       : {train_settings.base_model}")
    print(f"Dataset          : {args.data_path}")
    print(f"Output directory : {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(train_settings.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = InstructionDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_length=train_settings.max_seq_length,
    )

    model = load_base_model(train_settings.base_model, train_settings.use_4bit)
    model = get_peft_model(model, lora_settings.to_peft_config())

    if train_settings.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_settings.num_train_epochs,
        per_device_train_batch_size=train_settings.per_device_train_batch_size,
        gradient_accumulation_steps=train_settings.gradient_accumulation_steps,
        learning_rate=train_settings.learning_rate,
        warmup_ratio=train_settings.warmup_ratio,
        weight_decay=train_settings.weight_decay,
        logging_steps=train_settings.logging_steps,
        save_steps=train_settings.save_steps,
        save_total_limit=2,
        bf16=train_settings.bf16,
        fp16=not train_settings.bf16,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(tokenizer),
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Adapter saved to {output_dir}")
    print(f"Copy to adapters/{args.adapter_type}/ to use in the API server.")


if __name__ == "__main__":
    main()
