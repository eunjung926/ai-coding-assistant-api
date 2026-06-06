import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n### Response:\n"


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_training_text(instruction: str, output: str) -> tuple[str, str]:
    prompt = INSTRUCTION_TEMPLATE.format(instruction=instruction.strip())
    response = output.strip()
    return prompt, response


class InstructionDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = load_jsonl(data_path)
        self._validate_examples()

    def _validate_examples(self) -> None:
        if not self.examples:
            raise ValueError("Training dataset is empty.")

        for index, example in enumerate(self.examples):
            if "instruction" not in example or "output" not in example:
                raise ValueError(
                    f"Row {index} must contain 'instruction' and 'output' fields."
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        prompt, response = format_training_text(
            example["instruction"], example["output"]
        )

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        response_ids = self.tokenizer(
            response,
            add_special_tokens=False,
        )["input_ids"]

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            response_ids = response_ids + [eos_id]

        input_ids = (prompt_ids + response_ids)[: self.max_seq_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_seq_length]

        if len(labels) < len(input_ids):
            labels = labels + [-100] * (len(input_ids) - len(labels))

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CausalLMCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)

        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            pad_len = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
