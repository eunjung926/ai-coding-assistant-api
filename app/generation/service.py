import asyncio
import json
import time
from threading import Thread

try:
    import re2 as re
except ImportError:
    import re

import torch
from loguru import logger
from transformers import StoppingCriteriaList, TextIteratorStreamer

from app.config import settings
from app.generation.criteria import EndOfFunctionCriteria
from app.models.loader import ModelRegistry


def _build_prompt(input_text: str) -> str:
    normalized = re.sub(r"\s+", " ", input_text)
    return f"### Instruction:\n{normalized}\n### Response:\n"


def _build_stopping_criteria(prompt_length: int):
    return StoppingCriteriaList(
        [
            EndOfFunctionCriteria(
                prompt_length,
                list(settings.eof_strings),
                ModelRegistry.tokenizer,
            )
        ]
    )


async def stream_completion(input_text: str):
    try:
        prompt = _build_prompt(input_text)
        inputs = ModelRegistry.tokenizer(prompt, return_tensors="pt").to(
            ModelRegistry.model.device
        )

        streamer = TextIteratorStreamer(
            ModelRegistry.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": ModelRegistry.tokenizer.pad_token_id,
            "streamer": streamer,
            "stopping_criteria": _build_stopping_criteria(
                inputs["input_ids"].shape[1]
            ),
        }

        thread = Thread(
            target=ModelRegistry.model.generate, kwargs=generation_kwargs
        )

        start_time = time.time()
        first_token_time = None
        total_tokens = 0

        thread.start()

        for new_text in streamer:
            should_stop = False
            for eof in settings.eof_strings:
                if eof in new_text:
                    new_text = new_text.split(eof)[0]
                    should_stop = True
                    break

            if new_text:
                if first_token_time is None:
                    first_token_time = time.time()
                    logger.info(f"TTFT: {first_token_time - start_time:.3f}s")

                total_tokens += len(
                    ModelRegistry.tokenizer.encode(new_text, add_special_tokens=False)
                )
                yield f"data: {json.dumps({'token': new_text})}\n\n"
                await asyncio.sleep(0)

            if should_stop:
                break

        end_time = time.time()
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else 0
        tps = total_tokens / total_time if total_time > 0 else 0

        logger.info(
            f"Total time: {total_time:.3f}s | Tokens: {total_tokens} | TPS: {tps:.2f}"
        )

        yield (
            "data: "
            + json.dumps(
                {
                    "done": True,
                    "stats": {
                        "ttft": round(ttft, 3),
                        "total_time": round(total_time, 3),
                        "total_tokens": total_tokens,
                        "tokens_per_sec": round(tps, 2),
                    },
                }
            )
            + "\n\n"
        )

    except Exception as error:
        logger.error(f"Error in streaming: {error}")
        yield f"data: {json.dumps({'error': str(error)})}\n\n"


async def complete(input_text: str) -> dict:
    try:
        prompt = _build_prompt(input_text)
        inputs = ModelRegistry.tokenizer(prompt, return_tensors="pt").to(
            ModelRegistry.model.device
        )

        with torch.no_grad():
            outputs = ModelRegistry.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=ModelRegistry.tokenizer.pad_token_id,
                stopping_criteria=_build_stopping_criteria(
                    inputs["input_ids"].shape[1]
                ),
            )

        generated = ModelRegistry.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        response = generated.split("### Response:")[-1].strip()

        for eof in settings.eof_strings:
            response = response.split(eof)[0].strip()

        return {"status": True, "generated_code": response}

    except Exception as error:
        logger.error(f"Error in autocomplete: {error}")
        return {"status": False, "msg": str(error)}
