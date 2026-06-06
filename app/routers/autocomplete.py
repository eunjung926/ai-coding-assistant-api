from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from loguru import logger

from app.generation.service import complete, stream_completion
from app.models.loader import ModelRegistry

router = APIRouter(prefix="/autocompleteNLtoCode", tags=["autocomplete"])

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


@router.get("/vul/stream")
async def autocomplete_vul_stream(
    input: str = Query(..., min_length=1, max_length=2048),
):
    async with ModelRegistry.adapter_lock:
        ModelRegistry.model.set_adapter("default")

    return StreamingResponse(
        stream_completion(input),
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )


@router.get("/ske/stream")
async def autocomplete_ske_stream(
    input: str = Query(..., min_length=1, max_length=2048),
):
    async with ModelRegistry.adapter_lock:
        ModelRegistry.model.set_adapter("ske")

    return StreamingResponse(
        stream_completion(input),
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )


@router.get("/vul")
async def autocomplete_vul(
    input: str = Query(..., min_length=1, max_length=2048),
):
    logger.debug("vulnerable code generating")
    async with ModelRegistry.adapter_lock:
        ModelRegistry.model.set_adapter("default")
        return await complete(input)


@router.get("/ske")
async def autocomplete_ske(
    input: str = Query(..., min_length=1, max_length=2048),
):
    logger.debug("skeleton code generating")
    async with ModelRegistry.adapter_lock:
        ModelRegistry.model.set_adapter("ske")
        return await complete(input)
