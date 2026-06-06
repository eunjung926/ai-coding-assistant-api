from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def health_check():
    return {"message": "AI Coding Assistant API", "docs": "/docs"}
