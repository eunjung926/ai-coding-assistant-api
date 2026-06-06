import nest_asyncio
import uvicorn
from loguru import logger
from pyngrok import ngrok

from app.config import settings
from app.main import app


def start_ngrok_tunnel() -> None:
    if not settings.ngrok_authtoken:
        logger.info(
            "NGROK_AUTHTOKEN not set — skipping ngrok tunnel. "
            f"Server available at http://{settings.host}:{settings.port}"
        )
        return

    ngrok.set_auth_token(settings.ngrok_authtoken)
    try:
        if settings.ngrok_domain:
            public_url = ngrok.connect(settings.port, domain=settings.ngrok_domain)
        else:
            public_url = ngrok.connect(settings.port)
        logger.info(f"ngrok tunnel started: {public_url}")
    except Exception as error:
        logger.warning(f"ngrok failed: {error}")
        logger.info("Server running on localhost only")


if __name__ == "__main__":
    nest_asyncio.apply()
    start_ngrok_tunnel()

    uvicorn.run(
        app=app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=False,
        use_colors=False,
    )
