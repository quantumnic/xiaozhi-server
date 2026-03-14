"""Main FastAPI application for xiaozhi-server."""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI, WebSocket, status

import config
from xiaozhi.session import XiaozhiSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="xiaozhi-server",
    description="Self-hosted backend for xiaozhi-esp32 devices",
    version="0.1.0",
)


@app.get("/")
async def root():
    return {
        "name": "xiaozhi-server",
        "version": "0.1.0",
        "status": "running",
        "provider": config.LLM_PROVIDER,
        "model": config.get_llm_model(),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/xiaozhi/v1/")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for xiaozhi ESP32 devices."""
    # Optional auth check
    if config.AUTH_TOKEN:
        auth = websocket.headers.get("authorization", "")
        if auth != f"Bearer {config.AUTH_TOKEN}":
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            logger.warning("Auth failed from %s", websocket.client)
            return

    device_id = websocket.headers.get("device-id", "unknown")
    client_id = websocket.headers.get("client-id", "unknown")
    logger.info("New connection: device=%s, client=%s", device_id, client_id)

    session = XiaozhiSession(websocket)
    session._device_id = device_id
    session._client_id = client_id
    await session.handle()


def main():
    """Run the server."""
    errors = config.validate()
    if errors:
        for err in errors:
            logger.error("Config error: %s", err)
        raise SystemExit(1)

    logger.info(
        "Starting xiaozhi-server on %s:%d (provider=%s, model=%s)",
        config.SERVER_HOST,
        config.SERVER_PORT,
        config.LLM_PROVIDER,
        config.get_llm_model(),
    )
    uvicorn.run(
        "server:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        ws="websockets",
    )


if __name__ == "__main__":
    main()
