import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OPUS_URL = os.environ.get("OPUS_URL", "http://127.0.0.1:8888").rstrip("/")
BATCH_SIZE = int(os.environ.get("OPUS_BATCH_SIZE", "64"))
TIMEOUT = float(os.environ.get("OPUS_TIMEOUT", "360"))


class TranslateRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    src_lang: str = Field(..., min_length=1, examples=["en"])
    trg_lang: str = Field(..., min_length=1, examples=["es"])


class TranslateResponse(BaseModel):
    src_lang: str
    trg_lang: str
    translations: list[str]


class ModelInfo(BaseModel):
    src_lang: str
    trg_lang: str
    loaded: bool
    memory_bytes: int | None = None
    memory_mb: float | None = None
    configuration: str | None = None
    port: str | int | None = None


class ModelsResponse(BaseModel):
    loaded: list[ModelInfo]
    models: list[ModelInfo]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT))
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(title="Opus MT API", lifespan=lifespan)


def _chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


async def _opus_json(response):
    try:
        return response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Opus server returned a non-JSON response")


async def _translate_chunk(client, src_lang, trg_lang, texts):
    try:
        response = await client.post(
            "{}/api/translate".format(OPUS_URL),
            json={"from": src_lang, "to": trg_lang, "texts": texts},
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="Opus server unreachable: {}".format(exc))

    payload = await _opus_json(response)
    if response.status_code >= 400 or "error" in payload:
        status = response.status_code if response.status_code >= 400 else 502
        raise HTTPException(
            status_code=status,
            detail=payload.get("error", "Opus translation failed"),
        )
    translations = payload.get("translations")
    if not isinstance(translations, list) or len(translations) != len(texts):
        raise HTTPException(status_code=502, detail="Opus server returned an unexpected batch response")
    return translations


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models", response_model=ModelsResponse)
async def models():
    client = app.state.client
    try:
        response = await client.get("{}/api/models".format(OPUS_URL))
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="Opus server unreachable: {}".format(exc))
    payload = await _opus_json(response)
    if response.status_code >= 400 or "error" in payload:
        status = response.status_code if response.status_code >= 400 else 502
        raise HTTPException(status_code=status, detail=payload.get("error", "Failed to list models"))
    return payload


@app.post("/translate", response_model=TranslateResponse)
async def translate(body: TranslateRequest):
    if BATCH_SIZE < 1:
        raise HTTPException(status_code=500, detail="OPUS_BATCH_SIZE must be at least 1")
    client = app.state.client
    translations = []
    for chunk in _chunks(body.texts, BATCH_SIZE):
        translations.extend(await _translate_chunk(client, body.src_lang, body.trg_lang, chunk))
    return TranslateResponse(
        src_lang=body.src_lang,
        trg_lang=body.trg_lang,
        translations=translations,
    )
