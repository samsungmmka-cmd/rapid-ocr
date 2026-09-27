"""Vercel entrypoint: a top-level ``app = FastAPI(...)`` as Vercel's detector expects.

All routes, limits and API-key checks come from the real API in
``pdf_table_extractor/api.py``, mounted at the root. The outer app runs the
fail-closed security check at startup (mounted apps' lifespans do not run).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from pdf_table_extractor.api import app as api_app
from pdf_table_extractor.api import validate_security_config


@asynccontextmanager
async def lifespan(_app: FastAPI):
    validate_security_config()  # APP_ENV=production without a strong API key -> refuse to start
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/", api_app)
