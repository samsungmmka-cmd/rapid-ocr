FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    APP_ENV=production API_KEY_FILE=/run/secrets/api_key

# Runtime library for headless OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-api.txt ./
COPY vendor ./vendor
RUN pip install -r requirements-api.txt
COPY pdf_table_extractor ./pdf_table_extractor

# Bake all OCR models into the image so production needs no internet access.
RUN python -c "from pdf_table_extractor.ocr_models import OCRModels; m = OCRModels(); [m._engine(n) for n in ('small', 'medium', 'en')]; assert not m.unavailable, m.unavailable"

RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "pdf_table_extractor.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
