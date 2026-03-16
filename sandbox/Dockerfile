FROM python:3.11-slim

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    flask \
    requests \
    httpx \
    pydantic \
    sqlalchemy \
    pytest \
    pytest-asyncio \
    anyio[trio]

WORKDIR /workspace

CMD ["pytest", "-v", "--tb=short"]
