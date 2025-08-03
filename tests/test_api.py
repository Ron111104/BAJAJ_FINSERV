import os
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_run(monkeypatch):
    # Stub out the actual retrieval to avoid external calls
    monkeypatch.setattr("app.retrieve.query_document", lambda doc, q: "stubbed-answer")

    headers = {"Authorization": os.getenv("HACKRX_TOKEN")}
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/hackrx/run",
            json={"documents":"data/doc1.pdf","questions":["Test?"]},
            headers=headers
        )
    assert resp.status_code == 200
    assert resp.json() == {"answers":["stubbed-answer"]}
