"""Bonus: a tiny web API.

    uvicorn agent.api:app --reload
    curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
         -d '{"question": "Is coffee good for you?"}'
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent import config
from agent.graph import run_agent

app = FastAPI(title="Cited Research Agent")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    max_steps: int = Field(default=config.DEFAULT_MAX_STEPS, ge=1, le=20)


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    # A plain (non-async) function: FastAPI runs it in a thread, so the slow
    # agent doesn't block other requests.
    result = run_agent(req.question, max_steps=req.max_steps)
    return {"answer": result["answer"], "sources": result["sources"],
            "steps": result["tool_calls"]}
