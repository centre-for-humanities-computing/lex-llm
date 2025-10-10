"""
Main entry point for the Lex LLM Orchestration API.
"""
import os
import uvicorn
from fastapi import FastAPI
from lex_llm.api.routes import router

host = os.getenv("DEPLOY_HOST", "http://localhost")
port = int(os.getenv("DEPLOY_PORT", "8001"))

app = FastAPI(
    title="Lex LLM Orchestration API",
    description="API for orchestrating LLM tasks.",
    version="0.1.0",
)
app.include_router(router)


@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "ok"}


def main() -> None:
    """Run the FastAPI application."""
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
