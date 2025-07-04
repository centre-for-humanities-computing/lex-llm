"""
Main entry point for the Lex LLM Orchestration API.
"""

import uvicorn
from fastapi import FastAPI
from lex_llm.api.routes import router

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
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
