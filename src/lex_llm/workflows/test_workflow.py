from typing import Dict, Any, AsyncGenerator
from lex_llm.api.orchestrator import Orchestrator
from lex_llm.api.event_emitter import EventEmitter
from lex_llm.api.event_models import WorkflowRunRequest, WorkflowStepData, Source
import asyncio
import uuid


# Step 1: Query Analysis
async def query_analysis_step(
    context: Dict[str, Any], emitter: EventEmitter
) -> AsyncGenerator[str, None]:
    step_id = str(uuid.uuid4())
    user_input = context.get("user_input", "")
    conversation_history = context.get("conversation_history", [])
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="Query Analysis",
            status="started",
            input={
                "user_query": user_input,
                "history_length": len(conversation_history),
            },
        )
    )
    await asyncio.sleep(0.3)
    update_msg = "Parsing user intent and identifying required information..."
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="Query Analysis",
            status="in_progress",
            update=update_msg,
        )
    )
    reasoning_chunks = [
        "Let me analyze this query step by step. ",
        "The user seems to be asking about a complex topic that will require ",
        "multiple sources of information. I should search my knowledge base first, ",
        "then cross-reference with external sources to provide a comprehensive answer.",
    ]
    for chunk in reasoning_chunks:
        yield emitter.reasoning_chunk(chunk)
        await asyncio.sleep(0.15)
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="Query Analysis",
            status="completed",
            output={
                "intent": "information_request",
                "complexity": "high",
                "requires_sources": True,
            },
        )
    )


# Step 2: Knowledge Base Search
async def knowledge_base_search_step(
    context: Dict[str, Any], emitter: EventEmitter
) -> AsyncGenerator[str, None]:
    step_id = str(uuid.uuid4())
    user_input = context.get("user_input", "")
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="Knowledge Base Search",
            status="started",
            input={
                "search_terms": [
                    "artificial intelligence",
                    "machine learning",
                    "neural networks",
                ]
            },
        )
    )
    await asyncio.sleep(0.4)
    tool_call_kb = {
        "query": user_input,
        "limit": 10,
        "similarity_threshold": 0.8,
    }
    yield emitter.tool_call("search_knowledge_base", tool_call_kb)
    update_kb = "Found 15 potentially relevant documents, ranking by relevance..."
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="Knowledge Base Search",
            status="in_progress",
            update=update_kb,
        )
    )
    await asyncio.sleep(0.2)
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="Knowledge Base Search",
            status="completed",
            output={"documents_found": 8, "top_relevance_score": 0.94},
        )
    )
    # Add sources to context for later steps
    sources = [
        Source(
            id="kb_1",
            title="Introduction to Neural Networks",
            url="https://internal.kb/neural-networks-101",
        ),
        Source(
            id="kb_2",
            title="Machine Learning Fundamentals",
            url="https://internal.kb/ml-fundamentals",
        ),
    ]
    context["kb_sources"] = sources


# Step 3: External Source Search
async def external_source_search_step(
    context: Dict[str, Any], emitter: EventEmitter
) -> AsyncGenerator[str, None]:
    step_id = str(uuid.uuid4())
    user_input = context.get("user_input", "")
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="External Source Search",
            status="started",
            input={"search_query": user_input},
        )
    )
    tool_call_web = {
        "query": user_input,
        "num_results": 5,
        "recency_days": 30,
    }
    yield emitter.tool_call("web_search", tool_call_web)
    await asyncio.sleep(0.3)
    update_web = "Retrieving and processing web search results..."
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="External Source Search",
            status="in_progress",
            update=update_web,
        )
    )
    await asyncio.sleep(0.2)
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="External Source Search",
            status="completed",
            output={"web_results": 5, "average_authority_score": 8.7},
        )
    )
    sources = [
        Source(
            id="web_1",
            title="Recent Advances in AI Research",
            url="https://arxiv.org/abs/2024.12345",
        ),
        Source(
            id="web_2",
            title="Practical Applications of Deep Learning",
            url="https://research.google.com/pubs/deep-learning-apps",
        ),
        Source(
            id="web_3",
            title="Ethics in Artificial Intelligence",
            url="https://stanford.edu/ai-ethics-2024",
        ),
    ]
    context["web_sources"] = sources

    # Emit all sources (kb + web)
    all_sources = context.get("kb_sources", []) + sources
    yield emitter.sources(all_sources)


# Step 4: Information Synthesis
async def information_synthesis_step(
    context: Dict[str, Any], emitter: EventEmitter
) -> AsyncGenerator[str, None]:
    step_id = str(uuid.uuid4())
    kb_sources = context.get("kb_sources", [])
    web_sources = context.get("web_sources", [])
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="Information Synthesis",
            status="started",
            input={
                "internal_docs": len(kb_sources),
                "external_sources": len(web_sources),
            },
        )
    )
    reasoning_chunks = [
        "Now I need to synthesize information from multiple sources. ",
        "I'll start with the foundational concepts from the knowledge base, ",
        "then incorporate recent developments from external sources to provide ",
        "a comprehensive and up-to-date response.",
    ]
    for chunk in reasoning_chunks:
        yield emitter.reasoning_chunk(chunk)
        await asyncio.sleep(0.12)
    update_syn = "Cross-referencing sources and identifying key themes..."
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="Information Synthesis",
            status="in_progress",
            update=update_syn,
        )
    )
    tool_call_fact = {
        "claims": [
            "AI can process natural language",
            "Neural networks require training data",
        ],
        "sources": [s.id for s in (kb_sources + web_sources)[:3]],
    }
    yield emitter.tool_call("fact_checker", tool_call_fact)
    await asyncio.sleep(0.4)
    update_syn2 = "Fact-checking complete, organizing response structure..."
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="Information Synthesis",
            status="in_progress",
            update=update_syn2,
        )
    )
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="Information Synthesis",
            status="completed",
            output={"synthesized_points": 7, "confidence_score": 0.91},
        )
    )


# Step 5: Response Generation
async def response_generation_step(
    context: Dict[str, Any], emitter: EventEmitter
) -> AsyncGenerator[str, None]:
    step_id = str(uuid.uuid4())
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="Response Generation",
            status="started",
            input={"response_style": "comprehensive", "target_length": "medium"},
        )
    )
    reasoning_chunks = [
        "I'll structure my response to be clear and informative. ",
        "I'll start with a brief overview, then dive into specific details, ",
        "and conclude with practical implications.",
    ]
    for chunk in reasoning_chunks:
        yield emitter.reasoning_chunk(chunk)
        await asyncio.sleep(0.12)
    response_parts = [
        "Based on my analysis of multiple sources, I can provide you with a comprehensive answer. ",
        "\n\n**Overview**\n\nArtificial Intelligence represents a transformative field that combines ",
        "computational power with sophisticated algorithms to simulate human-like decision making. ",
        "The recent advances in this area have been particularly remarkable, with breakthrough ",
        "developments in natural language processing, computer vision, and autonomous systems.\n\n",
        "**Key Developments**\n\n1. **Neural Network Architectures**: Modern deep learning models ",
        "have evolved significantly, with transformer architectures revolutionizing how we approach ",
        "sequence modeling and attention mechanisms.\n\n2. **Training Methodologies**: ",
        "Self-supervised learning and few-shot learning approaches have reduced the dependency ",
        "on large labeled datasets, making AI more accessible and practical.\n\n",
        "3. **Ethical Considerations**: The field increasingly emphasizes responsible AI development, ",
        "focusing on fairness, transparency, and accountability in algorithmic decision-making.\n\n",
        "**Practical Applications**\n\nThese advances translate into real-world benefits across ",
        "healthcare, education, transportation, and scientific research. The integration of AI ",
        "systems into everyday workflows continues to accelerate, with particular emphasis on ",
        "human-AI collaboration rather than replacement.\n\n**Future Outlook**\n\n",
        "The trajectory suggests continued innovation in efficiency, interpretability, and ",
        "multimodal capabilities. As these technologies mature, we can expect more sophisticated ",
        "applications that better understand context and nuance in human communication.",
    ]
    for part in response_parts:
        for chunk in [part[i : i + 25] for i in range(0, len(part), 25)]:
            yield emitter.text_chunk(chunk)
            await asyncio.sleep(0.08)
    await asyncio.sleep(0.1)
    yield emitter.workflow_step(
        WorkflowStepData(
            step_id=step_id,
            name="Response Generation",
            status="completed",
            output={"response_length": 1247, "readability_score": 8.3},
        )
    )
    # Save the final response for orchestrator
    context["final_response"] = "".join(response_parts)


def get_metadata() -> dict:
    return {
        "workflow_id": "test_workflow",
        "name": "Test Workflow",
        "description": "Performs a mocked multi-step research and synthesis workflow.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_input": {"type": "string"},
                "conversation_id": {"type": "string"},
                "conversation_history": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["user_input", "conversation_id", "conversation_history"],
        },
        "steps": [
            {
                "name": "Query Analysis",
                "description": "Analyzes the user query and determines required information.",
            },
            {
                "name": "Knowledge Base Search",
                "description": "Searches the internal knowledge base for relevant documents.",
            },
            {
                "name": "External Source Search",
                "description": "Performs a web search for additional relevant information.",
            },
            {
                "name": "Information Synthesis",
                "description": "Synthesizes information from all sources and fact-checks claims.",
            },
            {
                "name": "Response Generation",
                "description": "Generates a comprehensive response for the user.",
            },
        ],
        "author": "Your Name or Team",
        "version": "1.0.0",
        "tags": ["mock", "demo", "research", "synthesis"],
    }


def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    """Returns the executable workflow object for the orchestrator."""
    return Orchestrator(
        request=request,
        steps=[
            query_analysis_step,
            knowledge_base_search_step,
            external_source_search_step,
            information_synthesis_step,
            response_generation_step,
        ],
        context={
            "conversation_history": request.conversation_history,
            "user_input": request.user_input,
        },
    )
