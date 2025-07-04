from collections.abc import AsyncGenerator
from lex_llm.api.event_models import WorkflowRunRequest
from lex_llm.orchestrator import WorkflowOrchestrator


def run_test_workflow(request: WorkflowRunRequest) -> AsyncGenerator[str, None]:
    """Implements the test workflow logic, previously in _run_workflow."""
    # Import here to avoid circular imports
    from lex_llm.api.event_emitter import EventEmitter
    from lex_llm.api.event_models import (
        WorkflowStepData,
        Source,
    )
    import asyncio
    import uuid

    async def _run() -> AsyncGenerator[str, None]:
        emitter = EventEmitter(conversation_id=request.conversation_id)
        # orchestrator.conversation_history = request.conversation_history
        # orchestrator.conversation_history.append(
        #     ConversationMessage(role="user", content=request.user_input)
        # )
        # --- Step 1: Query Analysis ---
        analysis_step = str(uuid.uuid4())
        yield emitter.workflow_step(
            WorkflowStepData(
                step_id=analysis_step,
                name="Query Analysis",
                status="started",
                input={
                    "user_query": request.user_input,
                    "history_length": len(request.conversation_history),
                },
            )
        )
        await asyncio.sleep(0.3)
        update_msg = "Parsing user intent and identifying required information..."
        yield emitter.workflow_step(
            WorkflowStepData(
                step_id=analysis_step,
                name="Query Analysis",
                status="in_progress",
                update=update_msg,
            )
        )
        reasoning1 = "Let me analyze this query step by step. "
        yield emitter.reasoning_chunk(reasoning1)
        await asyncio.sleep(0.2)
        reasoning2 = (
            "The user seems to be asking about a complex topic that will require "
        )
        yield emitter.reasoning_chunk(reasoning2)
        await asyncio.sleep(0.1)
        reasoning3 = (
            "multiple sources of information. I should search my knowledge base first, "
        )
        yield emitter.reasoning_chunk(reasoning3)
        await asyncio.sleep(0.15)
        reasoning4 = "then cross-reference with external sources to provide a comprehensive answer."
        yield emitter.reasoning_chunk(reasoning4)
        await asyncio.sleep(0.2)
        yield emitter.workflow_step(
            WorkflowStepData(
                step_id=analysis_step,
                name="Query Analysis",
                status="completed",
                output={
                    "intent": "information_request",
                    "complexity": "high",
                    "requires_sources": True,
                },
            )
        )
        # Step 2: Knowledge Base Search (runs concurrently with Step 3)
        kb_step = str(uuid.uuid4())
        yield emitter.workflow_step(
            WorkflowStepData(
                step_id=kb_step,
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
        # Step 3: External Source Search (concurrent)
        external_step = str(uuid.uuid4())
        yield emitter.workflow_step(
            WorkflowStepData(
                step_id=external_step,
                name="External Source Search",
                status="started",
                input={"search_query": request.user_input},
            )
        )
        await asyncio.sleep(0.4)
        # Tool call for knowledge base
        tool_call_kb = {
            "query": request.user_input,
            "limit": 10,
            "similarity_threshold": 0.8,
        }
        yield emitter.tool_call("search_knowledge_base", tool_call_kb)
        update_kb = "Found 15 potentially relevant documents, ranking by relevance..."
        yield emitter.workflow_step(
            WorkflowStepData(
                step_id=kb_step,
                name="Knowledge Base Search",
                status="in_progress",
                update=update_kb,
            )
        )
        # Tool call for web search
        tool_call_web = {
            "query": request.user_input,
            "num_results": 5,
            "recency_days": 30,
        }
        yield emitter.tool_call("web_search", tool_call_web)
        await asyncio.sleep(0.3)
        update_web = "Retrieving and processing web search results..."
        yield emitter.workflow_step(
            WorkflowStepData(
                step_id=external_step,
                name="External Source Search",
                status="in_progress",
                update=update_web,
            )
        )
        # Complete knowledge base search
        yield emitter.workflow_step(
            WorkflowStepData(
                step_id=kb_step,
                name="Knowledge Base Search",
                status="completed",
                output={"documents_found": 8, "top_relevance_score": 0.94},
            )
        )
        await asyncio.sleep(0.2)
        # Sources become available
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
        yield emitter.sources(sources)

        # Complete external search
        yield emitter.workflow_step(
            WorkflowStepData(
                step_id=external_step,
                name="External Source Search",
                status="completed",
                output={"web_results": 5, "average_authority_score": 8.7},
            )
        )
        # Step 4: Information Synthesis
        synthesis_step = str(uuid.uuid4())
        yield emitter.workflow_step(
            WorkflowStepData(
                step_id=synthesis_step,
                name="Information Synthesis",
                status="started",
                input={"internal_docs": 8, "external_sources": 5},
            )
        )
        await asyncio.sleep(0.3)
        reasoning5 = "Now I need to synthesize information from multiple sources. "
        yield emitter.reasoning_chunk(reasoning5)
        await asyncio.sleep(0.15)
        reasoning6 = (
            "I'll start with the foundational concepts from the knowledge base, "
        )
        yield emitter.reasoning_chunk(reasoning6)
        await asyncio.sleep(0.1)
        reasoning7 = (
            "then incorporate recent developments from external sources to provide "
        )
        yield emitter.reasoning_chunk(reasoning7)
        await asyncio.sleep(0.12)
        reasoning8 = "a comprehensive and up-to-date response."
        yield emitter.reasoning_chunk(reasoning8)
        await asyncio.sleep(0.2)
        update_syn = "Cross-referencing sources and identifying key themes..."
        yield emitter.workflow_step(
            WorkflowStepData(
                step_id=synthesis_step,
                name="Information Synthesis",
                status="in_progress",
                update=update_syn,
            )
        )
        # Tool call for fact checking
        tool_call_fact = {
            "claims": [
                "AI can process natural language",
                "Neural networks require training data",
            ],
            "sources": [s.id for s in sources[:3]],
        }
        yield emitter.tool_call("fact_checker", tool_call_fact)
        await asyncio.sleep(0.4)
        update_syn2 = "Fact-checking complete, organizing response structure..."
        yield emitter.workflow_step(
            WorkflowStepData(
                step_id=synthesis_step,
                name="Information Synthesis",
                status="in_progress",
                update=update_syn2,
            )
        )
        yield emitter.workflow_step(
            WorkflowStepData(
                step_id=synthesis_step,
                name="Information Synthesis",
                status="completed",
                output={"synthesized_points": 7, "confidence_score": 0.91},
            )
        )
        # Step 5: Response Generation
        response_step = str(uuid.uuid4())
        yield emitter.workflow_step(
            WorkflowStepData(
                step_id=response_step,
                name="Response Generation",
                status="started",
                input={"response_style": "comprehensive", "target_length": "medium"},
            )
        )
        await asyncio.sleep(0.2)
        reasoning9 = "I'll structure my response to be clear and informative. "
        yield emitter.reasoning_chunk(reasoning9)
        await asyncio.sleep(0.1)
        reasoning10 = (
            "I'll start with a brief overview, then dive into specific details, "
        )
        yield emitter.reasoning_chunk(reasoning10)
        await asyncio.sleep(0.12)
        reasoning11 = "and conclude with practical implications."
        yield emitter.reasoning_chunk(reasoning11)
        await asyncio.sleep(0.15)
        # Generate response text in chunks
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
                step_id=response_step,
                name="Response Generation",
                status="completed",
                output={"response_length": 1247, "readability_score": 8.3},
            )
        )
        # --- At the end, add the assistant message to conversation history ---
        # (Now handled by orchestrator)

    return _run()


def get_metadata() -> dict:
    """Returns metadata for the discovery endpoint."""
    return {
        "workflow_id": "test_workflow",
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
    }


def get_workflow() -> WorkflowOrchestrator:
    """Returns the executable workflow object for the orchestrator."""
    return WorkflowOrchestrator(
        workflow_id="test_workflow", workflow_func=run_test_workflow
    )
