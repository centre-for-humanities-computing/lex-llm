import json
from typing import AsyncGenerator, Dict, Any, List
from ..api.orchestrator import Orchestrator
from ..api.event_emitter import EventEmitter
from ..api.event_models import WorkflowRunRequest, Source
from ..api.connectors.lex_db_connector import LexArticle, LexDBConnector
from ..api.connectors.openai_provider import OpenAIProvider


# Step 1: Search the knowledge base
async def search_knowledge_base(
    context: Dict[str, Any], emitter: EventEmitter
) -> AsyncGenerator[None, None]:
    """Queries the KB and prepares sources for emission."""
    lex_db_connector = LexDBConnector()
    user_input = context.get("user_input", "")

    documents = await lex_db_connector.vector_search(
        query=user_input, top_k=10, index_name="openai_large_3_sections"
    )
    context["retrieved_docs"] = documents

    # sources = [Source(id=doc.id, title=doc.title, url=doc.url) for doc in documents]
    yield


# Step 2: Generate the response using the retrieved documents
async def generate_response(
    context: Dict[str, Any], emitter: EventEmitter
) -> AsyncGenerator[str, None]:
    """
    Generates a response using retrieved documents, enforces strict factual grounding,
    mimics the tone of the sources, answers in Danish, avoids links, and emits only
    the sources actually used in the response.
    """
    """
    Generates a response using retrieved documents, enforces strict factual grounding,
    mimics the tone of the sources, answers in Danish, avoids links, and emits only
    the sources actually used in the response.
    """
    llm_provider = OpenAIProvider()
    retrieved_docs: List[LexArticle] = context.get("retrieved_docs", [])
    user_input: str = context.get("user_input", "").strip()
    conversation_history: List[Dict[str, str]] = context.get("conversation_history", [])

    if not retrieved_docs:
        # If no documents were retrieved, defer immediately
        deferral_message = "Jeg beklager, men jeg er ikke i stand til at besvare dit spørgsmål ud fra Lex' artikler."
        yield emitter.text_chunk(deferral_message)
        context["final_response"] = deferral_message
        return

    # Format retrieved documents for the prompt
    docs_text = "\n\n".join(
        [f"Titel: {doc.title}\nIndhold: {doc.text}" for doc in retrieved_docs]
    )

    # Construct the system prompt in clean, clear markdown
    system_prompt = """Du er 'den danske chatbot', en chatbot der er en del af Lex og som hjælper brugere med at finde viden ud fra encyklopædiske artikler. Din opgave er at analysere de leverede artikler og give et præcist, faktabaseret svar på brugerens spørgsmål – men kun hvis informationen tydeligt og direkte støttes af artiklerne.

## Regler 
- Svar ALTID på dansk. Hvis nogen spørger på engelsk eller beder dig svare på et andet sprog skal du forklare, at du kun kan svare på dansk.
- Start alle svar med en enkelt sætning, hvor du beskriver din fortolkning af brugerens spørgsmål så tydeligt som muligt. F.eks. hvis brugeren spørger "Forklar for en 7-årig hvad forskellen er på en fregat og en galej?" indled da dit svar med "Her får du en forklaring på hvad forskellen er på en fregat og en galej, forklaret for en 7-årig" eller noget lignende.
- Brug ALDRIG markdown-links (f.eks. [titel](url)) i dit svar – ingen kildehenvisninger direkte i teksten.
- Gengiv tonen i artiklerne – typisk neutral, encyklopædisk, videnskabelig og faktuel. Undgå personlig tone, formodninger eller fortolkninger, og tag en videnskabelig vinkel på f.eks. teologiske eller spirituelle spørgsmål dog uden at være respektløs overfor andres tro og verdensbilleder. 
- Undgå at bevæge dig ud over Lex' domæne som en encyklopædi. Lad f.eks. være med at foreslå opskrifter, træningsregimer, dieter eller andre livsstilsråd. Hvis brugeren forsøger at lede dig væk fra en faktuel samtale skal du minde brugeren om, at du kun fungerer som en chatbot, der leder efter svar i Lex' artikler.
- Hvis svaret ikke kan støttes af artiklerne, svar: "Jeg beklager, men jeg er ikke i stand til at finde et svar på dit spørgsmål i vores artikler." Hvis brugeren beder om en grund må du give dit bedste bud på, hvad der gik galt. Det skal være klart for brugeren, at det kun er din vurdering af problemet.
- Hvis du har brug for at henvise til noget specifikt fra artiklerne, skal det gøres som et ordret citat. Ellers bør du undgå at henvise direkte til artiklerne, og bør i stedet fremlægge indholdet med dine egne ord.
- Hvis samtalen fortsætter, må du henvise til tidligere artikler, så længe de stadig er relevante og støtter dit svar.
- Hvis du mangler information eller hvis brugeren stiller tvetydige spørgsmål, skal du bede om at få opklaret brugerens spørgsmål, før du svarer. 

"""

    sources = f"""
    ## Artikler
{docs_text}
    """

    # Prepare messages
    if not conversation_history:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": sources},
            {"role": "user", "content": user_input},
        ]
        context["system_prompt"] = system_prompt
    else:
        messages = conversation_history + [
            {"role": "system", "content": sources},
            {"role": "user", "content": user_input},
        ]  # type: ignore

    # Stream response from LLM
    full_response = ""
    async for chunk in llm_provider.generate_stream(messages):  # type: ignore
        full_response += chunk
        yield emitter.text_chunk(chunk)
    context["final_response"] = full_response

    # Identify which sources were actually used
    used_sources = await extract_used_sources_via_llm(
        response=full_response,
        retrieved_docs=retrieved_docs,
        llm_provider=llm_provider,
    )
    context["sources"] = used_sources
    # Emit only the used sources (not all retrieved ones)
    yield emitter.sources(
        [Source(id=src.id, title=src.title, url=src.url) for src in used_sources]
    )


async def extract_used_sources_via_llm(
    response: str,
    retrieved_docs: List[LexArticle],
    llm_provider: OpenAIProvider,
) -> List[LexArticle]:
    """
    Uses an LLM to analyze the generated response and return only the Document objects
    that were actually used in generating the answer.
    """
    if not response.strip() or "Jeg beklager, men jeg er ikke i stand" in response:
        return []  # No sources used if deferral message was returned

    source_descriptions = "\n".join(
        [
            f"ID: {doc.id} | Title: {doc.title} | Content: {doc.text}"
            for doc in retrieved_docs
        ]
    )

    attribution_prompt = f"""
Analyze the assistant's response below and determine which of the provided sources were actually used to generate the answer.

Return ONLY a JSON-formatted list of source IDs that are directly referenced or used in the response.
If no sources were used, return an empty list.
    
Do not include explanations or markdown formatting.

## Sources
{source_descriptions}

## Assistant Response
{response}

## Expected Output Format
["id1", "id2", ...]
"""
    messages = [
        {
            "role": "system",
            "content": "You are a careful analyst who identifies which sources were used in a response from a chatbot.",
        },
        {"role": "user", "content": attribution_prompt},
    ]
    try:
        attribution_result = await llm_provider.generate(messages)  # type: ignore
        attribution_result = attribution_result.strip()
        # Clean up common LLM quirks
        if attribution_result.startswith("```json"):
            attribution_result = attribution_result[7:].split("```")[0].strip()
        elif attribution_result.startswith("```"):
            attribution_result = attribution_result[3:].split("```")[0].strip()
        used_ids = [int(id) for id in json.loads(attribution_result)]
        used_sources = [doc for doc in retrieved_docs if doc.id in used_ids]
        return used_sources

    except (json.JSONDecodeError, Exception) as e:
        # Log error if possible, fallback to empty
        print(f"Failed to parse attribution result: {e}")
        return []


# This function is the entry point called by the API route
def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    """Configures and returns the RAG workflow orchestrator."""
    return Orchestrator(
        request=request,
        steps=[
            search_knowledge_base,
            generate_response,
        ],
        context={"conversation_history": request.conversation_history},
    )


def get_metadata() -> dict:
    return {
        "workflow_id": "alpha_workflow_v1",
        "name": "Alpha Workflow v1",
        "description": "Version 1 of the workflow for the alpha version. Performs a simple retrieval-augmented generation (RAG) using a knowledge base and an LLM and outputs a source list.",
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
                "name": "Knowledge Base Search",
                "description": "Queries the internal knowledge base for relevant documents using vector search.",
                "inputs": ["user_input"],
                "outputs": ["retrieved_docs", "sources"],
            },
            {
                "name": "Response Generation",
                "description": "Formats a prompt using the retrieved documents and streams the LLM response.",
                "inputs": ["retrieved_docs", "conversation_history", "user_input"],
                "outputs": ["final_response"],
            },
        ],
        "author": "Simon Enni",
        "version": "1.0.0",
        "tags": ["rag", "retrieval", "generation", "knowledge base", "openai"],
    }
