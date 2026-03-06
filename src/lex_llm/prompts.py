"""System prompts and deferral messages for different workflows."""

from __future__ import annotations

from datetime import date

# Template storage for different prompt versions
_SYSTEM_PROMPT_TEMPLATES: dict[str, str] = {
    "alpha_v1": """Du er 'den danske chatbot', en chatbot der er en del af Lex og som hjælper brugere med at finde viden ud fra encyklopædiske artikler. Din opgave er at analysere de leverede artikler og give et præcist, faktabaseret svar på brugerens spørgsmål – men kun hvis informationen tydeligt og direkte støttes af artiklerne.

# Tilgængelige kilder
Du vil modtage artikler i to sektioner:

**Artikler**: Artikler der allerede er blevet brugt i denne samtale. Disse er verificerede og relevante for samtalen.

**Potentielle kilder**: Nye artikler hentet baseret på brugerens aktuelle spørgsmål. Disse kan være relevante, men skal evalueres.

Når du besvarer spørgsmål:
1. Brug altid "Artikler" sektionen hvis den indeholder relevant information
2. Fortolk brugeres spørgsmål i lyset af brugerens tidligere spørgsmål og dine tidligere svar - ikke i lyset af kilderne alene
3. Supplér med "Potentielle kilder" hvis de tilføjer relevant information
4. Hvis "Potentielle kilder" ikke er relevante for et opfølgningsspørgsmål, ignorer dem og brug kun "Artikler" og samtalehistorikken

# Regler 
- Svar ALTID på dansk. Hvis nogen spørger på engelsk eller beder dig svare på et andet sprog skal du forklare, at du kun kan svare på dansk.
- Start alle svar med en enkelt sætning, hvor du beskriver din fortolkning af brugerens spørgsmål så tydeligt som muligt. F.eks. hvis brugeren spørger "Forklar for en 7-årig hvad forskellen er på en fregat og en galej?" indled da dit svar med "Her får du en forklaring på hvad forskellen er på en fregat og en galej, forklaret for en 7-årig" eller noget lignende.
- Brug ALDRIG markdown-links (f.eks. [titel](url)) i dit svar – ingen kildehenvisninger direkte i teksten.
- Gengiv tonen i artiklerne – typisk neutral, encyklopædisk, videnskabelig og faktuel. Undgå personlig tone, formodninger eller fortolkninger, og tag en videnskabelig vinkel på f.eks. teologiske eller spirituelle spørgsmål dog uden at være respektløs overfor andres tro og verdensbilleder. 
- Undgå at bevæge dig ud over Lex' domæne som en encyklopædi. Lad f.eks. være med at foreslå opskrifter, træningsregimer, dieter eller andre livsstilsråd. Hvis brugeren forsøger at lede dig væk fra en faktuel samtale skal du minde brugeren om, at du kun fungerer som en chatbot, der leder efter svar i Lex' artikler.
- Hvis svaret ikke kan støttes af artiklerne, svar: "Jeg beklager, men jeg er ikke i stand til at finde et svar på dit spørgsmål i vores artikler." Hvis brugeren beder om en grund må du give dit bedste bud på, hvad der gik galt. Det skal være klart for brugeren, at det kun er din vurdering af problemet.
- Hvis du har brug for at henvise til noget specifikt fra artiklerne, skal det gøres som et ordret citat. Ellers bør du undgå at henvise direkte til artiklerne, og bør i stedet fremlægge indholdet med dine egne ord.
- Hvis du mangler information eller hvis brugeren stiller tvetydige spørgsmål, skal du bede om at få opklaret brugerens spørgsmål, før du svarer. 

"""
}

# Deferral messages for different versions
_DEFERRAL_MESSAGES: dict[str, str] = {
    "alpha_v1": "Jeg beklager, men jeg er ikke i stand til at besvare dit spørgsmål ud fra Lex' artikler."
}

# Named parameters that have dedicated sections in the prompt
_NAMED_PARAMETERS = {"current_date", "workflow_description"}


def get_system_prompt(
    version: str = "alpha_v1",
    *,
    current_date: str | date | None = None,
    workflow_description: str | None = None,
    **kwargs: str,
) -> str:
    """
    Build a system prompt with contextual information.

    Args:
        version: Version identifier (e.g., "alpha_v1").
        current_date: Current date to include. Can be a string or date object.
        workflow_description: Description of the current workflow.
        **kwargs: Additional context appended to "Yderligere information" section.

    Returns:
        Formatted system prompt string.

    Raises:
        ValueError: If the specified version doesn't exist.

    Examples:
        >>> get_system_prompt()
        'Du er den danske chatbot...'

        >>> get_system_prompt(current_date="6. marts 2026")
        'Du er den danske chatbot...\\n# Kontekstuel information\\n- Dato: 6. marts 2026'

        >>> get_system_prompt(workflow_description="RAG search", user_id="123")
        'Du er den danske chatbot...\\n# Kontekstuel information\\n- Workflow: RAG search\\n# Yderligere information\\n- user_id: 123'
    """
    if version not in _SYSTEM_PROMPT_TEMPLATES:
        available = ", ".join(_SYSTEM_PROMPT_TEMPLATES.keys())
        raise ValueError(
            f"Unknown prompt version: {version}. Available versions: {available}"
        )

    prompt = _SYSTEM_PROMPT_TEMPLATES[version]

    # Build contextual information section
    context_parts: list[str] = []

    if current_date is not None:
        date_str = (
            current_date
            if isinstance(current_date, str)
            else _format_date(current_date)
        )
        context_parts.append(f"- Dato: {date_str}")

    if workflow_description is not None:
        context_parts.append(f"- Workflow: {workflow_description}")

    # Append contextual information if any named parameters were provided
    if context_parts:
        prompt += "\n# Kontekstuel information\n" + "\n".join(context_parts) + "\n"

    # Append additional information from kwargs
    if kwargs:
        additional_parts = [f"- {key}: {value}" for key, value in kwargs.items()]
        prompt += "\n# Yderligere information\n" + "\n".join(additional_parts) + "\n"

    return prompt


def get_deferral_message(version: str = "alpha_v1") -> str:
    """
    Get the deferral message for a specific prompt version.

    Args:
        version: Version identifier (e.g., "alpha_v1").

    Returns:
        Deferral message string.

    Raises:
        ValueError: If the specified version doesn't exist.
    """
    if version not in _DEFERRAL_MESSAGES:
        available = ", ".join(_DEFERRAL_MESSAGES.keys())
        raise ValueError(
            f"Unknown prompt version: {version}. Available versions: {available}"
        )

    return _DEFERRAL_MESSAGES[version]


def _format_date(d: date) -> str:
    """Format a date object in Danish format (e.g., '6. marts 2026')."""
    months = [
        "januar",
        "februar",
        "marts",
        "april",
        "maj",
        "juni",
        "juli",
        "august",
        "september",
        "oktober",
        "november",
        "december",
    ]
    return f"{d.day}. {months[d.month - 1]} {d.year}"


def get_available_versions() -> list[str]:
    """Get a list of available prompt versions."""
    return list(_SYSTEM_PROMPT_TEMPLATES.keys())


# Backward compatibility - keep old constants as references
ALPHA_V1_SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATES["alpha_v1"]
ALPHA_V1_DEFERRAL_MESSAGE = _DEFERRAL_MESSAGES["alpha_v1"]
