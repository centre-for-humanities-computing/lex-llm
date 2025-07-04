import importlib
import os
from typing import Dict, Any, List
import types

WORKFLOWS_PATH = os.path.dirname(__file__).replace("api", "workflows")


def list_workflow_modules() -> List[str]:
    """List all available workflow module names (without .py) in the workflows folder."""
    return [
        name[:-3]
        for name in os.listdir(WORKFLOWS_PATH)
        if name.endswith(".py") and not name.startswith("__")
    ]


def get_workflow_module(workflow_id: str) -> types.ModuleType:
    """Dynamically import a workflow module by id."""
    try:
        return importlib.import_module(f"src.lex_llm.workflows.{workflow_id}")
    except ModuleNotFoundError:
        raise ImportError(
            f"Workflow module '{workflow_id}' not found. Available workflows: {list_workflow_modules()}"
        )


def get_all_workflow_metadata() -> List[Dict[str, Any]]:
    """Return metadata for all workflows."""
    metadata = []
    for workflow_id in list_workflow_modules():
        mod = get_workflow_module(workflow_id)
        if mod and hasattr(mod, "get_metadata"):
            metadata.append(mod.get_metadata())
    return metadata
