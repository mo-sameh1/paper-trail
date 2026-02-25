from typing import Any, Dict

from llm.model import LLMManager


class InferenceService:
    """
    Service class responsible for managing the execution of model inference tasks.
    Sits between the REST API handlers and the LLM manager logic.
    """

    def __init__(self, llm_manager: LLMManager | None = None) -> None:
        # In a real deployed state, the llm_manager would be injected here.
        self.llm_manager = llm_manager

    def extract_graph(self, text: str) -> Dict[str, Any]:
        """
        Processes text through the LLM out to standard entity and relationship schemas.
        Presently returns a mocked response structure showcasing the expected
        output graph format.
        """
        # (Mock implementation)
        # Future state: output = self.llm_manager.generate(NER_PROMPT + text)
        return {
            "entities": [
                {"id": "doc_1", "properties": {"type": "Document", "content": text}},
                {
                    "id": "org_cia",
                    "properties": {"type": "Organization", "name": "CIA"},
                },
            ],
            "relationships": [
                {
                    "source_id": "doc_1",
                    "target_id": "org_cia",
                    "type": "MENTIONS",
                    "properties": {"confidence": 0.95},
                }
            ],
        }
