import json
import re
from typing import Any, Dict, List, Set, Tuple

from llm.model import LLMManager
from llm.prompts import NERPrompts
from utils.logger import get_logger

logger = get_logger(__name__)


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
        """
        if not self.llm_manager:
            logger.info("InferenceService initializing LLMManager...")
            self.llm_manager = LLMManager(is_training=False)

        user_message = NERPrompts.RELATION_EXTRACTION_USER.format(document_text=text)
        prompt = self.llm_manager.format_chat_prompt(
            system_message=NERPrompts.SYSTEM_INSTRUCTION,
            user_message=user_message,
        )

        # Max tokens should be reasonably high to catch all output JSON
        response = self.llm_manager.generate(prompt, max_new_tokens=1024)
        logger.info(f"LLM response: {response[:300]}...")
        # Parse the JSON response
        return self._parse_json_response(response)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                parsed = json.loads(json_str)
                entities_map: Dict[str, Dict[str, Any]] = {}
                relationships = []

                for rel in parsed.get("relationships", []):
                    subj = str(rel.get("subject", "")).strip()
                    obj = str(rel.get("object", "")).strip()
                    rel_type = (
                        str(rel.get("relation", "RELATED_TO"))
                        .strip()
                        .upper()
                        .replace(" ", "_")
                    )
                    conf = rel.get("confidence", 1.0)

                    if not subj or not obj:
                        continue

                    subj_id = f"ent_{subj.lower().replace(' ', '_')}"
                    obj_id = f"ent_{obj.lower().replace(' ', '_')}"

                    # Deduplicate entities via dict keyed on id
                    if subj_id not in entities_map:
                        entities_map[subj_id] = {
                            "id": subj_id,
                            "properties": {"name": subj.lower()},
                        }
                    if obj_id not in entities_map:
                        entities_map[obj_id] = {
                            "id": obj_id,
                            "properties": {"name": obj.lower()},
                        }

                    relationships.append(
                        {
                            "source_id": subj_id,
                            "target_id": obj_id,
                            "type": rel_type if rel_type else "RELATED_TO",
                            "properties": {"confidence": conf},
                        }
                    )

                return {
                    "entities": list(entities_map.values()),
                    "relationships": relationships,
                }
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response.")

        return {"entities": [], "relationships": []}

    def judge_triples(
        self,
        extracted: Set[Tuple[str, str]],
        ground_truth: Set[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """
        Uses the LLM as a judge to compare extracted vs ground truth
        entity pairs with semantic understanding of name variations.
        Returns {"true_positives": int, "false_positives": int,
                 "false_negatives": int, "matched_pairs": [...]}.
        """
        if not self.llm_manager:
            logger.info("InferenceService initializing LLMManager...")
            self.llm_manager = LLMManager(is_training=False)

        ext_str = "\n".join(
            f"  ({s}, {o})" for s, o in sorted(extracted)
        )
        gt_str = "\n".join(
            f"  ({s}, {o})" for s, o in sorted(ground_truth)
        )

        user_message = NERPrompts.JUDGE_USER.format(
            extracted_triples=ext_str,
            ground_truth_triples=gt_str,
        )
        prompt = self.llm_manager.format_chat_prompt(
            system_message=NERPrompts.JUDGE_SYSTEM_INSTRUCTION,
            user_message=user_message,
        )

        response = self.llm_manager.generate(prompt, max_new_tokens=1024)
        logger.info(f"Judge response: {response[:300]}...")

        return self._parse_judge_response(
            response, len(extracted), len(ground_truth)
        )

    def _parse_judge_response(
        self, text: str, num_extracted: int, num_gt: int
    ) -> Dict[str, Any]:
        """Parse the judge LLM JSON response into tp/fp/fn counts."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                tp = int(parsed.get("true_positives", 0))
                fp = int(parsed.get("false_positives", num_extracted))
                fn = int(parsed.get("false_negatives", num_gt))
                # Sanity clamp: tp can't exceed either set size
                tp = min(tp, num_extracted, num_gt)
                fp = max(num_extracted - tp, 0)
                fn = max(num_gt - tp, 0)
                return {
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "matched_pairs": parsed.get("matched_pairs", []),
                }
            except (json.JSONDecodeError, ValueError):
                logger.error("Failed to parse judge JSON response.")

        # Fallback: assume no matches
        return {
            "true_positives": 0,
            "false_positives": num_extracted,
            "false_negatives": num_gt,
            "matched_pairs": [],
        }
