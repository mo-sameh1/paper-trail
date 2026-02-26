class NERPrompts:
    """
    Centralized location for all Named Entity Recognition prompts.
    Provides standard templates without scattering string literals across code.
    """

    SYSTEM_INSTRUCTION = (
        "You are an expert at extracting entity relationships from text. "
        "You always respond with valid JSON only, no extra text."
    )

    # Standard set of prompts for qualitative baseline testing
    EVALUATION_SAMPLES = [
        "Extract named entities from this text: The CIA employed John Doe in 2022.",
        (
            "Generate a JSON graph of entities and relations from: "
            "OpenAI was founded by Sam Altman."
        ),
    ]

    RELATION_EXTRACTION_USER = (
        "Analyze the following document and extract all explicitly stated entity "
        "relationships. Return the output STRICTLY as valid JSON matching the "
        "following schema:\n"
        "{{\n"
        '  "relationships": [\n'
        '    {{"subject": "Entity1", "relation": "RelationType", '
        '"object": "Entity2", "confidence": 0.95}}\n'
        "  ]\n"
        "}}\n\n"
        "Document:\n"
        "{document_text}"
    )

    JUDGE_SYSTEM_INSTRUCTION = (
        "You are a strict evaluation judge for entity relationship extraction. "
        "You compare extracted triples against ground truth triples and determine "
        "matches. You always respond with valid JSON only, no extra text."
    )

    JUDGE_USER = (
        "Compare the EXTRACTED relationships against the GROUND TRUTH relationships "
        "for the same document. Two triples match if they refer to the same pair of "
        "real-world entities, even if the names are slightly different "
        "(e.g. \"Schneider\" matches \"Wilfried Schneider\"). "
        "Ignore the relation type — only judge whether the subject-object entity "
        "pair matches.\n\n"
        "EXTRACTED (subject, object):\n{extracted_triples}\n\n"
        "GROUND TRUTH (subject, object):\n{ground_truth_triples}\n\n"
        "Return STRICTLY valid JSON with this schema:\n"
        "{{\n"
        '  "true_positives": <int>,\n'
        '  "false_positives": <int>,\n'
        '  "false_negatives": <int>,\n'
        '  "matched_pairs": [\n'
        '    {{"extracted": ["subj", "obj"], "ground_truth": ["subj", "obj"]}}\n'
        "  ]\n"
        "}}"
    )

    @staticmethod
    def format_training_prompt(
        instruction: str, input_text: str, output_text: str
    ) -> str:
        """
        Formats a row of data into an instruction-following prompt string.
        """
        return (
            f"Instruction: {instruction}\n"
            f"Input: {input_text}\n"
            f"Output: {output_text}"
        )
