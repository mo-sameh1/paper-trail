class NERPrompts:
    """
    Centralized location for all Named Entity Recognition prompts.
    Provides standard templates without scattering string literals across code.
    """

    SYSTEM_INSTRUCTION = (
        "Extract the named entities (PERSON, ORGANIZATION, LOCATION) "
        "from the following legal text."
    )

    # Standard set of prompts for qualitative baseline testing
    EVALUATION_SAMPLES = [
        "Extract named entities from this text: The CIA employed John Doe in 2022.",
        (
            "Generate a JSON graph of entities and relations from: "
            "OpenAI was founded by Sam Altman."
        ),
    ]

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
