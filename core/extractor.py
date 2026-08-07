import os
import json
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMExtractor:
    def __init__(self) -> None:
        self.model = "gpt-4o"
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        else:
            logger.warning(
                "OPENAI_API_KEY not set — LLM extraction disabled. "
                "Reddit posts will be skipped."
            )
            self.client = None

    def extract(self, timeline_json: str):
        if self.client is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Cannot perform LLM extraction."
            )

        from core.schemas import ExtractedClinicalProfile

        system_prompt = (
            "You are a clinical information extraction engine.\n"
            "Your only task is to transform the received text into a JSON object "
            "that strictly conforms to the provided schema.\n"
            "Rules:\n"
            "- Extract ONLY information explicitly present in the text.\n"
            "- Do NOT invent data or make clinical inferences.\n"
            "- For missing values use null; for missing lists use [].\n"
            "- episode_id: generate a short unique slug from the text (e.g. 'ep-breast-cancer-2023').\n"
            "- user_id: use 'anonymous' if no username is present.\n"
            "- baseline_status.value must be one of: assente, lieve, moderata, elevata, non_deducibile.\n"
            "- post_treatment_status.value must be one of: tornata_come_prima, inferiore_a_prima, "
            "superiore_a_prima, non_stabilizzata, non_deducibile.\n"
            "- supporting_quotes: include verbatim excerpts from the text with their source.\n"
            "- Return ONLY a valid JSON object matching the schema."
        )

        schema = ExtractedClinicalProfile.model_json_schema()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Timeline:\n{timeline_json}\n\n"
                        f"Schema JSON:\n{json.dumps(schema, ensure_ascii=False)}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        from core.schemas import ExtractedClinicalProfile
        return ExtractedClinicalProfile.model_validate(data)
