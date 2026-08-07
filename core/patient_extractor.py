"""
LLM extractor for patient-reported experiences (Reddit posts and similar).
Uses PatientExperienceProfile schema; extracts the personal health journey
rather than a clinical/scientific profile.
"""
import json
import logging
import os

logger = logging.getLogger(__name__)


class PatientExperienceExtractor:
    MODEL = "gpt-4o"

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
            logger.warning("OPENAI_API_KEY not set — PatientExperienceExtractor disabled.")

    def extract(self, title: str, text: str, author: str = "", url: str = "") -> dict:
        """
        Extract a structured PatientExperienceProfile from a Reddit post.

        Args:
            title: Post title.
            text:  Post body text.
            author: Reddit username (anonymised in output — not stored in profile).
            url:   Source URL (for reference only).

        Returns:
            dict conforming to PatientExperienceProfile.
        Raises:
            RuntimeError if API key not set.
        """
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        from core.patient_schema import PatientExperienceProfile

        schema = PatientExperienceProfile.model_json_schema()

        combined = f"Title: {title}\n\n{text or '(no body text)'}".strip()
        # Limit input to ~3 000 chars to keep cost reasonable
        if len(combined) > 3000:
            combined = combined[:2970] + "\n…[truncated]"

        combined += f"\n\nPost length (chars): {len(text or '')}"

        system_prompt = (
            "You are a medical anthropologist analyzing patient-reported health experiences.\n"
            "Extract a structured profile from the Reddit post below.\n\n"
            "RULES:\n"
            "- Extract ONLY what is explicitly stated. Never infer or embellish.\n"
            "- 'key_quotes' must be verbatim phrases copied from the post (max 3, each ≤120 chars).\n"
            "- 'experience_summary' must be 1-2 neutral sentences summarising the post.\n"
            "- For 'extraction_confidence', use 'high' if the post clearly describes a personal "
            "health journey, 'medium' if partial, 'low' if off-topic or ambiguous.\n"
            "- Do NOT store the author's username or any PII beyond age/sex/country if stated.\n"
            "- Use null for missing scalars; [] for missing lists.\n"
            "- Return ONLY a valid JSON object conforming strictly to the schema."
        )

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system",  "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{combined}\n\n"
                        f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        raw = json.loads(response.choices[0].message.content)

        # Fill code-side field
        raw["post_length_chars"] = len(text or "")

        # Validate shape
        profile = PatientExperienceProfile.model_validate(raw)
        return profile.model_dump()
