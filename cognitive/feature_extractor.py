from typing import List, Dict, Optional
import json
from llm_manager import llm_manager

class CognitiveSystem:
    def __init__(self):
        self.domain_tree = [
            "Technology/Software", "Technology/Hardware", "Science/Math", "Science/Physics",
            "Art/Literature", "Art/Music", "Daily Life/Food", "Daily Life/Travel", "Other"
        ]

    def extract_features(self, text: str, domain_hint: str = "") -> tuple[List[str], str]:
        """
        Extracted features: 3-5 keywords for subtask, up to 30 for global task.
        And a one-sentence compression description.
        """
        prompt = f"""
        Extract the most important features from the following text without prior context.
        Domain Hint: {domain_hint}
        Requirements:
        1. Extract 3 to 5 keywords.
        2. Write a single sentence summarizing the core content.
        Respond STRICTLY in JSON format:
        {{
            "keywords": ["kw1", "kw2", "kw3"],
            "summary": "One sentence summary."
        }}
        Text:
        {text}
        """
        try:
            response = llm_manager.invoke(prompt, source="feature_extractor.extract_features")
            data = json.loads(response.content)
            return data.get("keywords", []), data.get("summary", "")
        except Exception as e:
            llm_manager.log_event(f"Feature extraction failed: {e}", level=40)
            return [], text[:50] + "..."

    def determine_domain(self, text: str) -> str:
        """Assign an appropriate domain label from the built-in domain tree."""
        prompt = f"""
        Categorize the following text into ONE of these domains:
        {', '.join(self.domain_tree)}
        Reply with exactly the domain name.
        Text:
        {text}
        """
        try:
            response = llm_manager.invoke(prompt, source="feature_extractor.determine_domain")
            chosen = response.content.strip()
            if chosen in self.domain_tree:
                return chosen
            return "Other"
        except Exception:
            return "Other"
