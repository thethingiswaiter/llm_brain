import yaml
import os
import re
from typing import Dict, List, Any, Optional
from config import config

class SkillManager:
    STOPWORDS = {
        "a", "an", "the", "and", "for", "with", "that", "this", "from", "into", "your", "have",
        "in", "on", "at", "to", "of", "is", "are", "me", "my", "our", "please", "current",
        "will", "would", "could", "should", "about", "there", "their", "then", "than",
        "when", "where", "which", "what", "tell", "give", "show", "user", "name", "tool",
    }

    def __init__(self, skills_md_dir: str = None):
        self.skills_md_dir = config.resolve_path(skills_md_dir or config.skills_md_dir)
        self.loaded_skills: List[Dict[str, Any]] = []
        self.loaded_tool_skills: Dict[str, Dict[str, Any]] = {}
        self.loaded_skill_files: set[str] = set()
        if not os.path.exists(self.skills_md_dir):
            os.makedirs(self.skills_md_dir)
        self.refresh_skills()

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        normalized = []
        for keyword in keywords:
            if not isinstance(keyword, str):
                continue
            value = keyword.strip().lower()
            if value and value not in self.STOPWORDS:
                normalized.append(value)
        return normalized

    def _tokenize_text(self, text: str) -> List[str]:
        prepared_text = text.replace("_", " ").replace("-", " ").lower()
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", prepared_text)
        return [token for token in tokens if len(token) > 1 and token not in self.STOPWORDS]

    def _match_ratio(self, overlap: int, task_terms: set[str]) -> float:
        if not task_terms:
            return 0.0
        return overlap / len(task_terms)

    def _passes_threshold(self, overlap: int, task_terms: set[str], min_overlap: int, min_ratio: float) -> bool:
        if overlap < min_overlap:
            return False
        return self._match_ratio(overlap, task_terms) >= min_ratio

    def refresh_skills(self) -> int:
        self.loaded_skills = []
        self.loaded_skill_files = set()
        for filename in sorted(os.listdir(self.skills_md_dir)):
            if filename.endswith(".md"):
                self.load_skill_md(filename)
        return len(self.loaded_skills)

    def _evict_skill_cache(self, filename: str):
        self.loaded_skills = [
            skill for skill in self.loaded_skills
            if skill.get("source_file") != filename
        ]
        self.loaded_skill_files.discard(filename)

    def load_skill_md(self, filename: str, force_reload: bool = False) -> Optional[Dict[str, Any]]:
        filename = os.path.basename(filename)
        if force_reload:
            self._evict_skill_cache(filename)

        if filename in self.loaded_skill_files:
            for skill in self.loaded_skills:
                if skill.get("source_file") == filename:
                    return skill

        path = os.path.join(self.skills_md_dir, filename)
        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse YAML Front Matter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                yaml_content = parts[1].strip()
                try:
                    metadata = yaml.safe_load(yaml_content)
                    skill = {
                        "name": metadata.get("name", "Unknown Skill"),
                        "confidence": metadata.get("confidence", 50),
                        "keywords": self._normalize_keywords(metadata.get("keywords", [])),
                        "description": metadata.get("description", ""),
                        "entry_node": metadata.get("entry_node", ""),
                        "body": parts[2].strip(),
                        "source_file": filename,
                        "skill_type": "markdown",
                    }
                    self.loaded_skills.append(skill)
                    self.loaded_skill_files.add(filename)
                    return skill
                except yaml.YAMLError as exc:
                    print(f"Error parsing YAML front-matter in {filename}: {exc}")
        return None

    def register_tool(self, tool, source_type: str = "python", source_file: str = "") -> Optional[Dict[str, Any]]:
        tool_name = getattr(tool, "name", "")
        if not tool_name:
            return None
        if tool_name in self.loaded_tool_skills:
            return self.loaded_tool_skills[tool_name]

        description = getattr(tool, "description", "") or ""
        keywords = self._normalize_keywords(self._tokenize_text(f"{tool_name} {description}"))
        tool_skill = {
            "name": tool_name,
            "description": description,
            "keywords": keywords,
            "tool": tool,
            "source_type": source_type,
            "source_file": source_file,
            "skill_type": "tool",
        }
        self.loaded_tool_skills[tool_name] = tool_skill
        return tool_skill

    def register_tools(self, tools: List[Any], source_type: str = "python", source_file: str = "") -> List[Dict[str, Any]]:
        registered = []
        for tool in tools:
            tool_skill = self.register_tool(tool, source_type=source_type, source_file=source_file)
            if tool_skill:
                registered.append(tool_skill)
        return registered

    def find_best_skill(self, keywords: List[str]) -> Optional[Dict[str, Any]]:
        """Match markdown prompt skill by keywords."""
        best_skill = None
        max_overlap = 0
        normalized_keywords = set(self._normalize_keywords(keywords))
        for skill in self.loaded_skills:
            overlap = len(normalized_keywords & set(skill.get("keywords", [])))
            if not self._passes_threshold(
                overlap,
                normalized_keywords,
                config.prompt_skill_min_overlap,
                config.prompt_skill_min_match_ratio,
            ):
                continue
            if overlap > max_overlap:
                max_overlap = overlap
                best_skill = skill
        return best_skill

    def find_relevant_tools(self, task_description: str, extracted_keywords: List[str], limit: int = 3) -> List[Dict[str, Any]]:
        task_terms = set(self._normalize_keywords(extracted_keywords) + self._tokenize_text(task_description))
        if not task_terms:
            return []

        ranked_tools = []
        for tool_skill in self.loaded_tool_skills.values():
            tool_terms = set(tool_skill.get("keywords", []))
            overlap = len(task_terms & tool_terms)
            if not self._passes_threshold(
                overlap,
                task_terms,
                config.tool_skill_min_overlap,
                config.tool_skill_min_match_ratio,
            ):
                continue
            match_ratio = self._match_ratio(overlap, task_terms)
            ranked_tools.append((overlap, match_ratio, len(tool_terms), tool_skill))

        ranked_tools.sort(key=lambda item: (item[0], item[1], item[2], item[3]["name"]), reverse=True)
        return [item[3] for item in ranked_tools[:limit]]

    def assign_capabilities_to_task(self, task_description: str, extracted_keywords: List[str]) -> Dict[str, Any]:
        prompt_skill = self.find_best_skill(extracted_keywords)
        tool_skills = self.find_relevant_tools(task_description, extracted_keywords)
        return {
            "prompt_skill": prompt_skill,
            "tool_skills": tool_skills,
            "tools": [item["tool"] for item in tool_skills],
        }

    def assign_skill_to_task(self, task_description: str, extracted_keywords: List[str]):
        """Backwards-compatible wrapper for prompt skill selection."""
        capabilities = self.assign_capabilities_to_task(task_description, extracted_keywords)
        return capabilities.get("prompt_skill")
