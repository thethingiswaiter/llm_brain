import yaml
import os
from typing import Dict, List, Any

class SkillManager:
    def __init__(self, skills_md_dir: str = "d:\\file\\vscode\\llm_brain\\skills_md"):
        self.skills_md_dir = skills_md_dir
        self.loaded_skills: List[Dict[str, Any]] = []
        if not os.path.exists(self.skills_md_dir):
            os.makedirs(self.skills_md_dir)

    def load_skill_md(self, filename: str) -> Optional[Dict[str, Any]]:
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
                        "keywords": metadata.get("keywords", []),
                        "description": metadata.get("description", ""),
                        "entry_node": metadata.get("entry_node", ""),
                        "body": parts[2].strip()
                    }
                    self.loaded_skills.append(skill)
                    return skill
                except yaml.YAMLError as exc:
                    print(f"Error parsing YAML front-matter in {filename}: {exc}")
        return None

    def find_best_skill(self, keywords: List[str]) -> Optional[Dict[str, Any]]:
        """Match skill strictly by keywords."""
        best_skill = None
        max_overlap = 0
        for skill in self.loaded_skills:
            overlap = len(set(keywords) & set(skill.get("keywords", [])))
            if overlap > max_overlap:
                max_overlap = overlap
                best_skill = skill
        return best_skill

    def assign_skill_to_task(self, task_description: str, extracted_keywords: List[str]):
        """A sub-task strictly gets at most 1 skill to mount."""
        return self.find_best_skill(extracted_keywords)
