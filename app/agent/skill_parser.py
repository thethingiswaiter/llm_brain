import yaml
import os
import re
from typing import Dict, List, Any, Optional
from core.config import config
from core.llm.manager import llm_manager


class SkillManager:
    STOPWORDS = {
        "a", "an", "the", "and", "for", "with", "that", "this", "from", "into", "your", "have",
        "in", "on", "at", "to", "of", "is", "are", "me", "my", "our", "please", "current",
        "will", "would", "could", "should", "about", "there", "their", "then", "than",
        "when", "where", "which", "what", "tell", "give", "show", "user", "name", "tool",
        "请", "请问", "帮我", "一下", "看看", "看下", "查下", "查询一下", "告诉我", "我想", "我要", "有没有", "怎么", "如何",
    }
    TOOL_KEYWORD_HINTS = {
        "list_directory": [
            "目录",
            "当前目录",
            "工作区",
            "文件",
            "文件夹",
            "列表",
            "列出",
            "统计",
            "数量",
            "总数",
            "个数",
        ],
        "read_text_file": [
            "文件",
            "读取",
            "打开文件",
            "查看内容",
            "文本",
            "行",
        ],
        "grep_text": [
            "搜索",
            "查找",
            "匹配",
            "关键字",
            "文本搜索",
        ],
    }
    TOOL_EXECUTION_MARKERS = {
        "查", "查询", "列出", "统计", "读取", "读", "搜索", "查找", "写入", "保存", "删除", "执行",
        "目录", "文件", "路径", "终端", "命令", "workspace", "目录下", "文件夹",
        "list", "count", "read", "write", "search", "grep", "directory", "file", "path",
    }

    def __init__(self, skill_dir: str = None):
        self.skill_dir = config.resolve_path(skill_dir or config.skill_dir)
        self.loaded_skills: List[Dict[str, Any]] = []
        self.loaded_tool_skills: Dict[str, Dict[str, Any]] = {}
        self.loaded_skill_files: set[str] = set()
        if not os.path.exists(self.skill_dir):
            os.makedirs(self.skill_dir)
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
        raw_tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", prepared_text)
        expanded_tokens: list[str] = []
        for token in raw_tokens:
            if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
                expanded_tokens.append(token)
                for size in (2, 3):
                    if len(token) <= size:
                        continue
                    for index in range(len(token) - size + 1):
                        expanded_tokens.append(token[index:index + size])
            else:
                expanded_tokens.append(token)

        seen = set()
        normalized_tokens = []
        for token in expanded_tokens:
            if len(token) <= 1 or token in self.STOPWORDS or token in seen:
                continue
            seen.add(token)
            normalized_tokens.append(token)
        return normalized_tokens

    def _match_ratio(self, overlap: int, task_terms: set[str]) -> float:
        if not task_terms:
            return 0.0
        return overlap / len(task_terms)

    def _passes_threshold(self, overlap: int, task_terms: set[str], min_overlap: int, min_ratio: float) -> bool:
        if overlap < min_overlap:
            return False
        return self._match_ratio(overlap, task_terms) >= min_ratio

    def _build_route_reason(self, candidate_name: str, matched_terms: list[str], overlap: int, match_ratio: float) -> str:
        matched_text = ", ".join(matched_terms) if matched_terms else "none"
        return (
            f"matched {overlap} term(s) [{matched_text}] "
            f"with ratio={match_ratio:.2f} for candidate={candidate_name}"
        )

    def _is_tool_execution_task(self, task_description: str, extracted_keywords: List[str]) -> bool:
        task_terms = set(self._tokenize_text(task_description))
        keyword_terms = set(self._normalize_keywords(extracted_keywords))
        combined_terms = task_terms | keyword_terms
        if not combined_terms:
            return False
        return any(
            marker in combined_terms or any(marker in term for term in combined_terms)
            for marker in self.TOOL_EXECUTION_MARKERS
        )

    def _best_term_match(
        self,
        candidate_term_sets: List[set[str]],
        tool_terms: set[str],
    ) -> tuple[list[str], int, float, set[str]]:
        best_matched_terms: list[str] = []
        best_overlap = 0
        best_ratio = 0.0
        best_task_terms: set[str] = set()

        for task_terms in candidate_term_sets:
            if not task_terms:
                continue
            matched_terms = sorted(task_terms & tool_terms)
            overlap = len(matched_terms)
            match_ratio = self._match_ratio(overlap, task_terms)
            score = (match_ratio, overlap, -len(task_terms))
            best_score = (best_ratio, best_overlap, -len(best_task_terms))
            if score > best_score:
                best_matched_terms = matched_terms
                best_overlap = overlap
                best_ratio = match_ratio
                best_task_terms = task_terms

        return best_matched_terms, best_overlap, best_ratio, best_task_terms

    def refresh_skills(self) -> int:
        self.loaded_skills = []
        self.loaded_skill_files = set()
        for filename in sorted(os.listdir(self.skill_dir)):
            if filename.endswith(".md"):
                self.load_skill(filename)
        return len(self.loaded_skills)

    def _evict_skill_cache(self, filename: str):
        self.loaded_skills = [
            skill for skill in self.loaded_skills
            if skill.get("source_file") != filename
        ]
        self.loaded_skill_files.discard(filename)

    def load_skill(self, filename: str, force_reload: bool = False) -> Optional[Dict[str, Any]]:
        filename = os.path.basename(filename)
        if force_reload:
            self._evict_skill_cache(filename)

        if filename in self.loaded_skill_files:
            for skill in self.loaded_skills:
                if skill.get("source_file") == filename:
                    return skill

        path = os.path.join(self.skill_dir, filename)
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
                    llm_manager.log_event(
                        f"Markdown skill parse failed | file={filename} | error={exc}",
                        level=40,
                    )
        return None

    def register_tool(self, tool, source_type: str = "python", source_file: str = "") -> Optional[Dict[str, Any]]:
        tool_name = getattr(tool, "name", "")
        if not tool_name:
            return None
        if tool_name in self.loaded_tool_skills:
            return self.loaded_tool_skills[tool_name]

        description = getattr(tool, "description", "") or ""
        keywords = self._normalize_keywords(self._tokenize_text(f"{tool_name} {description}"))
        keywords = self._normalize_keywords(keywords + self.TOOL_KEYWORD_HINTS.get(tool_name, []))
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

    def unregister_tools(self, tool_names: List[str]) -> int:
        removed = 0
        for tool_name in tool_names:
            if tool_name in self.loaded_tool_skills:
                self.loaded_tool_skills.pop(tool_name, None)
                removed += 1
        return removed

    def find_best_skill(self, keywords: List[str]) -> Optional[Dict[str, Any]]:
        """Match markdown prompt skill by keywords."""
        best_skill = None
        best_score = (-1, -1.0, -1)
        normalized_keywords = set(self._normalize_keywords(keywords))
        for skill in self.loaded_skills:
            skill_terms = set(skill.get("keywords", []))
            matched_terms = sorted(normalized_keywords & skill_terms)
            overlap = len(matched_terms)
            if not self._passes_threshold(
                overlap,
                normalized_keywords,
                config.prompt_skill_min_overlap,
                config.prompt_skill_min_match_ratio,
            ):
                continue
            match_ratio = self._match_ratio(overlap, normalized_keywords)
            score = (overlap, match_ratio, len(skill_terms))
            if score > best_score:
                best_score = score
                best_skill = {
                    **skill,
                    "matched_terms": matched_terms,
                    "overlap_count": overlap,
                    "match_ratio": match_ratio,
                    "route_reason": self._build_route_reason(skill.get("name", "unknown_skill"), matched_terms, overlap, match_ratio),
                }
        return best_skill

    def _rank_tools(
        self,
        candidate_term_sets: List[set[str]],
        min_overlap: int,
        min_ratio: float,
        limit: int,
    ) -> List[Dict[str, Any]]:
        ranked_tools = []
        for tool_skill in self.loaded_tool_skills.values():
            tool_terms = set(tool_skill.get("keywords", []))
            matched_terms, overlap, match_ratio, task_terms = self._best_term_match(candidate_term_sets, tool_terms)
            if not self._passes_threshold(overlap, task_terms, min_overlap, min_ratio):
                continue
            ranked_tools.append(
                (
                    overlap,
                    match_ratio,
                    len(tool_terms),
                    {
                        **tool_skill,
                        "matched_terms": matched_terms,
                        "overlap_count": overlap,
                        "match_ratio": match_ratio,
                        "route_reason": self._build_route_reason(
                            tool_skill.get("name", "unknown_tool"), matched_terms, overlap, match_ratio
                        ),
                    },
                )
            )

        ranked_tools.sort(key=lambda item: (item[0], item[1], item[2], item[3]["name"]), reverse=True)
        return [item[3] for item in ranked_tools[:limit]]

    def find_relevant_tools(self, task_description: str, extracted_keywords: List[str], limit: int = 3) -> List[Dict[str, Any]]:
        keyword_terms = set(self._normalize_keywords(extracted_keywords))
        description_terms = set(self._tokenize_text(task_description))
        candidate_term_sets = []
        if keyword_terms:
            candidate_term_sets.append(keyword_terms)
        if description_terms and description_terms not in candidate_term_sets:
            candidate_term_sets.append(description_terms)

        if not candidate_term_sets:
            return []
        strict_matches = self._rank_tools(
            candidate_term_sets,
            config.tool_skill_min_overlap,
            config.tool_skill_min_match_ratio,
            limit,
        )
        if strict_matches:
            return strict_matches

        if not self._is_tool_execution_task(task_description, extracted_keywords):
            return []

        relaxed_matches = self._rank_tools(
            candidate_term_sets,
            min_overlap=1,
            min_ratio=0.0,
            limit=limit,
        )
        for item in relaxed_matches:
            item["route_reason"] = f"{item.get('route_reason', '')} | relaxed_tool_threshold"
        return relaxed_matches

    def assign_capabilities_to_task(self, task_description: str, extracted_keywords: List[str]) -> Dict[str, Any]:
        prompt_skill = self.find_best_skill(extracted_keywords)
        tool_skills = self.find_relevant_tools(task_description, extracted_keywords)
        prompt_skill_reason = prompt_skill.get("route_reason", "") if prompt_skill else ""
        if tool_skills and self._is_tool_execution_task(task_description, extracted_keywords):
            prompt_skill = None
            prompt_skill_reason = "suppressed_in_tool_execution_mode"
        return {
            "prompt_skill": prompt_skill,
            "prompt_skill_reason": prompt_skill_reason,
            "tool_skills": tool_skills,
            "tools": [item["tool"] for item in tool_skills],
            "tool_reasons": [item.get("route_reason", "") for item in tool_skills],
        }
