from typing import List, Dict, Optional
from core.llm.manager import llm_manager
from cognitive.structured_output import parse_json_object, StructuredOutputError


DEFAULT_DOMAIN_LABEL = "综合"


DOMAIN_TREE = [
    "马列",
    "哲学",
    "宗教",
    "社科",
    "政治",
    "法律",
    "军事",
    "经济",
    "文化",
    "科学",
    "教育",
    "体育",
    "语言",
    "文字",
    "中国文学",
    "世界文学",
    "音乐",
    "美术",
    "设计",
    "戏剧",
    "影视",
    "历史",
    "地理",
    "旅游",
    "自然科学",
    "数学",
    "物理",
    "化学",
    "天文",
    "地学",
    "生物",
    "医学",
    "卫生",
    "农业",
    "计算机",
    "电子",
    "通信",
    "自动化",
    "控制",
    "机械",
    "仪器",
    "建筑",
    "土木",
    "轻工",
    "生活服务",
    "交通",
    "航空",
    "航天",
    "环境",
    "安全",
    "综合",
]


DOMAIN_ALIASES = {
    "综合性图书": DEFAULT_DOMAIN_LABEL,
    "综合": DEFAULT_DOMAIN_LABEL,
    "通用": DEFAULT_DOMAIN_LABEL,
    "其他": DEFAULT_DOMAIN_LABEL,
}

class CognitiveSystem:
    def __init__(self):
        self.domain_tree = list(DOMAIN_TREE)

    def extract_keywords(self, text: str, limit: int = 5) -> List[str]:
        source_text = str(text or "").strip()
        if not source_text:
            return []

        normalized_limit = max(1, min(int(limit or 5), 10))
        prompt = f"""
        你是一个中文优先的关键词提取助手。
        请从下面这句话里提取最重要的 {normalized_limit} 个关键词或关键短语，并按重要性排序。

        要求:
        1. 优先返回有语义的词或短语，不要拆成无意义的字片段。
        2. 可以不完全照抄原句，允许使用更精炼但不改变原意的表达。
        3. 如果文件名、路径范围、环境范围是关键信息，可以保留。
        4. 只返回 JSON，不要解释。

        返回格式:
        {{
            "keywords": ["词1", "词2", "词3"]
        }}

        文本:
        {source_text}
        """
        try:
            response = llm_manager.invoke(prompt, source="feature_extractor.extract_keywords")
            data = parse_json_object(response.content, required_fields={"keywords": list})
            keywords = []
            seen = set()
            for item in data.get("keywords", []):
                value = str(item).strip()
                marker = value.lower()
                if not value or marker in seen:
                    continue
                seen.add(marker)
                keywords.append(value)
            return keywords[:normalized_limit]
        except StructuredOutputError as e:
            llm_manager.log_event(f"Keyword extraction structured parsing failed: {e}", level=40)
        except Exception as e:
            llm_manager.log_event(f"Keyword extraction failed: {e}", level=40)
        return []

    def rewrite_intent(self, text: str) -> str:
        """Rewrite user intent into a concise, style-normalized expression without adding new facts."""
        source_text = str(text or "").strip()
        if not source_text:
            return ""

        prompt = f"""
        你是一个中文优先的意图重述器。请将用户输入重写为更统一、简洁的表达。
        目标:
        1. 去掉语气助词、口头禅和风格噪声。
        2. 消除不同用户语言习惯差异，保留核心意图不变。
        3. 不添加原文没有的新信息。
        4. 原文中的前置条件、作用域限定、环境约束必须保留，不能省略或泛化。
        5. 像“当前目录/当前工作区/当前文件/当前分支/当前环境/已选中内容”这类条件属于核心意图的一部分，必须明确保留。
        6. 只返回重写后的单行文本，不要解释。
        用户输入:
        {source_text}
        """
        try:
            response = llm_manager.invoke(prompt, source="feature_extractor.rewrite_intent")
            rewritten = str(response.content or "").strip()
            return rewritten or source_text
        except Exception as e:
            llm_manager.log_event(f"Intent rewriting failed: {e}", level=40)
            return source_text

    def normalize_domain_label(self, value: str) -> str:
        normalized = str(value or "").strip().strip('"').strip("'")
        if normalized in self.domain_tree:
            return normalized
        if normalized in DOMAIN_ALIASES:
            return DOMAIN_ALIASES[normalized]

        compact = normalized.replace(" ", "")
        for domain in self.domain_tree:
            if compact == domain.replace(" ", ""):
                return domain
        return DEFAULT_DOMAIN_LABEL

    def extract_features(self, text: str, domain_hint: str = "") -> tuple[List[str], str]:
        """
        Extracted features: 3-5 keywords for subtask, up to 30 for global task.
        And a one-sentence compression description.
        """
        prompt = f"""
        你是一个中文优先的任务特征提取器。请在没有额外上下文的情况下，从下面文本中提取最重要的信息。
        领域提示: {domain_hint}
        要求:
        1. 提取 3 到 5 个关键词，优先保留中文关键词。
        2. 用一句中文总结核心内容。
        3. 只返回 JSON，不要输出解释或 Markdown 代码块。
        请严格按以下 JSON 格式返回:
        {{
            "keywords": ["kw1", "kw2", "kw3"],
            "summary": "一句话总结。"
        }}
        文本:
        {text}
        """
        try:
            response = llm_manager.invoke(prompt, source="feature_extractor.extract_features")
            data = parse_json_object(
                response.content,
                required_fields={"keywords": list, "summary": str},
            )
            keywords = [str(item).strip() for item in data.get("keywords", []) if str(item).strip()]
            return keywords[:30], data.get("summary", "").strip()
        except StructuredOutputError as e:
            llm_manager.log_event(f"Feature extraction structured parsing failed: {e}", level=40)
            return [], text[:50] + "..."
        except Exception as e:
            llm_manager.log_event(f"Feature extraction failed: {e}", level=40)
            return [], text[:50] + "..."

    def determine_domain(self, text: str) -> str:
        """Assign an appropriate domain label from the built-in domain tree."""
        prompt = f"""
        请将下面文本归类到以下领域中的一个。
        只允许返回领域名称本身，不要输出其他解释。
        可选领域:
        {', '.join(self.domain_tree)}
        文本:
        {text}
        """
        try:
            response = llm_manager.invoke(prompt, source="feature_extractor.determine_domain")
            return self.normalize_domain_label(response.content)
        except Exception:
            return DEFAULT_DOMAIN_LABEL
