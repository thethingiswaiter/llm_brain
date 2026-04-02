from typing import List, Dict, Any
from core.llm.manager import llm_manager
from cognitive.structured_output import parse_json_array, StructuredOutputError, StructuredOutputSchemaError

class TaskPlanner:
    DIRECT_LOOKUP_MARKERS = (
        "查询",
        "查看",
        "获取",
        "显示",
        "多少",
        "几个",
        "数量",
        "总数",
        "统计",
        "告诉我",
        "show me",
        "how many",
        "count",
        "total",
        "what is",
        "what's",
        "lookup",
        "look up",
        "get ",
        "find ",
    )
    SEQUENTIAL_MARKERS = (
        "然后",
        "再",
        "并且",
        "同时",
        "先",
        "after",
        "before",
        "then",
        "and then",
    )
    META_SUBTASK_MARKERS = (
        "identify",
        "determine",
        "use",
        "run",
        "parse",
        "extract",
        "verify",
        "validate",
        "return",
        "display",
        "collect",
        "识别",
        "确定",
        "使用",
        "运行",
        "解析",
        "提取",
        "验证",
        "返回",
        "显示",
        "收集",
    )

    def _single_task_plan(self, user_query: str) -> List[Dict[str, Any]]:
        description = str(user_query).strip() or "完成用户请求。"
        return [{
            "id": 1,
            "description": description,
            "expected_outcome": (
                "直接给出用户需要的结果。"
                "只有在关键信息确实缺失时，才提出简洁的澄清问题。"
            ),
        }]

    def _looks_like_direct_lookup(self, user_query: str) -> bool:
        text = " ".join(str(user_query or "").strip().lower().split())
        if not text:
            return False
        if any(marker in text for marker in self.SEQUENTIAL_MARKERS):
            return False
        if len(text) > 80 and len(text.split()) > 12:
            return False
        return any(marker in text for marker in self.DIRECT_LOOKUP_MARKERS)

    def _looks_over_decomposed(self, subtasks: List[Dict[str, Any]]) -> bool:
        if len(subtasks) <= 1:
            return False

        descriptions = [str(item.get("description", "")).strip().lower() for item in subtasks]
        if not all(descriptions):
            return False
        return all(any(marker in description for marker in self.META_SUBTASK_MARKERS) for description in descriptions)

    def split_task(self, user_query: str, thinking_mode: bool = False) -> List[Dict[str, Any]]:
        """
        根据核心需求规范，将复杂任务拆解成细粒度的“子任务”。
        要求：输出为JSON数组，每个子任务包含描述、预期结果等。
        """
        prompt = f"""
        你是一个中文优先的任务规划代理。请把下面的用户任务拆解成一组可执行、细粒度的子任务。
        要求:
        1. 每个子任务都应尽量独立，并且最好只需要一个工具或一种技能即可完成。
        2. 为每个子任务给出预期结果，用于后续校验。
        3. 如果用户请求本身就是直接查询、单条命令或单个事实问题，必须保持为恰好一个子任务，并保留原始意图。
        4. 除非用户明确要求多步骤流程，否则不要把简单请求拆成“识别/使用/解析/验证/返回”这类元步骤。
        5. 只返回 JSON，不要输出解释或 Markdown 代码块。
        
        请严格按以下 JSON 数组格式返回:
        [
            {{
                "id": 1,
                "description": "要执行的具体动作",
                "expected_outcome": "成功后应得到的结果"
            }}
        ]
        
        待拆解任务:
        {user_query}
        """
        try:
            response = llm_manager.invoke(prompt, source="planner.split_task")
            subtasks = parse_json_array(response.content)
            normalized_subtasks = []
            for index, item in enumerate(subtasks, start=1):
                if not isinstance(item, dict):
                    raise StructuredOutputSchemaError(f"Subtask {index} is not an object.")

                description = str(item.get("description", "")).strip()
                if not description:
                    raise StructuredOutputSchemaError(f"Subtask {index} missing description.")

                expected_outcome = str(item.get("expected_outcome", "User request fulfilled.")).strip()
                normalized_subtasks.append({
                    "id": item.get("id", index),
                    "description": description,
                    "expected_outcome": expected_outcome or "User request fulfilled.",
                })
            if not thinking_mode and self._looks_like_direct_lookup(user_query) and len(normalized_subtasks) > 1:
                llm_manager.log_event(
                    "Task splitting collapsed to single-step plan | reason=direct_lookup_default_mode"
                )
                return self._single_task_plan(user_query)
            if self._looks_like_direct_lookup(user_query) and self._looks_over_decomposed(normalized_subtasks):
                llm_manager.log_event(
                    "Task splitting collapsed to single-step plan | reason=direct_lookup_over_decomposed"
                )
                return self._single_task_plan(user_query)
            return normalized_subtasks
        except StructuredOutputError as e:
            llm_manager.log_event(f"Task splitting structured parsing failed: {e}", level=40)
            return self._single_task_plan(user_query)
        except Exception as e:
            llm_manager.log_event(f"Task splitting failed: {e}", level=40)
            # Fallback to single task
            return self._single_task_plan(user_query)
