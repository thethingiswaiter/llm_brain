from typing import List, Dict, Any
from core.llm.manager import llm_manager
from cognitive.structured_output import parse_json_array, StructuredOutputError, StructuredOutputSchemaError

class TaskPlanner:
    COMPLEX_TASK_MARKERS = [
        "分析",
        "排查",
        "规划",
        "重构",
        "修复",
        "迁移",
        "整理",
        "汇总",
        "设计",
        "实现",
        "再",
        "继续拆",
        "逐步",
        "多步",
        "分阶段",
        "collect",
        "analyze",
        "investigate",
        "replan",
        "refactor",
        "migrate",
        "design",
        "implement",
        "step by step",
        "multi-step",
    ]

    def _single_task_plan(self, user_query: str) -> List[Dict[str, Any]]:
        description = str(user_query).strip() or "完成用户请求。"
        return [{
            "id": 1,
            "description": description,
            "execution_mode": "leaf",
        }]

    def _contains_non_executable_manual_step(self, description: str) -> bool:
        combined = str(description or "").lower()
        markers = [
            "vscode",
            "visual studio code",
            "notepad",
            "notepad++",
            "文本编辑器",
            "编辑器",
            "手动",
            "人工",
            "人类",
            "请用户",
            "让用户",
            "用户手动",
            "open in editor",
            "manual",
            "manually",
            "ask the user to",
        ]
        return any(marker in combined for marker in markers)

    def _format_planning_capability_context(self, capability_context: Dict[str, Any] | None) -> str:
        if not isinstance(capability_context, dict):
            return ""

        lines: list[str] = []
        planning_policy = str(capability_context.get("planning_policy", "")).strip()
        if planning_policy:
            lines.append(f"规划原则: {planning_policy}")

        prompt_skills = capability_context.get("prompt_skills", [])
        if isinstance(prompt_skills, list) and prompt_skills:
            lines.append("当前可用 Prompt 技能:")
            for item in prompt_skills:
                if not isinstance(item, dict):
                    continue
                keywords = ", ".join(item.get("keywords", [])[:8])
                lines.append(
                    f"- {item.get('name', '')}: {item.get('description', '')}"
                    f" | keywords: {keywords} | source: {item.get('source_file', '')}"
                )

        tools = capability_context.get("tools", [])
        if isinstance(tools, list) and tools:
            lines.append("当前可用工具:")
            for item in tools:
                if not isinstance(item, dict):
                    continue
                keywords = ", ".join(item.get("keywords", [])[:8])
                constraints = "; ".join(item.get("constraints", [])[:4])
                arguments = "; ".join(item.get("arguments", [])[:4])
                tool_line = (
                    f"- {item.get('name', '')}: {item.get('description', '')}"
                    f" | keywords: {keywords} | source_type: {item.get('source_type', '')}"
                )
                if arguments:
                    tool_line += f" | args: {arguments}"
                if constraints:
                    tool_line += f" | constraints: {constraints}"
                lines.append(tool_line)

        if not lines:
            return ""
        return "\n".join(lines)

    def _normalize_execution_mode(self, value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"leaf", "single_step", "single-step", "atomic"}:
            return "leaf"
        if normalized in {"decomposable", "complex", "multi_step", "multi-step", "compound"}:
            return "decomposable"
        return "leaf"

    def _should_force_decomposable(self, description: str) -> bool:
        normalized = str(description or "").strip().lower()
        if not normalized:
            return False
        if any(marker in normalized for marker in self.COMPLEX_TASK_MARKERS):
            return True
        if normalized.count("并") >= 2 or normalized.count("然后") >= 1:
            return True
        return False

    def split_task(
        self,
        user_query: str,
        thinking_mode: bool = False,
        capability_context: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        根据核心需求规范，将复杂任务拆解成细粒度的“子任务”。
        要求：输出为JSON数组，每个子任务包含描述、预期结果等。
        """
        capability_block = self._format_planning_capability_context(capability_context)
        capability_section = f"\n\n当前能力清单:\n{capability_block}" if capability_block else ""
        prompt = f"""
        你是一个中文优先的任务规划代理。请把下面的用户任务拆解成一组可执行、细粒度的子任务。
        要求:
        1. 每个子任务都应尽量独立，并且最好只需要一个工具或一种技能即可完成。
          2. 为每个子任务标注 execution_mode，只能是 "leaf" 或 "decomposable"。
              - "leaf": 这是单步可执行子任务，应直接进入执行与校验。
              - "decomposable": 这是仍需继续拆解的复杂子任务，应在执行前再次规划。
        3. 规划时只能依赖当前已经注册的技能和工具，不要假设存在未列出的专用能力。
        4. 如果某个技能或工具已经足以直接完成请求，应优先保持计划简洁，避免拆成没有独立价值的元步骤。
        5. 针对具体问题的处理方式应该来自技能正文、工具描述和能力边界，而不是凭空发明隐藏流程。
        6. 子任务必须是 agent 自己可执行的动作，禁止输出“手动操作 / 人工检查 / 让用户自己做”这类依赖人类或外部 GUI 手动完成的步骤。
        7. 只返回 JSON，不要输出解释或 Markdown 代码块。{capability_section}
        
        请严格按以下 JSON 数组格式返回:
        [
            {{
                "id": 1,
                "description": "要执行的具体动作",
                "execution_mode": "leaf"
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

                execution_mode = self._normalize_execution_mode(item.get("execution_mode", "leaf"))
                if self._should_force_decomposable(description):
                    execution_mode = "decomposable"
                if self._contains_non_executable_manual_step(description):
                    raise StructuredOutputSchemaError(
                        f"Subtask {index} contains manual or editor-dependent action: {description}"
                    )
                normalized_subtasks.append({
                    "id": item.get("id", index),
                    "description": description,
                    "execution_mode": execution_mode,
                })
            return normalized_subtasks
        except StructuredOutputError as e:
            llm_manager.log_event(f"Task splitting structured parsing failed: {e}", level=40)
            return self._single_task_plan(user_query)
        except Exception as e:
            llm_manager.log_event(f"Task splitting failed: {e}", level=40)
            # Fallback to single task
            return self._single_task_plan(user_query)
