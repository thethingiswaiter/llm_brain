import re
from typing import List, Dict, Any
from core.llm.manager import llm_manager
from cognitive.structured_output import (
    StructuredOutputError,
    StructuredOutputSchemaError,
    StructuredOutputFunctionCallError,
    invoke_function_call,
    parse_json_array,
)

class TaskPlanner:
    EXPLICIT_TARGET_PATTERNS = [
        re.compile(r"[A-Za-z]:\\(?:[^\\\s\"'`]+\\)*[^\\\s\"'`]+\.[A-Za-z0-9]{1,12}"),
        re.compile(r"(?:\./|\.\./|\.\\|\.\.\\|/)(?:[^/\s\"'`]+/)*[^/\s\"'`]+\.[A-Za-z0-9]{1,12}"),
        re.compile(r"\b[\w.-]+\.[A-Za-z0-9]{1,12}\b"),
    ]
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
    WRITE_MARKERS = ["写", "写入", "保存", "创建", "追加", "覆盖", "修改", "更新", "生成", "补充", "添加", "write", "save", "create", "append", "overwrite", "update"]

    def _extract_replan_focus_description(self, user_query: str) -> str:
        lines = [str(line).strip() for line in str(user_query or "").splitlines() if str(line).strip()]
        if not lines:
            return ""

        if not any("失败子任务:" in line or "失败的子任务" in line for line in lines):
            return ""

        for line in lines:
            if "失败子任务:" not in line:
                continue
            description = line.split("失败子任务:", 1)[1].strip()
            if description:
                return description
        return ""

    def _single_task_plan(self, user_query: str) -> List[Dict[str, Any]]:
        description = self._extract_replan_focus_description(user_query) or str(user_query).strip() or "完成用户请求。"
        description = self._preserve_explicit_targets(description, self._extract_explicit_targets(user_query))
        return [{
            "id": 1,
            "description": description,
            "execution_mode": "leaf",
        }]

    def _contains_non_executable_manual_step(self, description: str) -> bool:
        combined = str(description or "").lower()
        patterns = [
            r"(?:打开|使用|在)\s*vscode",
            r"visual\s+studio\s+code",
            r"notepad\+\+",
            r"notepad",
            r"文本编辑器",
            r"编辑器",
            r"手动",
            r"人工",
            r"人类",
            r"请用户",
            r"让用户",
            r"用户手动",
            r"open\s+in\s+editor",
            r"manual",
            r"manually",
            r"ask\s+the\s+user\s+to",
        ]
        return any(re.search(pattern, combined) for pattern in patterns)

    def _looks_like_write_or_create_step(self, description: str) -> bool:
        normalized = str(description or "").strip().lower()
        if not normalized:
            return False
        return any(marker in normalized for marker in self.WRITE_MARKERS)

    def _derive_missing_write_description(self, user_query: str, explicit_targets: List[str]) -> str:
        original = str(user_query or "").strip()
        target = explicit_targets[0] if explicit_targets else ""
        if not target:
            relaxed_match = re.search(r"([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12})", original)
            if relaxed_match:
                target = relaxed_match.group(1)
        if not target:
            target = "目标文件"
        normalized = original.lower()
        if "补充" in original and "配置" in original:
            return f"补充 {target} 的基础配置"
        if any(marker in normalized for marker in ("append", "追加")):
            return f"向 {target} 追加请求中提到的内容"
        if any(marker in normalized for marker in ("create", "创建")):
            return f"创建并写入 {target}"
        if any(marker in normalized for marker in ("update", "修改", "更新", "覆盖", "overwrite")):
            return f"更新 {target} 的内容"
        return self._preserve_explicit_targets(original, explicit_targets) or original or f"写入 {target}"

    def _append_missing_required_write_step(
        self,
        user_query: str,
        subtasks: List[Dict[str, Any]],
        explicit_targets: List[str],
    ) -> List[Dict[str, Any]]:
        normalized_query = str(user_query or "").strip().lower()
        if not normalized_query:
            return subtasks
        if not any(marker in normalized_query for marker in self.WRITE_MARKERS):
            return subtasks
        if any(self._looks_like_write_or_create_step(item.get("description", "")) for item in subtasks):
            return subtasks

        write_description = self._derive_missing_write_description(user_query, explicit_targets)
        if not write_description:
            return subtasks

        extended_subtasks = list(subtasks)
        extended_subtasks.append({
            "id": len(extended_subtasks) + 1,
            "description": self._preserve_explicit_targets(write_description, explicit_targets),
            "execution_mode": "leaf",
        })
        return extended_subtasks

    def _is_low_value_save_step(self, description: str) -> bool:
        normalized = str(description or "").strip().lower()
        if not normalized:
            return False
        save_only_patterns = [
            r"^保存[\s\S]{0,40}(文件|代码)?$",
            r"^将[\s\S]{0,40}(保存|写回).*$",
            r"^save(\s+the)?\s+file.*$",
        ]
        if any(re.search(pattern, normalized) for pattern in save_only_patterns):
            return True
        return "保存" in normalized and not self._looks_like_write_or_create_step(normalized)

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

    def _extract_explicit_targets(self, user_query: str) -> List[str]:
        query = str(user_query or "")
        candidates: list[str] = []
        seen: set[str] = set()
        for pattern in self.EXPLICIT_TARGET_PATTERNS:
            for match in pattern.findall(query):
                value = str(match or "").strip().strip(",.;:)}]\"'")
                marker = value.lower()
                if not value or marker in seen:
                    continue
                seen.add(marker)
                candidates.append(value)
        return candidates[:3]

    def _preserve_explicit_targets(self, description: str, explicit_targets: List[str]) -> str:
        normalized_description = str(description or "").strip()
        if not normalized_description or not explicit_targets:
            return normalized_description

        lowered_description = normalized_description.lower()
        missing_targets = [target for target in explicit_targets if target.lower() not in lowered_description]
        if not missing_targets:
            return normalized_description

        target_text = "，".join(missing_targets)
        return f"{normalized_description}（目标路径: {target_text}）"

    def split_task(
        self,
        user_query: str,
        thinking_mode: bool = False,
        capability_context: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        根据核心需求规范，将复杂任务拆解成细粒度的“子任务”。
        要求：优先通过函数调用返回结构化结果；如果模型不支持函数调用，则输出 JSON 数组。
        """
        capability_block = self._format_planning_capability_context(capability_context)
        capability_section = f"\n\n当前能力清单:\n{capability_block}" if capability_block else ""
        explicit_targets = self._extract_explicit_targets(user_query)
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
        7. 如果用户请求里包含明确的文件路径、文件名或目标位置，相关子任务 description 必须原样保留这些目标，不得把完整路径泛化成目录、类型或“某个配置文件”。
        8. 只返回 JSON，不要输出解释或 Markdown 代码块。{capability_section}
        
        如果模型支持函数调用，请直接调用函数返回结构化结果，不要输出解释文本。
        如果模型不支持函数调用，请严格按以下 JSON 数组格式返回:
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
            try:
                function_payload = invoke_function_call(
                    prompt,
                    function_name="submit_plan",
                    function_description="Return the subtask plan for the current user request.",
                    parameters_schema={
                        "type": "object",
                        "properties": {
                            "subtasks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "description": {"type": "string"},
                                        "execution_mode": {"type": "string", "enum": ["leaf", "decomposable"]},
                                    },
                                    "required": ["description", "execution_mode"],
                                },
                            }
                        },
                        "required": ["subtasks"],
                    },
                    source="planner.split_task.function_call",
                    fallback_content_parser=parse_json_array,
                    fallback_content_adapter=lambda value: {"subtasks": value},
                )
                subtasks = function_payload.get("subtasks")
                if not isinstance(subtasks, list):
                    raise StructuredOutputSchemaError("Function call field subtasks must be a JSON array.")
            except StructuredOutputFunctionCallError:
                response = llm_manager.invoke(prompt, source="planner.split_task")
                subtasks = parse_json_array(response.content)
            normalized_subtasks = []
            for index, item in enumerate(subtasks, start=1):
                if not isinstance(item, dict):
                    raise StructuredOutputSchemaError(f"Subtask {index} is not an object.")

                description = str(item.get("description", "")).strip()
                if not description:
                    raise StructuredOutputSchemaError(f"Subtask {index} missing description.")
                description = self._preserve_explicit_targets(description, explicit_targets)

                execution_mode = self._normalize_execution_mode(item.get("execution_mode", "leaf"))
                if self._should_force_decomposable(description):
                    execution_mode = "decomposable"
                if self._contains_non_executable_manual_step(description):
                    raise StructuredOutputSchemaError(
                        f"Subtask {index} contains manual or editor-dependent action: {description}"
                    )
                if self._is_low_value_save_step(description):
                    if any(self._looks_like_write_or_create_step(existing.get("description", "")) for existing in normalized_subtasks):
                        continue
                normalized_subtasks.append({
                    "id": item.get("id", index),
                    "description": description,
                    "execution_mode": execution_mode,
                })
            normalized_subtasks = self._append_missing_required_write_step(user_query, normalized_subtasks, explicit_targets)
            if not normalized_subtasks:
                return self._single_task_plan(user_query)
            return normalized_subtasks
        except StructuredOutputError as e:
            llm_manager.log_event(f"Task splitting structured parsing failed: {e}", level=40)
            return self._single_task_plan(user_query)
        except Exception as e:
            llm_manager.log_event(f"Task splitting failed: {e}", level=40)
            # Fallback to single task
            return self._single_task_plan(user_query)
