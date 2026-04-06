from typing import Dict, Any, Tuple
import re
from core.llm.manager import llm_manager
from cognitive.structured_output import (
    StructuredOutputError,
    StructuredOutputFunctionCallError,
    StructuredOutputSchemaError,
    invoke_function_call,
    parse_json_object,
)

class Reflector:
    def _looks_like_write_task(self, subtask: str, expected_outcome: str = "") -> bool:
        combined = f"{subtask}\n{expected_outcome}".lower()
        markers = [
            "写", "写入", "保存", "创建", "追加", "覆盖", "修改", "更新", "生成", "补充", "添加",
            "write", "save", "create", "append", "overwrite", "modify", "update",
        ]
        return any(marker in combined for marker in markers)

    def _looks_like_observation_task(self, subtask: str, expected_outcome: str = "") -> bool:
        combined = f"{subtask}\n{expected_outcome}".lower()
        markers = [
            "查看", "读取", "列出", "查询", "显示", "检查", "获取",
            "read", "show", "list", "inspect", "query", "get",
        ]
        return any(marker in combined for marker in markers)

    def _looks_like_user_input_is_still_required(self, actual_result: str) -> bool:
        normalized = str(actual_result or "").lower()
        markers = [
            "请补充", "请确认", "请提供", "缺少", "未提供", "参数不足", "需要你", "missing", "need user",
            "need confirmation", "please provide", "please confirm",
        ]
        return any(marker in normalized for marker in markers)

    def _has_concrete_observation(self, actual_result: str) -> bool:
        normalized = str(actual_result or "").strip()
        if not normalized:
            return False
        if self._looks_like_user_input_is_still_required(normalized):
            return False
        evidence_markers = ["如下", "内容", "结果", "显示", "找到", "未找到", "路径", "文件"]
        if any(marker in normalized for marker in evidence_markers):
            return True
        if re.search(r'".+"|`.+`', normalized):
            return True
        return len(normalized) >= 12

    def verify_and_reflect(self, subtask: str, expected_outcome: str, actual_result: str) -> Tuple[bool, str, str]:
        """
        根据核心需求规范，验证“预测结果”与“实际结果”。
        纠偏分支与异常阻断：若失败，需触发自主反思。
        Returns:
            - is_success: bool
            - reflection_note: str (分析原因：特征不足？外部缺失？推演错乱？)
            - action: 'continue', 'retry', 'ask_user'
        """
        if (
            self._looks_like_observation_task(subtask, expected_outcome)
            and not self._looks_like_write_task(subtask, expected_outcome)
            and self._has_concrete_observation(actual_result)
        ):
            return True, "已基于现有工具结果获得可直接返回的观测结果。", "continue"

        expected_section = f"预期结果: {expected_outcome}\n" if str(expected_outcome or "").strip() else ""
        prompt = f"""
        你是一个中文优先的反思与校验代理，负责分析当前子任务是否完成。
        子任务: {subtask}
        {expected_section}实际结果: {actual_result}

        请判断实际结果是否已经完成当前子任务。
        如果提供了预期结果，可以把它作为辅助参考；如果没有预期结果，就直接围绕子任务本身做判断。
        如果任务本质上只是查看、读取、列出、查询现有信息，只要已经拿到可直接返回给用户的观测结果，就应判定为 success=true、action=continue；
        即使结果显示目标内容是占位符、为空、未找到或与主观预期不一致，也不要因此改成 ask_user，除非确实缺少用户必须补充的外部信息。
        如果失败，请分析原因，例如信息缺失、参数不足、逻辑错误、工具受限等。
        然后决定下一步动作:
        - "continue" if success or failure is trivial and doesn't block progress.
        - "retry" if a different approach to the same task might work.
        - "ask_user" if completely blocked and extra knowledge/action is needed.

        如果模型支持函数调用，请直接调用函数返回结构化结果，不要输出解释文本。
        如果模型不支持函数调用，只返回 JSON，不要输出解释或 Markdown 代码块。
        请严格按以下 JSON 返回:
        {{
            "success": true/false,
            "reflection": "中文分析说明",
            "action": "continue" / "retry" / "ask_user"
        }}
        """
        try:
            try:
                data = invoke_function_call(
                    prompt,
                    function_name="submit_reflection",
                    function_description="Return the validation result for the current subtask.",
                    parameters_schema={
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean"},
                            "reflection": {"type": "string"},
                            "action": {"type": "string", "enum": ["continue", "retry", "ask_user"]},
                        },
                        "required": ["success", "reflection", "action"],
                    },
                    source="reflector.verify_and_reflect.function_call",
                    fallback_content_parser=lambda value: parse_json_object(
                        value,
                        required_fields={"success": bool, "reflection": str, "action": str},
                    ),
                )
            except StructuredOutputFunctionCallError:
                response = llm_manager.invoke(prompt, source="reflector.verify_and_reflect")
                raw_payload = getattr(response, "content", response)
                try:
                    data = parse_json_object(
                        raw_payload,
                        required_fields={"success": bool, "reflection": str, "action": str},
                    )
                except StructuredOutputError:
                    data = parse_json_object(
                        llm_manager._stringify_response(response),
                        required_fields={"success": bool, "reflection": str, "action": str},
                    )
            if data["action"] not in {"continue", "retry", "ask_user"}:
                raise StructuredOutputSchemaError("Field action must be one of continue/retry/ask_user.")
            return (
                data.get("success", False),
                data.get("reflection", "No specific reflection generated."),
                data.get("action", "ask_user")
            )
        except StructuredOutputError as e:
            llm_manager.log_event(f"Reflection structured parsing failed: {e}", level=40)
            return False, f"Failed to parse reflection: {e}", "ask_user"
        except Exception as e:
            llm_manager.log_event(f"Reflection failed: {e}", level=40)
            return False, f"Failed to parse reflection: {e}", "ask_user"
