from typing import Dict, Any, Tuple
from core.llm.manager import llm_manager
from cognitive.structured_output import parse_json_object, StructuredOutputError, StructuredOutputSchemaError

class Reflector:
    def verify_and_reflect(self, subtask: str, expected_outcome: str, actual_result: str) -> Tuple[bool, str, str]:
        """
        根据核心需求规范，验证“预测结果”与“实际结果”。
        纠偏分支与异常阻断：若失败，需触发自主反思。
        Returns:
            - is_success: bool
            - reflection_note: str (分析原因：特征不足？外部缺失？推演错乱？)
            - action: 'continue', 'retry', 'ask_user'
        """
        prompt = f"""
        你是一个中文优先的反思与校验代理，负责分析当前子任务是否完成。
        子任务: {subtask}
        预期结果: {expected_outcome}
        实际结果: {actual_result}

        请判断实际结果是否满足预期结果。
        如果失败，请分析原因，例如信息缺失、参数不足、逻辑错误、工具受限等。
        然后决定下一步动作:
        - "continue" if success or failure is trivial and doesn't block progress.
        - "retry" if a different approach to the same task might work.
        - "ask_user" if completely blocked and extra knowledge/action is needed.

        只返回 JSON，不要输出解释或 Markdown 代码块。
        请严格按以下 JSON 返回:
        {{
            "success": true/false,
            "reflection": "中文分析说明",
            "action": "continue" / "retry" / "ask_user"
        }}
        """
        try:
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
