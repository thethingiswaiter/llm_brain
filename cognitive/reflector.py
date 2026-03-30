from typing import Dict, Any, Tuple
import json
from llm_manager import llm_manager

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
        You are a reflection AI analyzing a subtask execution.
        Subtask: {subtask}
        Expected Outcome (Prediction): {expected_outcome}
        Actual Result: {actual_result}

        Determine if the actual result met the expected outcome.
        If it failed, analyze why (e.g., missing features, missing information, logic error).
        Decide the next action:
        - "continue" if success or failure is trivial and doesn't block progress.
        - "retry" if a different approach to the same task might work.
        - "ask_user" if completely blocked and extra knowledge/action is needed.

        Respond strictly in JSON:
        {{
            "success": true/false,
            "reflection": "Detailed analysis notes",
            "action": "continue" / "retry" / "ask_user"
        }}
        """
        try:
            response = llm_manager.invoke(prompt, source="reflector.verify_and_reflect")
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.strip("`").replace("json\n", "", 1)
            
            data = json.loads(content)
            return (
                data.get("success", False),
                data.get("reflection", "No specific reflection generated."),
                data.get("action", "ask_user")
            )
        except Exception as e:
            llm_manager.log_event(f"Reflection failed: {e}", level=40)
            return False, f"Failed to parse reflection: {e}", "ask_user"
