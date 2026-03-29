import json
from typing import List, Dict, Any
from llm_manager import llm_manager

class TaskPlanner:
    def __init__(self):
        self.llm = llm_manager.get_llm()

    def split_task(self, user_query: str) -> List[Dict[str, Any]]:
        """
        根据核心需求规范，将复杂任务拆解成细粒度的“子任务”。
        要求：输出为JSON数组，每个子任务包含描述、预期结果等。
        """
        prompt = f"""
        You are an advanced planning agent. Break down the following complex user task into a sequence of actionable, fine-grained subtasks.
        Requirement:
        1. Each subtask must be independent enough to potentially map to exactly ONE tool or skill.
        2. Define the expected outcome (prediction) for each subtask to be used later for verification.
        
        Respond STRICTLY in JSON array format:
        [
            {{
                "id": 1,
                "description": "Specific action to take",
                "expected_outcome": "What the successful result should look like"
            }}
        ]
        
        Task to split:
        {user_query}
        """
        try:
            response = self.llm.invoke(prompt)
            # Find JSON content
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.strip("`").replace("json\n", "", 1)
            
            subtasks = json.loads(content)
            if isinstance(subtasks, list):
                return subtasks
            return []
        except Exception as e:
            print(f"Task splitting failed: {e}")
            # Fallback to single task
            return [{"id": 1, "description": user_query, "expected_outcome": "User request fulfilled."}]
