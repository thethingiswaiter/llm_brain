from langchain_core.tools import tool
import datetime

@tool
def get_current_time() -> str:
    """获取当前的日期和时间。当用户询问时间、今天是几号等问题时使用此工具。"""
    now = datetime.datetime.now()
    return f"当前的系统时间是 {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def calculate_sum(a: float, b: float) -> float:
    """计算两个数字的和。当用户需要将两个数字相加时使用此工具。"""
    return a + b

@tool
def get_mock_weather(city: str) -> str:
    """获取指定城市的当前天气情况（模拟数据）。当用户询问某个城市的天气时使用此工具。"""
    mock_data = {
        "北京": "晴天，气温 22°C，微风",
        "上海": "多云，气温 25°C，相对湿度较高",
        "广州": "阵雨，气温 28°C，建议带伞"
    }
    return mock_data.get(city, f"{city} 的天气未知，但大概率是个好天气！")

# 必须导出一个名为 tools 的列表，供 agent_core 自动扫描加载
tools = [get_current_time, calculate_sum, get_mock_weather]
