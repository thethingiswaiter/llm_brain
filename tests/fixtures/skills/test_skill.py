from langchain_core.tools import tool
import datetime


CHINA_TIMEZONE = datetime.timezone(datetime.timedelta(hours=8), name="CST")


@tool
def get_current_time() -> str:
    """Return current time."""
    now = datetime.datetime.now(CHINA_TIMEZONE)
    return f"Current time is {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def calculate_sum(a: float, b: float) -> float:
    """Calculate sum of two numbers."""
    return a + b


@tool
def get_mock_weather(city: str) -> str:
    """Return mock weather data for a city."""
    mock_data = {
        "beijing": "Sunny, 22C, light wind",
        "shanghai": "Cloudy, 25C, humid",
        "guangzhou": "Showers, 28C, umbrella suggested",
    }
    return mock_data.get(city.lower(), f"Weather for {city} is unknown")


tools = [get_current_time, calculate_sum, get_mock_weather]


