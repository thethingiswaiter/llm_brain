from langchain_core.tools import tool


@tool
def sample_hello(name: str):
    """Greets the user by name."""
    return f"Hello, {name}! I am a placeholder skill."


# Important: Must define an iterable named 'tools' for the auto-loader to import correctly
tools = [sample_hello]
