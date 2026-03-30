import json
import os
from typing import Any, Dict, Tuple, Type

import yaml
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model


class MCPManager:
    TYPE_MAPPING: Dict[str, Type[Any]] = {
        "string": str,
        "str": str,
        "integer": int,
        "int": int,
        "number": float,
        "float": float,
        "boolean": bool,
        "bool": bool,
    }

    def __init__(self):
        self.loaded_servers: Dict[str, Dict[str, Any]] = {}
        self.loaded_tools: Dict[str, StructuredTool] = {}

    def load_server(self, config_path: str) -> Tuple[bool, str, Dict[str, Any] | None, list[StructuredTool]]:
        if not os.path.exists(config_path):
            return False, f"MCP config not found: {config_path}", None, []

        try:
            server_config = self._read_config(config_path)
            server_name = server_config.get("name") or os.path.splitext(os.path.basename(config_path))[0]
            tools = self._build_tools(server_name, server_config)
        except Exception as exc:
            return False, f"Failed to load MCP config {os.path.basename(config_path)}: {exc}", None, []

        self.loaded_servers[server_name] = {
            "name": server_name,
            "description": server_config.get("description", ""),
            "transport": server_config.get("transport", "configured"),
            "source": config_path,
            "tool_names": [tool.name for tool in tools],
        }
        for tool in tools:
            self.loaded_tools[tool.name] = tool

        return True, f"Loaded MCP server {server_name} with {len(tools)} tool(s).", self.loaded_servers[server_name], tools

    def _read_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as file_handle:
            if config_path.endswith((".yaml", ".yml")):
                data = yaml.safe_load(file_handle)
            else:
                data = json.load(file_handle)

        if not isinstance(data, dict):
            raise ValueError("MCP config root must be an object")
        if not isinstance(data.get("tools", []), list) or not data.get("tools"):
            raise ValueError("MCP config must define a non-empty tools list")
        return data

    def _build_tools(self, server_name: str, server_config: Dict[str, Any]) -> list[StructuredTool]:
        tools = []
        for tool_config in server_config.get("tools", []):
            tools.append(self._build_tool(server_name, tool_config))
        return tools

    def _build_tool(self, server_name: str, tool_config: Dict[str, Any]) -> StructuredTool:
        tool_name = tool_config.get("name")
        if not tool_name:
            raise ValueError("Each MCP tool must define a name")

        description = tool_config.get("description", f"Tool exposed by MCP server {server_name}.")
        parameters = tool_config.get("parameters", {})
        args_schema = self._build_args_schema(server_name, tool_name, parameters)
        response_template = tool_config.get(
            "response_template",
            f"MCP tool {tool_name} on server {server_name} executed with arguments: {{arguments}}",
        )
        static_response = tool_config.get("static_response", "")

        def tool_callable(**kwargs):
            if static_response:
                return static_response.format(**kwargs)
            if "{arguments}" in response_template:
                return response_template.format(arguments=json.dumps(kwargs, ensure_ascii=False), **kwargs)
            return response_template.format(**kwargs)

        return StructuredTool.from_function(
            func=tool_callable,
            name=tool_name,
            description=description,
            args_schema=args_schema,
        )

    def _build_args_schema(self, server_name: str, tool_name: str, parameters: Dict[str, Any]) -> Type[BaseModel]:
        fields: Dict[str, tuple[Any, Any]] = {}
        for param_name, param_config in parameters.items():
            if not isinstance(param_config, dict):
                raise ValueError(f"Parameter {param_name} for tool {tool_name} must be an object")

            param_type = self.TYPE_MAPPING.get(str(param_config.get("type", "string")).lower(), str)
            default = ... if param_config.get("required", True) else param_config.get("default", None)
            description = param_config.get("description", f"Argument {param_name} for tool {tool_name}.")
            fields[param_name] = (param_type, Field(default=default, description=description))

        if not fields:
            fields["request"] = (
                str,
                Field(default=..., description=f"Input request for MCP tool {tool_name} on server {server_name}."),
            )

        model_name = f"{server_name.title().replace('-', '').replace('_', '')}{tool_name.title().replace('-', '').replace('_', '')}Args"
        return create_model(model_name, **fields)