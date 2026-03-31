import atexit
import asyncio
import json
import os
import shlex
import sys
from threading import Event, Lock, Thread
from datetime import timedelta
from typing import Any, Dict, Tuple, Type

import yaml
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

try:
    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError:
    anyio = None
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


class _PersistentStdioConnection:
    def __init__(self, server_parameters: Any, startup_timeout_seconds: float = 10.0):
        self.server_parameters = server_parameters
        self.startup_timeout_seconds = startup_timeout_seconds
        self._ready_event = Event()
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._session = None
        self._start_error: Exception | None = None
        self._runtime_error: Exception | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._thread = Thread(target=self._thread_main, name="mcp-stdio-connection", daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=self.startup_timeout_seconds):
            raise TimeoutError(f"Timed out starting MCP stdio connection after {self.startup_timeout_seconds} seconds")
        if self._start_error:
            raise self._start_error

    def _thread_main(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        try:
            async with stdio_client(self.server_parameters) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    self._loop = asyncio.get_running_loop()
                    self._session = session
                    self._stop_event = asyncio.Event()
                    await session.initialize()
                    self._ready_event.set()
                    await self._stop_event.wait()
        except Exception as exc:
            if not self._ready_event.is_set():
                self._start_error = exc
                self._ready_event.set()
            else:
                self._runtime_error = exc
        finally:
            self._session = None
            self._loop = None
            self._stop_event = None

    def ensure_running(self) -> None:
        if self._start_error:
            raise RuntimeError(f"MCP stdio connection failed to start: {self._start_error}")
        if self._runtime_error:
            raise RuntimeError(f"MCP stdio connection terminated unexpectedly: {self._runtime_error}")
        if not self._thread or not self._thread.is_alive() or self._loop is None or self._session is None:
            raise RuntimeError("MCP stdio connection is not active")

    def list_tools(self, timeout_seconds: float = 20.0) -> list[Any]:
        self.ensure_running()
        future = asyncio.run_coroutine_threadsafe(self._session.list_tools(), self._loop)
        return list(future.result(timeout=timeout_seconds).tools)

    def call_tool(self, tool_name: str, arguments: Dict[str, Any], timeout_seconds: float = 20.0) -> Any:
        self.ensure_running()
        read_timeout = timedelta(seconds=timeout_seconds) if timeout_seconds > 0 else None
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(tool_name, arguments=arguments or None, read_timeout_seconds=read_timeout),
            self._loop,
        )
        return future.result(timeout=timeout_seconds + 2 if timeout_seconds > 0 else None)

    def close(self, timeout_seconds: float = 5.0) -> None:
        if self._loop and self._stop_event and self._thread and self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._stop_event.set)
            self._thread.join(timeout=timeout_seconds)


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
        "object": dict,
        "array": list,
    }

    def __init__(self):
        self.loaded_servers: Dict[str, Dict[str, Any]] = {}
        self.loaded_tools: Dict[str, StructuredTool] = {}
        self._remote_connections: Dict[str, _PersistentStdioConnection] = {}
        self._connection_lock = Lock()
        atexit.register(self.close_all)

    def load_server(self, server_ref: str) -> Tuple[bool, str, Dict[str, Any] | None, list[StructuredTool]]:
        if not server_ref:
            return False, "MCP server reference is required.", None, []

        try:
            source_path = None
            if server_ref.startswith("stdio:"):
                server_name, server_config = self._build_inline_stdio_server(server_ref)
                source = server_ref
            else:
                if not os.path.exists(server_ref):
                    return False, f"MCP config or server not found: {server_ref}", None, []
                source_path = os.path.abspath(server_ref)
                if source_path.endswith(".py"):
                    server_name, server_config = self._build_python_stdio_server(source_path)
                else:
                    server_config = self._read_config(source_path)
                    server_name = server_config.get("name") or os.path.splitext(os.path.basename(source_path))[0]
                source = source_path

            transport = str(server_config.get("transport", "config")).lower()
            if transport == "stdio":
                tools = self._build_remote_tools(server_name, server_config, source)
            else:
                tools = self._build_tools(server_name, server_config)
        except Exception as exc:
            label = os.path.basename(server_ref) if os.path.exists(server_ref) else server_ref
            return False, f"Failed to load MCP server {label}: {exc}", None, []

        self.loaded_servers[server_name] = {
            "name": server_name,
            "description": server_config.get("description", ""),
            "transport": server_config.get("transport", "configured"),
            "source": source,
            "tool_names": [tool.name for tool in tools],
            "connection_key": source if str(server_config.get("transport", "config")).lower() == "stdio" else "",
            "server_config": server_config,
        }
        for tool in tools:
            self.loaded_tools[tool.name] = tool

        return True, f"Loaded MCP server {server_name} with {len(tools)} tool(s).", self.loaded_servers[server_name], tools

    def _build_inline_stdio_server(self, server_ref: str) -> tuple[str, Dict[str, Any]]:
        command_line = server_ref[len("stdio:") :].strip()
        if not command_line:
            raise ValueError("stdio MCP reference must include a command line")

        parts = shlex.split(command_line, posix=os.name != "nt")
        if not parts:
            raise ValueError("stdio MCP reference must include a command")

        command = parts[0]
        args = parts[1:]
        server_name = os.path.splitext(os.path.basename(args[0]))[0] if args else os.path.basename(command)
        return server_name or "stdio_mcp", {
            "name": server_name or "stdio_mcp",
            "description": f"stdio MCP server launched from inline reference: {command_line}",
            "transport": "stdio",
            "command": command,
            "args": args,
        }

    def _build_python_stdio_server(self, server_path: str) -> tuple[str, Dict[str, Any]]:
        server_name = os.path.splitext(os.path.basename(server_path))[0]
        return server_name, {
            "name": server_name,
            "description": f"stdio MCP server launched from Python script {server_path}",
            "transport": "stdio",
            "command": sys.executable,
            "args": [server_path],
            "cwd": os.path.dirname(server_path),
        }

    def _read_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as file_handle:
            if config_path.endswith((".yaml", ".yml")):
                data = yaml.safe_load(file_handle)
            else:
                data = json.load(file_handle)

        if not isinstance(data, dict):
            raise ValueError("MCP config root must be an object")

        transport = str(data.get("transport", "config")).lower()
        if transport == "stdio":
            if not data.get("command"):
                raise ValueError("stdio MCP config must define a command")
            if not isinstance(data.get("args", []), list):
                raise ValueError("stdio MCP config args must be a list when provided")
            return data

        if not isinstance(data.get("tools", []), list) or not data.get("tools"):
            raise ValueError("MCP config must define a non-empty tools list")
        return data

    def _build_tools(self, server_name: str, server_config: Dict[str, Any]) -> list[StructuredTool]:
        tools = []
        for tool_config in server_config.get("tools", []):
            tools.append(self._build_tool(server_name, tool_config))
        return tools

    def _build_remote_tools(self, server_name: str, server_config: Dict[str, Any], connection_key: str) -> list[StructuredTool]:
        tool_specs = self._list_stdio_tools(server_config, connection_key)
        tools = []
        for tool_spec in tool_specs:
            tools.append(self._build_remote_tool(server_name, server_config, tool_spec, connection_key))
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

    def _build_remote_tool(self, server_name: str, server_config: Dict[str, Any], tool_spec: Any, connection_key: str) -> StructuredTool:
        tool_name = getattr(tool_spec, "name", None)
        if not tool_name:
            raise ValueError("Remote MCP tool is missing a name")

        description = getattr(tool_spec, "description", None) or f"Tool exposed by MCP server {server_name}."
        input_schema = getattr(tool_spec, "inputSchema", None) or {}
        args_schema = self._build_args_schema_from_json_schema(server_name, tool_name, input_schema)

        def tool_callable(**kwargs):
            return self._call_stdio_tool(server_config, connection_key, tool_name, kwargs)

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

        model_name = f"{server_name.title().replace('-', '').replace('_', '')}{tool_name.title().replace('-', '').replace('_', '')}Args"
        return create_model(model_name, **fields)

    def _build_args_schema_from_json_schema(
        self,
        server_name: str,
        tool_name: str,
        input_schema: Dict[str, Any],
    ) -> Type[BaseModel]:
        if not isinstance(input_schema, dict):
            raise ValueError(f"Remote MCP tool {tool_name} returned an invalid input schema")

        properties = input_schema.get("properties", {}) or {}
        required = set(input_schema.get("required", []) or [])
        fields: Dict[str, tuple[Any, Any]] = {}
        for param_name, param_config in properties.items():
            if not isinstance(param_config, dict):
                continue
            param_type = self.TYPE_MAPPING.get(str(param_config.get("type", "string")).lower(), Any)
            default = ... if param_name in required else param_config.get("default", None)
            description = param_config.get("description", f"Argument {param_name} for MCP tool {tool_name}.")
            fields[param_name] = (param_type, Field(default=default, description=description))

        model_name = f"{server_name.title().replace('-', '').replace('_', '')}{tool_name.title().replace('-', '').replace('_', '')}RemoteArgs"
        return create_model(model_name, **fields)

    def _list_stdio_tools(self, server_config: Dict[str, Any], connection_key: str) -> list[Any]:
        if anyio is None or ClientSession is None or StdioServerParameters is None or stdio_client is None:
            raise RuntimeError("mcp package is required for stdio MCP transport")
        connection = self._get_or_create_connection(connection_key, server_config)
        timeout_seconds = float(server_config.get("timeout_seconds", 20))
        return connection.list_tools(timeout_seconds=timeout_seconds)

    def _call_stdio_tool(self, server_config: Dict[str, Any], connection_key: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        if anyio is None or ClientSession is None or StdioServerParameters is None or stdio_client is None:
            raise RuntimeError("mcp package is required for stdio MCP transport")
        connection = self._get_or_create_connection(connection_key, server_config)
        timeout_seconds = float(server_config.get("timeout_seconds", 20))
        result = connection.call_tool(tool_name, arguments=arguments, timeout_seconds=timeout_seconds)
        return self._format_call_tool_result(result)

    def _get_or_create_connection(self, connection_key: str, server_config: Dict[str, Any]) -> _PersistentStdioConnection:
        with self._connection_lock:
            connection = self._remote_connections.get(connection_key)
            if connection is None:
                connection = self._create_connection(server_config)
                self._remote_connections[connection_key] = connection
            else:
                try:
                    connection.ensure_running()
                except Exception:
                    try:
                        connection.close()
                    except Exception:
                        pass
                    connection = self._create_connection(server_config)
                    self._remote_connections[connection_key] = connection
            return connection

    def _create_connection(self, server_config: Dict[str, Any]) -> _PersistentStdioConnection:
        server_parameters = self._build_stdio_server_parameters(server_config)
        connection = _PersistentStdioConnection(
            server_parameters,
            startup_timeout_seconds=float(server_config.get("startup_timeout_seconds", 10)),
        )
        connection.start()
        return connection

    def _build_stdio_server_parameters(self, server_config: Dict[str, Any]) -> Any:
        command = server_config.get("command")
        if not command:
            raise ValueError("stdio MCP config must define a command")

        args = [str(item) for item in server_config.get("args", [])]
        env = server_config.get("env")
        cwd = server_config.get("cwd")
        return StdioServerParameters(
            command=str(command),
            args=args,
            env=env,
            cwd=cwd,
        )

    def _format_call_tool_result(self, result: Any) -> str:
        content_items = []
        text_chunks = []
        for item in getattr(result, "content", []) or []:
            if hasattr(item, "model_dump"):
                dumped = item.model_dump(mode="json")
            else:
                dumped = str(item)
            content_items.append(dumped)
            if isinstance(dumped, dict) and dumped.get("type") == "text":
                text_chunks.append(dumped.get("text", ""))

        if getattr(result, "isError", False):
            error_message = "\n\n".join(chunk for chunk in text_chunks if chunk).strip()
            if not error_message:
                error_message = json.dumps(content_items, ensure_ascii=False)
            raise RuntimeError(error_message)

        text_output = "\n\n".join(chunk for chunk in text_chunks if chunk).strip()
        if text_output:
            return text_output
        return json.dumps(content_items, ensure_ascii=False)

    def list_servers(self) -> list[Dict[str, Any]]:
        return [dict(item) for item in sorted(self.loaded_servers.values(), key=lambda entry: entry["name"])]

    def unload_server(self, server_ref: str) -> tuple[bool, str, Dict[str, Any] | None]:
        normalized_ref = (server_ref or "").strip()
        if not normalized_ref:
            return False, "Usage: unload_mcp_server <server_name|source>", None

        match_name = None
        for server_name, server_info in self.loaded_servers.items():
            source = server_info.get("source", "")
            basename = os.path.basename(source) if source and not str(source).startswith("stdio:") else source
            if normalized_ref in {server_name, source, basename}:
                match_name = server_name
                break

        if not match_name:
            return False, f"MCP server is not loaded: {normalized_ref}", None

        server_info = self.loaded_servers.pop(match_name)
        for tool_name in server_info.get("tool_names", []):
            self.loaded_tools.pop(tool_name, None)

        connection_key = server_info.get("connection_key", "")
        if connection_key:
            with self._connection_lock:
                connection = self._remote_connections.pop(connection_key, None)
            if connection is not None:
                connection.close()

        return True, f"Unloaded MCP server {match_name} with {len(server_info.get('tool_names', []))} tool(s).", server_info

    def refresh_server(self, server_ref: str) -> tuple[bool, str, Dict[str, Any] | None, list[StructuredTool]]:
        normalized_ref = (server_ref or "").strip()
        if not normalized_ref:
            return False, "Usage: refresh_mcp_server <server_name|source>", None, []

        match_info = None
        for server_name, server_info in self.loaded_servers.items():
            source = server_info.get("source", "")
            basename = os.path.basename(source) if source and not str(source).startswith("stdio:") else source
            if normalized_ref in {server_name, source, basename}:
                match_info = dict(server_info)
                break

        if not match_info:
            return False, f"MCP server is not loaded: {normalized_ref}", None, []

        source = match_info.get("source", normalized_ref)
        unload_success, unload_message, _ = self.unload_server(match_info["name"])
        if not unload_success:
            return False, unload_message, None, []

        return self.load_server(source)

    def close_all(self) -> None:
        with self._connection_lock:
            connections = list(self._remote_connections.values())
            self._remote_connections.clear()
        for connection in connections:
            try:
                connection.close()
            except Exception:
                pass