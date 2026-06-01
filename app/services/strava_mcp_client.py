"""Strava MCP client — replaces direct REST API calls with MCP tool calls.

Spwans the Strava MCP server (@r-huijts/strava-mcp-server) as a subprocess
and communicates via the Model Context Protocol (JSON-RPC over stdio).
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_NPX_PATH = os.environ.get("NPX_PATH", "npx")


class McpError(Exception):
    """Raised when an MCP tool call returns an error."""


class StravaMcpClient:
    """MCP client for the Strava MCP server.

    Manages a subprocess and communicates via JSON-RPC 2.0 over stdio.
    Uses asyncio subprocess streams directly.
    """

    def __init__(self, command: str = _NPX_PATH, args: Optional[list[str]] = None):
        self._command = command
        self._args = args or ["-y", "@r-huijts/strava-mcp-server"]
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._read_task: Optional[asyncio.Task] = None
        self._connected = False

    async def connect(self):
        """Start the MCP server subprocess and perform initialization handshake."""
        if self._connected:
            return

        logger.info(f"Starting MCP server: {self._command} {' '.join(self._args)}")
        self._proc = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # asyncio.create_subprocess_exec returns StreamWriter for stdin
        # and StreamReader for stdout when using PIPE
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        # Background reader task
        self._read_task = asyncio.create_task(self._read_loop())

        # Initialize session per MCP spec
        init_result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "cycling-training-app", "version": "1.0.0"},
        })
        logger.info(f"MCP server initialized: {init_result.get('serverInfo', {})}")

        # Send initialized notification
        await self._send_notification("notifications/initialized", {})
        self._connected = True

    async def _read_loop(self):
        """Read JSON-RPC messages from stdout and route to pending futures."""
        try:
            while self._proc and self._proc.returncode is None:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                if not line_str:
                    continue
                try:
                    msg = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    future = self._pending.pop(msg_id)
                    if not future.cancelled():
                        if "error" in msg:
                            err = msg["error"]
                            future.set_exception(
                                McpError(f"MCP error ({err.get('code', 0)}): {err.get('message', '')}")
                            )
                        else:
                            future.set_result(msg.get("result", {}))
                # Ignore notifications for now
        except Exception as e:
            if self._proc and self._proc.returncode is None:
                logger.warning(f"MCP read loop error: {e}")

    async def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and wait for the response."""
        if not self._proc or not self._proc.stdin:
            raise McpError("MCP server not connected")

        self._request_id += 1
        req_id = self._request_id
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future

        payload = json.dumps(request) + "\n"
        self._proc.stdin.write(payload.encode())
        await self._proc.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise McpError(f"MCP request timed out: {method}")

    async def _send_notification(self, method: str, params: dict):
        """Send a JSON-RPC notification (no response expected)."""
        if not self._proc or not self._proc.stdin:
            return
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        payload = json.dumps(request) + "\n"
        self._proc.stdin.write(payload.encode())
        await self._proc.stdin.drain()

    async def call_tool(self, name: str, arguments: dict = None) -> Any:
        """Call an MCP tool and return its result."""
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })
        content = result.get("content", [])
        if content and isinstance(content, list) and content[0].get("type") == "text":
            text = content[0].get("text", "")
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return text
        return result

    async def close(self):
        """Shut down the MCP server connection."""
        self._connected = False
        if self._read_task:
            self._read_task.cancel()
            self._read_task = None
        if self._proc:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._proc.kill()
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    pass
            self._proc = None


# ── Singleton ──

_client_instance: Optional[StravaMcpClient] = None
_client_lock = asyncio.Lock()


async def _get_client() -> StravaMcpClient:
    """Get or create the shared MCP client (singleton)."""
    global _client_instance
    if _client_instance is None:
        async with _client_lock:
            if _client_instance is None:
                client = StravaMcpClient()
                await client.connect()
                _client_instance = client
    return _client_instance


# ── Public API ──


async def get_athlete_profile() -> Optional[Dict[str, Any]]:
    """Get the authenticated athlete's profile via MCP."""
    try:
        client = await _get_client()
        status = await client.call_tool("check-strava-connection", {})
        if isinstance(status, dict) and status.get("connected") is False:
            logger.warning("Strava not connected")
            return None
        return await client.call_tool("get-athlete-profile", {})
    except Exception as e:
        logger.error(f"Failed to get athlete profile via MCP: {e}")
        return None


async def get_activities(
    page: int = 1,
    per_page: int = 30,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Fetch activities via MCP."""
    try:
        client = await _get_client()
        arguments: Dict[str, Any] = {
            "page": page,
            "per_page": min(per_page, 200),
        }
        if after:
            arguments["after"] = after.isoformat()
        if before:
            arguments["before"] = before.isoformat()

        result = await client.call_tool("get-all-activities", arguments)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "activities" in result:
            return result["activities"]
        return []
    except Exception as e:
        logger.error(f"Failed to fetch activities via MCP (page {page}): {e}")
        return []


async def check_connection() -> bool:
    """Check if Strava is connected via MCP."""
    try:
        client = await _get_client()
        result = await client.call_tool("check-strava-connection", {})
        if isinstance(result, dict):
            return result.get("connected", False)
        return False
    except Exception as e:
        logger.error(f"Failed to check Strava connection: {e}")
        return False


async def fetch_all_recent_activities(
    after: datetime,
    activity_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch all activities after a given date, paginating automatically."""
    all_activities: List[Dict[str, Any]] = []
    page = 1

    while True:
        activities = await get_activities(page=page, per_page=50, after=after)
        if not activities:
            break
        if activity_types:
            activities = [
                a for a in activities
                if a.get("type") in activity_types
            ]
        all_activities.extend(activities)
        page += 1
        if len(activities) < 50:
            break

    return all_activities


async def connect_strava() -> Dict[str, Any]:
    """Initiate Strava OAuth connection via MCP (browser-based)."""
    try:
        client = await _get_client()
        result = await client.call_tool("connect-strava", {})
        return result if isinstance(result, dict) else {}
    except Exception as e:
        logger.error(f"Failed to connect Strava via MCP: {e}")
        return {"error": str(e)}


async def disconnect_strava() -> Dict[str, Any]:
    """Disconnect Strava via MCP."""
    try:
        client = await _get_client()
        result = await client.call_tool("disconnect-strava", {})
        return result if isinstance(result, dict) else {}
    except Exception as e:
        logger.error(f"Failed to disconnect Strava via MCP: {e}")
        return {"error": str(e)}
