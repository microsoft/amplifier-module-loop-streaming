"""
Streaming orchestrator module for Amplifier.
Provides token-by-token streaming responses.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any
from typing import Optional

from amplifier_core import HookRegistry
from amplifier_core import ModuleCoordinator
from amplifier_core import ToolResult

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the streaming orchestrator module."""
    config = config or {}
    orchestrator = StreamingOrchestrator(config)
    await coordinator.mount("orchestrator", orchestrator)
    logger.info("Mounted StreamingOrchestrator")
    return


class StreamingOrchestrator:
    """
    Streaming implementation of the agent loop.
    Yields tokens as they're generated for real-time display.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.max_iterations = config.get("max_iterations", 50)
        self.stream_delay = config.get("stream_delay", 0.01)  # Artificial delay for demo

    async def execute(
        self, prompt: str, context, providers: dict[str, Any], tools: dict[str, Any], hooks: HookRegistry
    ) -> str:
        """
        Execute with streaming - returns full response but could be modified to stream.

        Note: This is a simplified version. A real streaming implementation would
        need to modify the core interfaces to support AsyncIterator returns.
        """
        # For now, collect the stream and return as string
        # In a real implementation, the interface would support streaming
        full_response = ""

        async for token in self._execute_stream(prompt, context, providers, tools, hooks):
            full_response += token

        return full_response

    async def _execute_stream(
        self, prompt: str, context, providers: dict[str, Any], tools: dict[str, Any], hooks: HookRegistry
    ) -> AsyncIterator[str]:
        """
        Internal streaming execution.
        Yields tokens as they're generated.
        """
        # Emit session start
        await hooks.emit("session:start", {"prompt": prompt})

        # Add user message
        await context.add_message({"role": "user", "content": prompt})

        # Select provider
        provider = self._select_provider(providers)
        if not provider:
            yield "Error: No providers available"
            return

        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # Get messages
            messages = await context.get_messages()

            # Check if provider supports streaming
            if hasattr(provider, "stream"):
                # Use streaming if available
                async for chunk in self._stream_from_provider(provider, messages, context, tools, hooks):
                    yield chunk

                # Check for tool calls after streaming
                # This is simplified - real implementation would parse during stream
                if await self._has_pending_tools(context):
                    # Process tools
                    await self._process_tools(context, tools, hooks)
                    continue
                else:
                    # No more tools, we're done
                    break
            else:
                # Fallback to non-streaming
                try:
                    response = await provider.complete(messages)

                    # Parse tool calls
                    tool_calls = provider.parse_tool_calls(response)

                    if not tool_calls:
                        # Stream the final response token by token
                        async for token in self._tokenize_stream(response.content):
                            yield token

                        await context.add_message({"role": "assistant", "content": response.content})
                        break

                    # Process tool calls
                    for tool_call in tool_calls:
                        await self._execute_tool(tool_call, tools, context, hooks)

                except Exception as e:
                    logger.error(f"Provider error: {e}")
                    yield f"\nError: {e}"
                    break

            # Check compaction
            if await context.should_compact():
                await hooks.emit("context:pre-compact", {})
                await context.compact()

        # Emit session end
        await hooks.emit("session:end", {})

    async def _stream_from_provider(self, provider, messages, context, tools, hooks) -> AsyncIterator[str]:
        """Stream tokens from provider that supports streaming."""
        # This is a simplified example
        # Real implementation would handle streaming tool calls

        full_response = ""

        async for chunk in provider.stream(messages):
            token = chunk.get("content", "")
            if token:
                yield token
                full_response += token
                await asyncio.sleep(self.stream_delay)  # Artificial delay for demo

        # Add complete message to context
        if full_response:
            await context.add_message({"role": "assistant", "content": full_response})

    async def _tokenize_stream(self, text: str) -> AsyncIterator[str]:
        """
        Simulate token-by-token streaming from complete text.
        In production, this would be real streaming from the provider.
        """
        # Split into words for demo
        tokens = text.split()

        for i, token in enumerate(tokens):
            if i > 0:
                yield " "
            yield token
            await asyncio.sleep(self.stream_delay)

    async def _execute_tool(self, tool_call, tools: dict[str, Any], context, hooks: HookRegistry) -> None:
        """Execute a single tool call."""
        # Pre-tool hook
        hook_result = await hooks.emit("tool:pre", {"tool": tool_call.tool, "arguments": tool_call.arguments})

        if hook_result.action == "deny":
            await context.add_message(
                {"role": "system", "content": f"Tool {tool_call.tool} denied: {hook_result.reason}"}
            )
            return

        # Get tool
        tool = tools.get(tool_call.tool)
        if not tool:
            await context.add_message({"role": "system", "content": f"Tool {tool_call.tool} not found"})
            return

        # Execute
        try:
            result = await tool.execute(tool_call.arguments)
        except Exception as e:
            result = ToolResult(success=False, error={"message": str(e)})

        # Post-tool hook
        await hooks.emit(
            "tool:post",
            {"tool": tool_call.tool, "result": result.model_dump() if hasattr(result, "model_dump") else str(result)},
        )

        # Add result
        await context.add_message(
            {
                "role": "tool",
                "name": tool_call.tool,
                "content": str(result.output) if result.success else f"Error: {result.error}",
            }
        )

    async def _has_pending_tools(self, context) -> bool:
        """Check if there are pending tool calls."""
        # Simplified - would need to track tool calls properly
        return False

    async def _process_tools(self, context, tools, hooks) -> None:
        """Process any pending tool calls."""
        # Simplified - would process tracked tool calls
        pass

    def _select_provider(self, providers: dict[str, Any]) -> Any:
        """Select a provider."""
        if not providers:
            return None

        # Prefer providers that support streaming
        for _name, provider in providers.items():
            if hasattr(provider, "stream"):
                return provider

        # Fallback to first available
        return next(iter(providers.values()))
