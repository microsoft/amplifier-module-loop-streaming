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
from amplifier_core.events import CONTENT_BLOCK_END
from amplifier_core.events import CONTENT_BLOCK_START

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
        self.extended_thinking = config.get("extended_thinking", False)

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
                    # Convert tools dict to list for provider
                    tools_list = list(tools.values()) if tools else []
                    # Build kwargs for provider
                    kwargs = {}
                    if tools_list:
                        kwargs["tools"] = tools_list
                    if self.extended_thinking:
                        kwargs["extended_thinking"] = True
                    response = await provider.complete(messages, **kwargs)

                    # Emit content block events if present
                    content_blocks = getattr(response, "content_blocks", None)
                    if content_blocks:
                        for idx, block in enumerate(content_blocks):
                            # Emit block start
                            await hooks.emit(
                                CONTENT_BLOCK_START,
                                {
                                    "data": {
                                        "block_type": block.type.value,
                                        "block_index": idx,
                                        "metadata": getattr(block, "raw", None),
                                    }
                                },
                            )

                            # Emit block end with complete block
                            await hooks.emit(
                                CONTENT_BLOCK_END, {"data": {"block_index": idx, "block": block.to_dict()}}
                            )

                    # Parse tool calls
                    tool_calls = provider.parse_tool_calls(response)

                    if not tool_calls:
                        # Stream the final response token by token
                        async for token in self._tokenize_stream(response.content):
                            yield token

                        # Build assistant message with thinking block if present
                        assistant_msg = {"role": "assistant", "content": response.content}

                        # Preserve thinking blocks for Anthropic extended thinking
                        if content_blocks:
                            for block in content_blocks:
                                if hasattr(block, "type") and block.type.value == "thinking":
                                    # Store the raw thinking block to preserve signature
                                    assistant_msg["thinking_block"] = block.raw if hasattr(block, "raw") else None
                                    break

                        await context.add_message(assistant_msg)
                        break

                    # Add assistant message with tool calls and thinking block
                    assistant_msg = {
                        "role": "assistant",
                        "content": response.content if response.content else "",
                        "tool_calls": [{"id": tc.id, "tool": tc.tool, "arguments": tc.arguments} for tc in tool_calls],
                    }

                    # Preserve thinking blocks for Anthropic extended thinking
                    if content_blocks:
                        for block in content_blocks:
                            if hasattr(block, "type") and block.type.value == "thinking":
                                # Store the raw thinking block to preserve signature
                                assistant_msg["thinking_block"] = block.raw if hasattr(block, "raw") else None
                                break

                    await context.add_message(assistant_msg)

                    # Process tool calls (display handled by streaming UI via tool:pre/post events)
                    for tool_call in tool_calls:
                        # Execute tool (hooks will display via tool:pre/post events)
                        await self._execute_tool_with_result(tool_call, tools, context, hooks)

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

        # Convert tools dict to list for provider
        tools_list = list(tools.values()) if tools else []
        async for chunk in provider.stream(messages, tools=tools_list):
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
        Simulate token-by-token streaming from complete text while preserving newlines.
        In production, this would be real streaming from the provider.
        """
        # Split by lines first to preserve newlines
        lines = text.split("\n")

        for line_idx, line in enumerate(lines):
            # Split line into words
            words = line.split()

            # Yield words with spaces
            for word_idx, word in enumerate(words):
                if word_idx > 0:
                    yield " "
                yield word
                await asyncio.sleep(self.stream_delay)

            # Yield newline after each line except the last
            if line_idx < len(lines) - 1:
                yield "\n"

    async def _execute_tool(self, tool_call, tools: dict[str, Any], context, hooks: HookRegistry) -> None:
        """Execute a single tool call (legacy method for compatibility)."""
        await self._execute_tool_with_result(tool_call, tools, context, hooks)

    async def _execute_tool_with_result(self, tool_call, tools: dict[str, Any], context, hooks: HookRegistry) -> dict:
        """Execute a single tool call and return result info."""
        # Pre-tool hook
        hook_result = await hooks.emit("tool:pre", {"tool": tool_call.tool, "arguments": tool_call.arguments})

        if hook_result.action == "deny":
            # Add tool_result message (not system) so Anthropic API accepts it
            await context.add_message(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Tool execution denied: {hook_result.reason}",
                }
            )
            return {"success": False, "error": f"Denied: {hook_result.reason}"}

        # Get tool
        tool = tools.get(tool_call.tool)
        if not tool:
            # Add tool_result message (not system) so Anthropic API accepts it
            await context.add_message(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error: Tool '{tool_call.tool}' not found",
                }
            )
            return {"success": False, "error": "Tool not found"}

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

        # Add result with tool_call_id
        await context.add_message(
            {
                "role": "tool",
                "name": tool_call.tool,
                "tool_call_id": tool_call.id,
                "content": str(result.output) if result.success else f"Error: {result.error}",
            }
        )

        return {"success": result.success, "error": result.error if not result.success else None}

    async def _has_pending_tools(self, context) -> bool:
        """Check if there are pending tool calls."""
        # Simplified - would need to track tool calls properly
        return False

    async def _process_tools(self, context, tools, hooks) -> None:
        """Process any pending tool calls."""
        # Simplified - would process tracked tool calls
        pass

    def _select_provider(self, providers: dict[str, Any]) -> Any:
        """Select a provider based on priority."""
        if not providers:
            return None

        # Collect providers with their priority (default priority is 100)
        provider_list = []
        for name, provider in providers.items():
            # Try to get priority from provider's config or attributes
            priority = 100  # Default priority
            if hasattr(provider, "priority"):
                priority = provider.priority
            elif hasattr(provider, "config") and isinstance(provider.config, dict):
                priority = provider.config.get("priority", 100)

            provider_list.append((priority, name, provider))

        # Sort by priority (lower number = higher priority)
        provider_list.sort(key=lambda x: x[0])

        # Return the highest priority provider
        if provider_list:
            return provider_list[0][2]

        return None
