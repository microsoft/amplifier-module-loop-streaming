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
from amplifier_core.events import ORCHESTRATOR_COMPLETE
from amplifier_core.events import PROMPT_SUBMIT
from amplifier_core.events import PROVIDER_REQUEST
from amplifier_core.events import TOOL_POST
from amplifier_core.events import TOOL_PRE
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import Message
from amplifier_core.message_models import ToolSpec

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the streaming orchestrator module."""
    config = config or {}

    # Declare observable lifecycle events for this module
    # (hooks-logging will auto-discover and log these)
    coordinator.register_contributor(
        "observability.events",
        "loop-streaming",
        lambda: [
            "session:start",  # When session begins
            "session:end",  # When session completes
            "context:pre-compact",  # Before context compaction
        ],
    )

    orchestrator = StreamingOrchestrator(config)
    await coordinator.mount("orchestrator", orchestrator)
    logger.info("Mounted StreamingOrchestrator with observable events")
    return


class StreamingOrchestrator:
    """
    Streaming implementation of the agent loop.
    Yields tokens as they're generated for real-time display.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        # -1 means unlimited iterations (default)
        max_iter_config = config.get("max_iterations", -1)
        self.max_iterations = int(max_iter_config) if max_iter_config != -1 else -1
        self.stream_delay = config.get("stream_delay", 0.01)  # Artificial delay for demo
        self.extended_thinking = config.get("extended_thinking", False)

    async def execute(
        self,
        prompt: str,
        context,
        providers: dict[str, Any],
        tools: dict[str, Any],
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None = None,
    ) -> str:
        """
        Execute with streaming - returns full response but could be modified to stream.

        Note: This is a simplified version. A real streaming implementation would
        need to modify the core interfaces to support AsyncIterator returns.
        """
        # For now, collect the stream and return as string
        # In a real implementation, the interface would support streaming
        full_response = ""
        iteration_count = 0

        async for token, iteration in self._execute_stream(prompt, context, providers, tools, hooks, coordinator):
            full_response += token
            iteration_count = iteration

        # Emit orchestrator complete event
        await hooks.emit(
            ORCHESTRATOR_COMPLETE,
            {
                "orchestrator": "loop-streaming",
                "turn_count": iteration_count,
                "status": "success" if full_response else "incomplete",
            },
        )

        return full_response

    async def _execute_stream(
        self,
        prompt: str,
        context,
        providers: dict[str, Any],
        tools: dict[str, Any],
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None = None,
    ) -> AsyncIterator[tuple[str, int]]:
        """
        Internal streaming execution.
        Yields tuples of (token, iteration) as they're generated.
        """
        # Emit and process prompt submit (allows hooks to inject context before processing)
        result = await hooks.emit(PROMPT_SUBMIT, {"prompt": prompt})
        if coordinator:
            result = await coordinator.process_hook_result(result, "prompt:submit", "orchestrator")
            if result.action == "deny":
                yield (f"Operation denied: {result.reason}", 0)
                return

        # Emit session start
        await hooks.emit("session:start", {"prompt": prompt})

        # Add user message
        await context.add_message({"role": "user", "content": prompt})

        # Select provider
        provider = self._select_provider(providers)
        if not provider:
            yield ("Error: No providers available", 0)
            return

        # Find provider name for event emission
        provider_name = None
        for name, prov in providers.items():
            if prov is provider:
                provider_name = name
                break

        iteration = 0

        while self.max_iterations == -1 or iteration < self.max_iterations:
            iteration += 1

            # Emit provider request BEFORE getting messages (allows hook injections)
            result = await hooks.emit(PROVIDER_REQUEST, {"provider": provider_name, "iteration": iteration})
            if coordinator:
                result = await coordinator.process_hook_result(result, "provider:request", "orchestrator")
                if result.action == "deny":
                    yield (f"Operation denied: {result.reason}", iteration)
                    return

            # Get messages (includes permanent injections)
            message_dicts = await context.get_messages()
            message_dicts = list(message_dicts)  # Convert to list for modification

            # Append ephemeral injection if present (temporary, not stored)
            if result.action == "inject_context" and result.ephemeral and result.context_injection:
                # Check if we should append to last tool result
                if result.append_to_last_tool_result and len(message_dicts) > 0:
                    last_msg = message_dicts[-1]
                    # Append to last message if it's a tool result
                    if last_msg.get("role") == "tool":
                        # Append to existing content
                        original_content = last_msg.get("content", "")
                        message_dicts[-1] = {
                            **last_msg,
                            "content": f"{original_content}\n\n{result.context_injection}",
                        }
                        logger.debug("Appended ephemeral injection to last tool result message")
                    else:
                        # Fall back to new message if last message isn't a tool result
                        message_dicts.append(
                            {"role": result.context_injection_role, "content": result.context_injection}
                        )
                        logger.debug(
                            f"Last message role is '{last_msg.get('role')}', not 'tool' - "
                            "created new message for injection"
                        )
                else:
                    # Default behavior: append as new message
                    message_dicts.append({"role": result.context_injection_role, "content": result.context_injection})

            # Convert dicts to ChatRequest for provider
            messages_objects = [Message(**msg) for msg in message_dicts]

            # Convert tools to ToolSpec format for ChatRequest
            tools_list = None
            if tools:
                tools_list = [
                    ToolSpec(name=t.name, description=t.description, parameters=t.input_schema) for t in tools.values()
                ]

            chat_request = ChatRequest(messages=messages_objects, tools=tools_list)
            logger.info(f"[ORCHESTRATOR] ChatRequest created with {len(tools_list) if tools_list else 0} tools")
            if tools_list:
                logger.debug(f"[ORCHESTRATOR] Tool names: {[t.name for t in tools_list]}")

            # Check if provider supports streaming
            if hasattr(provider, "stream"):
                # Use streaming if available
                async for chunk in self._stream_from_provider(provider, chat_request, context, tools, hooks):
                    yield (chunk, iteration)

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
                    # Build kwargs for provider
                    kwargs = {}
                    if self.extended_thinking:
                        kwargs["extended_thinking"] = True
                    response = await provider.complete(chat_request, **kwargs)

                    # Emit content block events if present
                    content_blocks = getattr(response, "content_blocks", None)
                    if content_blocks:
                        total_blocks = len(content_blocks)
                        for idx, block in enumerate(content_blocks):
                            # Emit block start
                            await hooks.emit(
                                CONTENT_BLOCK_START,
                                {
                                    "block_type": block.type.value,
                                    "block_index": idx,
                                    "total_blocks": total_blocks,
                                    "metadata": getattr(block, "raw", None),
                                },
                            )

                            # Emit block end with complete block, usage, and total count
                            event_data = {
                                "block_index": idx,
                                "total_blocks": total_blocks,
                                "block": block.to_dict(),
                            }
                            if response.usage:
                                event_data["usage"] = response.usage.model_dump()
                            await hooks.emit(CONTENT_BLOCK_END, event_data)

                    # Parse tool calls
                    tool_calls = provider.parse_tool_calls(response)

                    if not tool_calls:
                        # Extract text content from response for streaming
                        # Use .text field if available (e.g., OpenAI provider), otherwise extract from content blocks
                        if hasattr(response, "text") and response.text:
                            response_text = response.text
                        else:
                            response_text = self._extract_text_from_content(response.content)

                        # Stream the final response token by token
                        async for token in self._tokenize_stream(response_text):
                            yield (token, iteration)

                        # Store structured content from response.content (our Pydantic models)
                        # This preserves reasoning state, thinking blocks, etc.
                        # response.content = list of our ContentBlock models (TextBlock, ThinkingBlock, etc.)
                        # response.content_blocks = raw SDK objects (for streaming events only)
                        response_content = getattr(response, "content", None)
                        if response_content and isinstance(response_content, list):
                            # Convert ContentBlock objects to dicts for serialization
                            content_dicts = [
                                block.model_dump() if hasattr(block, "model_dump") else block
                                for block in response_content
                            ]
                            logger.info(f"[ORCHESTRATOR] Storing {len(content_dicts)} content blocks")
                            for i, block_dict in enumerate(content_dicts):
                                logger.info(
                                    f"[ORCHESTRATOR]   Block {i}: type={block_dict.get('type')}, has_content={'content' in block_dict}"
                                )
                            assistant_msg = {
                                "role": "assistant",
                                "content": content_dicts,
                            }
                        else:
                            assistant_msg = {"role": "assistant", "content": response_text}

                        # Preserve thinking blocks for Anthropic extended thinking (backward compat)
                        # Extract from response.content (Pydantic models) instead of content_blocks (streaming objects)
                        if response_content:
                            for block in response_content:
                                if hasattr(block, "type") and block.type == "thinking":
                                    # Store the full thinking block with signature
                                    assistant_msg["thinking_block"] = block.model_dump()
                                    break

                        # Preserve provider metadata (provider-agnostic passthrough)
                        # This enables providers to maintain state across steps (e.g., OpenAI reasoning items)
                        if hasattr(response, "metadata") and response.metadata:
                            assistant_msg["metadata"] = response.metadata

                        await context.add_message(assistant_msg)
                        break

                    # Add assistant message with tool calls
                    # Store structured content blocks (preserves reasoning state, thinking blocks, etc.)
                    # Extract text for display/logging only
                    if hasattr(response, "text") and response.text:
                        response_text = response.text
                    else:
                        response_text = self._extract_text_from_content(response.content) if response.content else ""

                    # Store structured content from response.content (our Pydantic models)
                    response_content = getattr(response, "content", None)
                    if response_content and isinstance(response_content, list):
                        assistant_msg = {
                            "role": "assistant",
                            "content": [
                                block.model_dump() if hasattr(block, "model_dump") else block
                                for block in response_content
                            ],
                            "tool_calls": [
                                {"id": tc.id, "tool": tc.name, "arguments": tc.arguments} for tc in tool_calls
                            ],
                        }
                    else:
                        assistant_msg = {
                            "role": "assistant",
                            "content": response_text,
                            "tool_calls": [
                                {"id": tc.id, "tool": tc.name, "arguments": tc.arguments} for tc in tool_calls
                            ],
                        }

                    # Preserve thinking blocks for Anthropic extended thinking (backward compat)
                    # Extract from response.content (Pydantic models) instead of content_blocks (streaming objects)
                    if response_content:
                        for block in response_content:
                            if hasattr(block, "type") and block.type == "thinking":
                                # Store the full thinking block with signature
                                assistant_msg["thinking_block"] = block.model_dump()
                                break

                    # Preserve provider metadata (provider-agnostic passthrough)
                    # This enables providers to maintain state across steps (e.g., OpenAI reasoning items)
                    if hasattr(response, "metadata") and response.metadata:
                        assistant_msg["metadata"] = response.metadata

                    await context.add_message(assistant_msg)

                    # Process tool calls in parallel (user guidance: assume parallel intent)
                    # Execute tools concurrently, but add results to context sequentially for determinism
                    import uuid

                    parallel_group_id = str(uuid.uuid4())

                    # Execute all tools in parallel (no context updates inside)
                    tool_tasks = [
                        self._execute_tool_only(tc, tools, hooks, parallel_group_id, coordinator) for tc in tool_calls
                    ]
                    tool_results = await asyncio.gather(*tool_tasks)

                    # Add all results to context in original order (sequential, deterministic)
                    for tool_call_id, content in tool_results:
                        await context.add_message(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": content,
                            }
                        )

                except Exception as e:
                    logger.error(f"Provider error: {e}")
                    yield (f"\nError: {e}", iteration)
                    break

            # Check compaction
            if await context.should_compact():
                await hooks.emit("context:pre-compact", {})
                await context.compact()

        # Check if we exceeded max iterations (only if not unlimited)
        if self.max_iterations != -1 and iteration >= self.max_iterations:
            logger.warning(f"Max iterations ({self.max_iterations}) reached")

            # Inject system reminder to agent before returning
            await hooks.emit(PROVIDER_REQUEST, {"provider": provider_name, "iteration": iteration, "max_reached": True})

            # Get one final response with the reminder (via _execute_stream helper)
            message_dicts = await context.get_messages()
            message_dicts = list(message_dicts)
            message_dicts.append(
                {
                    "role": "system",
                    "content": """<system-reminder>
You have reached the maximum number of iterations for this turn. Please provide a response to the user now, summarizing your progress and noting what remains to be done. You can continue in the next turn if needed.
</system-reminder>""",
                }
            )

            try:
                # Convert dicts to ChatRequest
                messages_objects = [Message(**msg) for msg in message_dicts]

                # Convert tools to ToolSpec format for ChatRequest
                tools_list = None
                if tools:
                    tools_list = [
                        ToolSpec(name=t.name, description=t.description, parameters=t.input_schema)
                        for t in tools.values()
                    ]

                max_iter_chat_request = ChatRequest(messages=messages_objects, tools=tools_list)

                kwargs = {}
                if self.extended_thinking:
                    kwargs["extended_thinking"] = True

                response = await provider.complete(max_iter_chat_request, **kwargs)
                content = response.content if hasattr(response, "content") else str(response)

                if content:
                    # Yield the final response
                    async for token in self._tokenize_stream(content):
                        yield (token, iteration)

                    # Add to context
                    await context.add_message({"role": "assistant", "content": content})

            except Exception as e:
                logger.error(f"Error getting final response after max iterations: {e}")

        # Emit session end
        await hooks.emit("session:end", {})

    async def _stream_from_provider(self, provider, chat_request, context, tools, hooks) -> AsyncIterator[str]:
        """Stream tokens from provider that supports streaming."""
        # This is a simplified example
        # Real implementation would handle streaming tool calls

        full_response = ""

        # Convert tools dict to list for provider
        tools_list = list(tools.values()) if tools else []
        async for chunk in provider.stream(chat_request, tools=tools_list):
            token = chunk.get("content", "")
            if token:
                yield token
                full_response += token
                await asyncio.sleep(self.stream_delay)  # Artificial delay for demo

        # Add complete message to context
        if full_response:
            await context.add_message({"role": "assistant", "content": full_response})

    def _extract_text_from_content(self, content) -> str:
        """Extract text from content blocks.

        Args:
            content: Either a string or list of ContentBlock objects

        Returns:
            Extracted text as string
        """
        if isinstance(content, str):
            return content

        if not content:
            return ""

        # Extract text from content blocks
        # NOTE: Only extract from TextBlock, NOT ThinkingBlock
        # Thinking blocks have visibility="internal" and are rendered separately in the UI
        # Including them here causes thinking text to appear in main response (duplication)
        text_parts = []
        for block in content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            # Skip thinking blocks - they're rendered separately
            # elif hasattr(block, "thinking"):
            #     text_parts.append(block.thinking)

        return "\n\n".join(text_parts)

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

    async def _execute_tool(
        self,
        tool_call,
        tools: dict[str, Any],
        context,
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None = None,
    ) -> None:
        """Execute a single tool call (legacy method for compatibility)."""
        await self._execute_tool_with_result(tool_call, tools, context, hooks, coordinator)

    async def _execute_tool_only(
        self,
        tool_call,
        tools: dict[str, Any],
        hooks: HookRegistry,
        parallel_group_id: str,
        coordinator: ModuleCoordinator | None = None,
    ) -> tuple[str, str]:
        """Execute a single tool in parallel without adding to context.

        Returns (tool_call_id, content) tuple.
        Never raises - errors become error messages.
        """
        try:
            # Pre-tool hook
            pre_result = await hooks.emit(
                TOOL_PRE,
                {
                    "tool_name": tool_call.name,
                    "tool_input": tool_call.arguments,
                    "parallel_group_id": parallel_group_id,
                },
            )
            if coordinator:
                pre_result = await coordinator.process_hook_result(pre_result, "tool:pre", tool_call.name)
                if pre_result.action == "deny":
                    return (tool_call.id, f"Denied by hook: {pre_result.reason}")

            # Get tool
            tool = tools.get(tool_call.name)
            if not tool:
                error_msg = f"Error: Tool '{tool_call.name}' not found"
                await hooks.emit(
                    "tool:error",
                    {
                        "tool": tool_call.name,
                        "error": {"type": "RuntimeError", "msg": error_msg},
                        "parallel_group_id": parallel_group_id,
                    },
                )
                return (tool_call.id, error_msg)

            # Execute
            try:
                result = await tool.execute(tool_call.arguments)
            except Exception as e:
                result = ToolResult(success=False, error={"message": str(e)})

            # Serialize result for logging
            result_data = result.model_dump() if hasattr(result, "model_dump") else str(result)

            # Post-tool hook
            post_result = await hooks.emit(
                TOOL_POST,
                {
                    "tool_name": tool_call.name,
                    "tool_input": tool_call.arguments,
                    "result": result_data,
                    "parallel_group_id": parallel_group_id,
                },
            )
            if coordinator:
                await coordinator.process_hook_result(post_result, "tool:post", tool_call.name)

            # Return result content
            content = str(result.output) if result.success else f"Error: {result.error}"
            return (tool_call.id, content)

        except Exception as e:
            # Safety net: errors become error messages
            logger.error(f"Tool {tool_call.name} failed: {e}")
            error_msg = f"Internal error executing tool: {str(e)}"
            await hooks.emit(
                "tool:error",
                {
                    "tool": tool_call.name,
                    "error": {"type": type(e).__name__, "msg": str(e)},
                    "parallel_group_id": parallel_group_id,
                },
            )
            return (tool_call.id, error_msg)

    async def _execute_tool_with_result(
        self,
        tool_call,
        tools: dict[str, Any],
        context,
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None = None,
    ) -> dict:
        """Execute a single tool call and return result info.

        Guarantees that a tool response is always added to context, even if errors occur.
        This prevents orphaned tool calls that corrupt conversation state.
        """
        response_added = False

        try:
            # Pre-tool hook
            pre_result = await hooks.emit(
                TOOL_PRE,
                {
                    "tool_name": tool_call.name,
                    "tool_input": tool_call.arguments,
                },
            )
            if coordinator:
                pre_result = await coordinator.process_hook_result(pre_result, "tool:pre", tool_call.name)
                if pre_result.action == "deny":
                    # Add tool_result message (not system) so Anthropic API accepts it
                    await context.add_message(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Tool execution denied: {pre_result.reason}",
                        }
                    )
                    response_added = True
                    return {"success": False, "error": f"Denied: {pre_result.reason}"}

            # Get tool
            tool = tools.get(tool_call.name)
            if not tool:
                # Add tool_result message (not system) so Anthropic API accepts it
                await context.add_message(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: Tool '{tool_call.name}' not found",
                    }
                )
                response_added = True
                return {"success": False, "error": "Tool not found"}

            # Execute
            try:
                result = await tool.execute(tool_call.arguments)
            except Exception as e:
                result = ToolResult(success=False, error={"message": str(e)})

            # Serialize result for logging
            result_data = result.model_dump() if hasattr(result, "model_dump") else str(result)

            # Post-tool hook
            post_result = await hooks.emit(
                TOOL_POST,
                {
                    "tool_name": tool_call.name,
                    "tool_input": tool_call.arguments,
                    "result": result_data,
                },
            )
            if coordinator:
                await coordinator.process_hook_result(post_result, "tool:post", tool_call.name)

            # Add result with tool_call_id
            await context.add_message(
                {
                    "role": "tool",
                    "name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "content": str(result.output) if result.success else f"Error: {result.error}",
                }
            )
            response_added = True

            return {"success": result.success, "error": result.error if not result.success else None}

        except Exception as e:
            # Safety net: Ensure a tool response is ALWAYS added to prevent orphaned tool calls
            logger.error(f"Unexpected error executing tool {tool_call.name}: {e}", exc_info=True)

            if not response_added:
                try:
                    await context.add_message(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Internal error executing tool: {str(e)}",
                        }
                    )
                except Exception as inner_e:
                    # Critical failure: Even adding error response failed
                    logger.error(f"Critical: Failed to add error response for tool_call_id {tool_call.id}: {inner_e}")

            return {"success": False, "error": str(e)}

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
