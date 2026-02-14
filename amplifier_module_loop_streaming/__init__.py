"""
Streaming orchestrator module for Amplifier.
Provides token-by-token streaming responses.
"""

# Amplifier module metadata
__amplifier_module_type__ = "orchestrator"

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from amplifier_core import HookRegistry
from amplifier_core import ModuleCoordinator
from amplifier_core import ToolResult
from amplifier_core.events import CONTENT_BLOCK_END
from amplifier_core.events import CONTENT_BLOCK_START
from amplifier_core.events import ORCHESTRATOR_COMPLETE
from amplifier_core.events import PROMPT_SUBMIT
from amplifier_core.events import PROVIDER_ERROR
from amplifier_core.events import PROVIDER_REQUEST
from amplifier_core.events import TOOL_ERROR
from amplifier_core.events import TOOL_POST
from amplifier_core.events import TOOL_PRE
from amplifier_core.llm_errors import LLMError
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
            "execution:start",  # When orchestrator execution begins
            "execution:end",  # When orchestrator execution completes
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
        self.stream_delay = config.get(
            "stream_delay", 0.01
        )  # Artificial delay for demo
        self.extended_thinking = config.get("extended_thinking", False)
        self.min_delay_between_calls_ms = config.get("min_delay_between_calls_ms", 0)
        self._last_provider_call_end: float | None = None  # Timestamp tracking
        # Store ephemeral injections from tool:post hooks for next iteration
        self._pending_ephemeral_injections: list[dict[str, Any]] = []

    async def _apply_rate_limit_delay(
        self, hooks: HookRegistry, iteration: int
    ) -> None:
        """Apply rate limit delay if configured and needed.

        Only delays if:
        - min_delay_between_calls_ms > 0 (enabled)
        - This is not the first call (has previous timestamp)
        - Elapsed time < configured minimum
        """
        if self.min_delay_between_calls_ms <= 0:
            return  # Disabled

        if self._last_provider_call_end is None:
            return  # First call, no delay needed

        elapsed_ms = (time.monotonic() - self._last_provider_call_end) * 1000
        remaining_ms = self.min_delay_between_calls_ms - elapsed_ms

        if remaining_ms > 0:
            await hooks.emit(
                "orchestrator:rate_limit_delay",
                {
                    "delay_ms": remaining_ms,
                    "configured_ms": self.min_delay_between_calls_ms,
                    "elapsed_ms": elapsed_ms,
                    "iteration": iteration,
                },
            )
            await asyncio.sleep(remaining_ms / 1000)

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
        full_response = ""
        iteration_count = 0
        error: Exception | None = None

        try:
            async for token, iteration in self._execute_stream(
                prompt, context, providers, tools, hooks, coordinator
            ):
                full_response += token
                iteration_count = iteration
        except Exception as e:
            error = e

        # Always emit orchestrator complete event (observability)
        if error:
            status = "error"
        elif coordinator and coordinator.cancellation.is_cancelled:
            status = "cancelled"
        else:
            status = "success" if full_response else "incomplete"

        await hooks.emit(
            ORCHESTRATOR_COMPLETE,
            {
                "orchestrator": "loop-streaming",
                "turn_count": iteration_count,
                "status": status,
            },
        )

        if error:
            raise error

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
        prompt_submit_result = await hooks.emit(PROMPT_SUBMIT, {"prompt": prompt})
        if coordinator:
            prompt_submit_result = await coordinator.process_hook_result(
                prompt_submit_result, "prompt:submit", "orchestrator"
            )
            if prompt_submit_result.action == "deny":
                yield (f"Operation denied: {prompt_submit_result.reason}", 0)
                return

        # Store ephemeral injection from prompt:submit for use in the loop
        # (must be stored before provider:request overwrites 'result')
        if (
            prompt_submit_result.action == "inject_context"
            and prompt_submit_result.ephemeral
            and prompt_submit_result.context_injection
        ):
            self._pending_ephemeral_injections.append(
                {
                    "role": prompt_submit_result.context_injection_role,
                    "content": prompt_submit_result.context_injection,
                    "append_to_last_tool_result": prompt_submit_result.append_to_last_tool_result,
                }
            )
            logger.debug(
                "Stored ephemeral injection from prompt:submit for first iteration"
            )

        # Emit execution start
        await hooks.emit("execution:start", {"prompt": prompt})

        # Reset rate limit tracking for new session
        self._last_provider_call_end = None

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
            # Check for cancellation at iteration start
            if coordinator and coordinator.cancellation.is_cancelled:
                # Don't yield more content, just exit
                return

            iteration += 1

            # Emit provider request BEFORE getting messages (allows hook injections)
            result = await hooks.emit(
                PROVIDER_REQUEST, {"provider": provider_name, "iteration": iteration}
            )
            if coordinator:
                result = await coordinator.process_hook_result(
                    result, "provider:request", "orchestrator"
                )
                if result.action == "deny":
                    yield (f"Operation denied: {result.reason}", iteration)
                    return

            # Get messages for LLM request (context handles compaction internally)
            # Pass provider for dynamic budget calculation based on model's context window
            message_dicts = await context.get_messages_for_request(provider=provider)
            message_dicts = list(message_dicts)  # Convert to list for modification

            # Append ephemeral injection if present (temporary, not stored)
            if (
                result.action == "inject_context"
                and result.ephemeral
                and result.context_injection
            ):
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
                        logger.debug(
                            "Appended ephemeral injection to last tool result message"
                        )
                    else:
                        # Fall back to new message if last message isn't a tool result
                        message_dicts.append(
                            {
                                "role": result.context_injection_role,
                                "content": result.context_injection,
                            }
                        )
                        logger.debug(
                            f"Last message role is '{last_msg.get('role')}', not 'tool' - "
                            "created new message for injection"
                        )
                else:
                    # Default behavior: append as new message
                    message_dicts.append(
                        {
                            "role": result.context_injection_role,
                            "content": result.context_injection,
                        }
                    )

            # Apply pending ephemeral injections from tool:post hooks
            if self._pending_ephemeral_injections:
                for injection in self._pending_ephemeral_injections:
                    if (
                        injection.get("append_to_last_tool_result")
                        and len(message_dicts) > 0
                    ):
                        last_msg = message_dicts[-1]
                        if last_msg.get("role") == "tool":
                            original_content = last_msg.get("content", "")
                            message_dicts[-1] = {
                                **last_msg,
                                "content": f"{original_content}\n\n{injection['content']}",
                            }
                            logger.debug(
                                "Applied pending ephemeral injection to last tool result"
                            )
                        else:
                            message_dicts.append(
                                {
                                    "role": injection["role"],
                                    "content": injection["content"],
                                }
                            )
                            logger.debug(
                                "Last message not a tool result, created new message for injection"
                            )
                    else:
                        message_dicts.append(
                            {"role": injection["role"], "content": injection["content"]}
                        )
                        logger.debug(
                            "Applied pending ephemeral injection as new message"
                        )
                # Clear pending injections after applying
                self._pending_ephemeral_injections = []

            # Convert dicts to ChatRequest for provider
            messages_objects = [Message(**msg) for msg in message_dicts]

            # Convert tools to ToolSpec format for ChatRequest
            tools_list = None
            if tools:
                tools_list = [
                    ToolSpec(
                        name=t.name,
                        description=t.description,
                        parameters=t.input_schema,
                    )
                    for t in tools.values()
                ]

            chat_request = ChatRequest(
                messages=messages_objects,
                tools=tools_list,
                reasoning_effort=self.config.get("reasoning_effort"),
            )
            logger.info(
                f"[ORCHESTRATOR] ChatRequest created with {len(tools_list) if tools_list else 0} tools"
            )
            if tools_list:
                logger.debug(
                    f"[ORCHESTRATOR] Tool names: {[t.name for t in tools_list]}"
                )

            # Apply rate limit delay before provider call
            await self._apply_rate_limit_delay(hooks, iteration)

            # Check if provider supports streaming
            if hasattr(provider, "stream"):
                # Use streaming if available
                async for chunk in self._stream_from_provider(
                    provider,
                    chat_request,
                    context,
                    tools,
                    hooks,
                    coordinator,
                    provider_name=provider_name,
                ):
                    # Check for immediate cancellation between chunks
                    if coordinator and coordinator.cancellation.is_immediate:
                        return
                    yield (chunk, iteration)

                # Update rate limit timestamp after streaming completes
                self._last_provider_call_end = time.monotonic()

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
                # Build kwargs for provider
                kwargs = {}
                if self.extended_thinking:
                    kwargs["extended_thinking"] = True
                try:
                    response = await provider.complete(chat_request, **kwargs)
                except LLMError as e:
                    await hooks.emit(
                        PROVIDER_ERROR,
                        {
                            "provider": provider_name,
                            "error": {"type": type(e).__name__, "msg": str(e)},
                            "retryable": e.retryable,
                            "status_code": e.status_code,
                        },
                    )
                    raise
                except Exception as e:
                    await hooks.emit(
                        PROVIDER_ERROR,
                        {
                            "provider": provider_name,
                            "error": {"type": type(e).__name__, "msg": str(e)},
                        },
                    )
                    raise

                # Update rate limit timestamp after non-streaming response
                self._last_provider_call_end = time.monotonic()

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
                        response_text = self._extract_text_from_content(
                            response.content
                        )

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
                            block.model_dump()
                            if hasattr(block, "model_dump")
                            else block
                            for block in response_content
                        ]
                        logger.info(
                            f"[ORCHESTRATOR] Storing {len(content_dicts)} content blocks"
                        )
                        for i, block_dict in enumerate(content_dicts):
                            logger.info(
                                f"[ORCHESTRATOR]   Block {i}: type={block_dict.get('type')}, has_content={'content' in block_dict}"
                            )
                        assistant_msg = {
                            "role": "assistant",
                            "content": content_dicts,
                        }
                    else:
                        assistant_msg = {
                            "role": "assistant",
                            "content": response_text,
                        }

                    # Preserve thinking blocks for Anthropic extended thinking (backward compat)
                    # Use response_content (our Pydantic models) not content_blocks (raw SDK objects)
                    if response_content and isinstance(response_content, list):
                        for block in response_content:
                            block_type = getattr(block, "type", None)
                            type_value = (
                                getattr(block_type, "value", block_type)
                                if block_type
                                else None
                            )
                            if type_value == "thinking":
                                # Store the thinking block as dict to preserve signature
                                assistant_msg["thinking_block"] = (
                                    block.model_dump()
                                    if hasattr(block, "model_dump")
                                    else None
                                )
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
                    response_text = (
                        self._extract_text_from_content(response.content)
                        if response.content
                        else ""
                    )

                # --- P1 FIX: Yield intermediate text before tool execution ---
                # The no-tool-calls branch (line 459) yields text via
                # _tokenize_stream(). This branch was missing that step,
                # causing intermediate text to be silently dropped.
                if response_text:
                    async for token in self._tokenize_stream(response_text):
                        yield (token, iteration)
                # --- END P1 FIX ---

                # Store structured content from response.content (our Pydantic models)
                response_content = getattr(response, "content", None)
                if response_content and isinstance(response_content, list):
                    assistant_msg = {
                        "role": "assistant",
                        "content": [
                            block.model_dump()
                            if hasattr(block, "model_dump")
                            else block
                            for block in response_content
                        ],
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "tool": tc.name,
                                "arguments": tc.arguments,
                            }
                            for tc in tool_calls
                        ],
                    }
                else:
                    assistant_msg = {
                        "role": "assistant",
                        "content": response_text,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "tool": tc.name,
                                "arguments": tc.arguments,
                            }
                            for tc in tool_calls
                        ],
                    }

                # Preserve thinking blocks for Anthropic extended thinking (backward compat)
                # Use response_content (our Pydantic models) not content_blocks (raw SDK objects)
                if response_content and isinstance(response_content, list):
                    for block in response_content:
                        block_type = getattr(block, "type", None)
                        type_value = (
                            getattr(block_type, "value", block_type)
                            if block_type
                            else None
                        )
                        if type_value == "thinking":
                            # Store the thinking block as dict to preserve signature
                            assistant_msg["thinking_block"] = (
                                block.model_dump()
                                if hasattr(block, "model_dump")
                                else None
                            )
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
                # Wrap in try/except for CancelledError to handle immediate cancellation
                tool_tasks = [
                    self._execute_tool_only(
                        tc, tools, hooks, parallel_group_id, coordinator
                    )
                    for tc in tool_calls
                ]

                try:
                    tool_results = await asyncio.gather(*tool_tasks)
                except asyncio.CancelledError:
                    # Immediate cancellation (second Ctrl+C) - synthesize cancelled results
                    # for ALL tool_calls to maintain tool_use/tool_result pairing
                    logger.info(
                        "Tool execution cancelled - synthesizing cancelled results"
                    )
                    for tc in tool_calls:
                        await context.add_message(
                            {
                                "role": "tool",
                                "name": tc.name,
                                "tool_call_id": tc.id,
                                "content": f'{{"error": "Tool execution was cancelled by user", "cancelled": true, "tool": "{tc.name}"}}',
                            }
                        )
                    # Re-raise to let the cancellation propagate
                    raise

                # Check for cancellation after tools complete (graceful cancellation)
                if coordinator and coordinator.cancellation.is_cancelled:
                    # MUST add tool results to context before returning
                    # Otherwise we leave orphaned tool_calls without matching tool_results
                    # which violates provider API contracts (Anthropic, OpenAI)
                    for tool_call_id, tool_name, content in tool_results:
                        await context.add_message(
                            {
                                "role": "tool",
                                "name": tool_name,
                                "tool_call_id": tool_call_id,
                                "content": content,
                            }
                        )
                    # Exit the loop - orchestrator complete event will be emitted in execute()
                    return

                # Add all results to context in original order (sequential, deterministic)
                # Note: Context manager handles compaction internally when get_messages_for_request() is called
                for tool_call_id, tool_name, content in tool_results:
                    await context.add_message(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "tool_call_id": tool_call_id,
                            "content": content,
                        }
                    )

        # Check if we exceeded max iterations (only if not unlimited)
        if self.max_iterations != -1 and iteration >= self.max_iterations:
            logger.warning(f"Max iterations ({self.max_iterations}) reached")

            # Inject system reminder to agent before returning
            await hooks.emit(
                PROVIDER_REQUEST,
                {
                    "provider": provider_name,
                    "iteration": iteration,
                    "max_reached": True,
                },
            )

            # Get one final response with the reminder (via _execute_stream helper)
            message_dicts = await context.get_messages_for_request(provider=provider)
            message_dicts = list(message_dicts)
            message_dicts.append(
                {
                    "role": "user",
                    "content": """<system-reminder source="orchestrator-loop-limit">
You have reached the maximum number of iterations for this turn. Please provide a response to the user now, summarizing your progress and noting what remains to be done. You can continue in the next turn if needed.

DO NOT mention this iteration limit or reminder to the user explicitly. Simply wrap up naturally.
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
                        ToolSpec(
                            name=t.name,
                            description=t.description,
                            parameters=t.input_schema,
                        )
                        for t in tools.values()
                    ]

                max_iter_chat_request = ChatRequest(
                    messages=messages_objects,
                    tools=tools_list,
                    reasoning_effort=self.config.get("reasoning_effort"),
                )

                kwargs = {}
                if self.extended_thinking:
                    kwargs["extended_thinking"] = True

                response = await provider.complete(max_iter_chat_request, **kwargs)
                content = (
                    response.content if hasattr(response, "content") else str(response)
                )

                if content:
                    # Yield the final response
                    async for token in self._tokenize_stream(content):
                        yield (token, iteration)

                    # Add to context
                    await context.add_message({"role": "assistant", "content": content})

            except LLMError as e:
                await hooks.emit(
                    PROVIDER_ERROR,
                    {
                        "provider": provider_name,
                        "error": {"type": type(e).__name__, "msg": str(e)},
                        "retryable": e.retryable,
                        "status_code": e.status_code,
                    },
                )
                logger.error(f"Error getting final response after max iterations: {e}")
            except Exception as e:
                await hooks.emit(
                    PROVIDER_ERROR,
                    {
                        "provider": provider_name,
                        "error": {"type": type(e).__name__, "msg": str(e)},
                    },
                )
                logger.error(f"Error getting final response after max iterations: {e}")

        # Emit execution end
        await hooks.emit("execution:end", {})

    async def _stream_from_provider(
        self,
        provider,
        chat_request,
        context,
        tools,
        hooks,
        coordinator=None,
        provider_name=None,
    ) -> AsyncIterator[str]:
        """Stream tokens from provider that supports streaming.

        Args:
            provider: The provider to stream from
            chat_request: The chat request to send
            context: The context manager
            tools: Available tools
            hooks: Hook registry
            coordinator: Optional coordinator for cancellation support
            provider_name: Name of the provider for event emission
        """
        # This is a simplified example
        # Real implementation would handle streaming tool calls

        full_response = ""

        # Convert tools dict to list for provider
        tools_list = list(tools.values()) if tools else []
        try:
            stream_iter = provider.stream(chat_request, tools=tools_list)
        except LLMError as e:
            await hooks.emit(
                PROVIDER_ERROR,
                {
                    "provider": provider_name,
                    "error": {"type": type(e).__name__, "msg": str(e)},
                    "retryable": e.retryable,
                    "status_code": e.status_code,
                },
            )
            raise
        except Exception as e:
            await hooks.emit(
                PROVIDER_ERROR,
                {
                    "provider": provider_name,
                    "error": {"type": type(e).__name__, "msg": str(e)},
                },
            )
            raise

        async for chunk in stream_iter:
            # Check for immediate cancellation between chunks
            if coordinator and coordinator.cancellation.is_immediate:
                # Add partial response to context before exiting
                if full_response:
                    await context.add_message(
                        {"role": "assistant", "content": full_response}
                    )
                return

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
        Simulate token-by-token streaming from complete text while preserving whitespace.
        In production, this would be real streaming from the provider.

        Preserves:
        - Leading whitespace (critical for code block indentation)
        - Multiple consecutive spaces (critical for ASCII art alignment)
        - Newlines between lines
        """
        lines = text.split("\n")

        for line_idx, line in enumerate(lines):
            # Split into tokens while PRESERVING all whitespace runs
            # \S+ matches non-whitespace sequences, \s+ matches whitespace sequences
            # This ensures multiple spaces are preserved (e.g., for ASCII art tables)
            tokens = re.findall(r"\S+|\s+", line)

            for token in tokens:
                yield token
                # Only delay on non-whitespace tokens for natural streaming effect
                if token.strip():
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
        await self._execute_tool_with_result(
            tool_call, tools, context, hooks, coordinator
        )

    async def _execute_tool_only(
        self,
        tool_call,
        tools: dict[str, Any],
        hooks: HookRegistry,
        parallel_group_id: str,
        coordinator: ModuleCoordinator | None = None,
    ) -> tuple[str, str, str]:
        """Execute a single tool in parallel without adding to context.

        Returns (tool_call_id, name, content) tuple.
        Never raises - errors become error messages.
        """
        try:
            # Pre-tool hook
            pre_result = await hooks.emit(
                TOOL_PRE,
                {
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "tool_input": tool_call.arguments,
                    "parallel_group_id": parallel_group_id,
                },
            )
            if coordinator:
                pre_result = await coordinator.process_hook_result(
                    pre_result, "tool:pre", tool_call.name
                )
                if pre_result.action == "deny":
                    return (
                        tool_call.id,
                        tool_call.name,
                        f"Denied by hook: {pre_result.reason}",
                    )

            # Get tool
            tool = tools.get(tool_call.name)
            if not tool:
                error_msg = f"Error: Tool '{tool_call.name}' not found"
                await hooks.emit(
                    TOOL_ERROR,
                    {
                        "tool_name": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "error": {"type": "RuntimeError", "msg": error_msg},
                        "parallel_group_id": parallel_group_id,
                    },
                )
                return (tool_call.id, tool_call.name, error_msg)

            # Register tool with cancellation token for visibility
            if coordinator:
                coordinator.cancellation.register_tool_start(
                    tool_call.id, tool_call.name
                )

            # Execute
            try:
                result = await tool.execute(tool_call.arguments)
            except Exception as e:
                result = ToolResult(success=False, error={"message": str(e)})
            finally:
                # Always unregister tool from cancellation token
                if coordinator:
                    coordinator.cancellation.register_tool_complete(tool_call.id)

            # Serialize result for logging
            result_data = (
                result.model_dump() if hasattr(result, "model_dump") else str(result)
            )

            # Post-tool hook
            post_result = await hooks.emit(
                TOOL_POST,
                {
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "tool_input": tool_call.arguments,
                    "result": result_data,
                    "parallel_group_id": parallel_group_id,
                },
            )
            if coordinator:
                await coordinator.process_hook_result(
                    post_result, "tool:post", tool_call.name
                )

            # Store ephemeral injection from tool:post for next iteration
            if (
                post_result.action == "inject_context"
                and post_result.ephemeral
                and post_result.context_injection
            ):
                self._pending_ephemeral_injections.append(
                    {
                        "role": post_result.context_injection_role,
                        "content": post_result.context_injection,
                        "append_to_last_tool_result": post_result.append_to_last_tool_result,
                    }
                )
                logger.debug(
                    f"Stored ephemeral injection from tool:post ({tool_call.name}) for next iteration"
                )

            # Check if a hook modified the tool result.
            # hooks.emit() chains modify actions: when a hook
            # returns action="modify", the data dict is replaced.
            # We detect this by checking if the returned "result"
            # is a different object than what we originally sent.
            modified_result = None
            if post_result and post_result.data is not None:
                returned_result = post_result.data.get("result")
                if returned_result is not None and returned_result is not result_data:
                    modified_result = returned_result

            if modified_result is not None:
                if isinstance(modified_result, (dict, list)):
                    content = json.dumps(modified_result)
                else:
                    content = str(modified_result)
            else:
                content = result.get_serialized_output()
            return (tool_call.id, tool_call.name, content)

        except Exception as e:
            # Safety net: errors become error messages
            logger.error(f"Tool {tool_call.name} failed: {e}")
            error_msg = f"Internal error executing tool: {str(e)}"
            await hooks.emit(
                TOOL_ERROR,
                {
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "error": {"type": type(e).__name__, "msg": str(e)},
                    "parallel_group_id": parallel_group_id,
                },
            )
            return (tool_call.id, tool_call.name, error_msg)

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
                    "tool_call_id": tool_call.id,
                    "tool_input": tool_call.arguments,
                },
            )
            if coordinator:
                pre_result = await coordinator.process_hook_result(
                    pre_result, "tool:pre", tool_call.name
                )
                if pre_result.action == "deny":
                    # Add tool_result message (not system) so Anthropic API accepts it
                    await context.add_message(
                        {
                            "role": "tool",
                            "name": tool_call.name,
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
                        "name": tool_call.name,
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
            result_data = (
                result.model_dump() if hasattr(result, "model_dump") else str(result)
            )

            # Post-tool hook
            post_result = await hooks.emit(
                TOOL_POST,
                {
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "tool_input": tool_call.arguments,
                    "result": result_data,
                },
            )
            if coordinator:
                await coordinator.process_hook_result(
                    post_result, "tool:post", tool_call.name
                )

            # Store ephemeral injection from tool:post for next iteration
            if (
                post_result.action == "inject_context"
                and post_result.ephemeral
                and post_result.context_injection
            ):
                self._pending_ephemeral_injections.append(
                    {
                        "role": post_result.context_injection_role,
                        "content": post_result.context_injection,
                        "append_to_last_tool_result": post_result.append_to_last_tool_result,
                    }
                )
                logger.debug(
                    f"Stored ephemeral injection from tool:post ({tool_call.name}) for next iteration"
                )

            # Check if a hook modified the tool result.
            # hooks.emit() chains modify actions: when a hook
            # returns action="modify", the data dict is replaced.
            # We detect this by checking if the returned "result"
            # is a different object than what we originally sent.
            modified_result = None
            if post_result and post_result.data is not None:
                returned_result = post_result.data.get("result")
                if returned_result is not None and returned_result is not result_data:
                    modified_result = returned_result

            if modified_result is not None:
                if isinstance(modified_result, (dict, list)):
                    tool_content = json.dumps(modified_result)
                else:
                    tool_content = str(modified_result)
            else:
                tool_content = result.get_serialized_output()

            await context.add_message(
                {
                    "role": "tool",
                    "name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "content": tool_content,
                }
            )
            response_added = True

            return {
                "success": result.success,
                "error": result.error if not result.success else None,
            }

        except Exception as e:
            # Safety net: Ensure a tool response is ALWAYS added to prevent orphaned tool calls
            logger.error(
                f"Unexpected error executing tool {tool_call.name}: {e}", exc_info=True
            )

            if not response_added:
                try:
                    await context.add_message(
                        {
                            "role": "tool",
                            "name": tool_call.name,
                            "tool_call_id": tool_call.id,
                            "content": f"Internal error executing tool: {str(e)}",
                        }
                    )
                except Exception as inner_e:
                    # Critical failure: Even adding error response failed
                    logger.error(
                        f"Critical: Failed to add error response for tool_call_id {tool_call.id}: {inner_e}"
                    )

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
