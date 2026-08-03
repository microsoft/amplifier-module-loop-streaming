# Amplifier Streaming Loop Orchestrator Module

Token-level streaming orchestration for real-time response delivery.

## Prerequisites

- **Python 3.11+**
- **[UV](https://github.com/astral-sh/uv)** - Fast Python package manager

### Installing UV

```bash
# macOS/Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Purpose

Provides streaming orchestration that delivers LLM responses token-by-token for improved perceived performance and user experience.

## Contract

**Module Type:** Orchestrator
**Mount Point:** `orchestrators`
**Entry Point:** `amplifier_module_loop_streaming:mount`

## Behavior

- Token-level streaming from provider
- Real-time response delivery
- **Parallel tool execution**: Multiple tool calls execute concurrently
- Deterministic context updates: Results added in original order
- Progressive rendering
- Interruptible generation

## Configuration

```toml
[[orchestrators]]
module = "loop-streaming"
name = "streaming"
config = {
    max_iterations = -1,             # Maximum iterations (-1 = unlimited, default)
    goal_stall_threshold = 3,        # /goal: consecutive no-tool continuation turns
                                      # before the stall judge is consulted
    goal_model_role = "fast",        # /goal: routing-matrix model role requested for
                                      # the evaluator/stall-judge/summary calls, via
                                      # the model_role_resolver coordinator capability
    stream_delay = 0.0,              # Per-token artificial delay (seconds), for
                                      # human-facing typing animation (0.0 = off)
    extended_thinking = false,       # Enable extended thinking on the main
                                      # conversational turns (not the /goal internal
                                      # calls, which always disable it)
    min_delay_between_calls_ms = 0,  # Minimum delay between provider calls (rate
                                      # limiting; 0 = disabled)
}
```

## Usage

```python
# In amplifier configuration
[session]
orchestrator = "loop-streaming"
```

Perfect for:

- Interactive CLI applications
- Web UIs with progressive rendering
- Long-form content generation

## Dependencies

- `amplifier-core>=1.0.0`

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
