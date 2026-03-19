"""
Pytest configuration for module tests.

Behavioral tests use inheritance from amplifier-core base classes.
See tests/test_behavioral.py for the inherited tests.

The amplifier-core pytest plugin provides fixtures automatically:
- module_path: Detected path to this module
- module_type: Detected type (provider, tool, hook, etc.)
- provider_module, tool_module, etc.: Mounted module instances
"""

import importlib
import sys
import types

# ---------------------------------------------------------------------------
# Stub amplifier_core.llm_errors when the Rust-backed build is not available.
#
# The full amplifier-core wheel (built by maturin) ships llm_errors.py as a
# Python submodule.  The pure-Python fallback installed from git via
# ``uv sync`` does NOT include it.  Without this stub, any test that imports
# StreamingOrchestrator (which imports LLMError at module level) fails during
# collection with ``ModuleNotFoundError: No module named
# 'amplifier_core.llm_errors'``.
# ---------------------------------------------------------------------------

if "amplifier_core.llm_errors" not in sys.modules:
    try:
        importlib.import_module("amplifier_core.llm_errors")
    except (ImportError, ModuleNotFoundError):
        _stub = types.ModuleType("amplifier_core.llm_errors")
        _stub.__doc__ = "Test stub for amplifier_core.llm_errors"

        # -- Minimal error hierarchy (mirrors the real module) --

        class LLMError(Exception):
            def __init__(
                self,
                message="",
                *,
                provider=None,
                model=None,
                status_code=None,
                retryable=False,
                retry_after=None,
                delay_multiplier=1.0,
            ):
                super().__init__(message)
                self.provider = provider
                self.model = model
                self.status_code = status_code
                self.retryable = retryable
                self.retry_after = retry_after
                self.delay_multiplier = delay_multiplier

        class RateLimitError(LLMError):
            def __init__(self, message="", *, retryable=True, **kw):
                super().__init__(message, retryable=retryable, **kw)

        class AuthenticationError(LLMError):
            pass

        class ContextLengthError(LLMError):
            pass

        class ContentFilterError(LLMError):
            pass

        class InvalidRequestError(LLMError):
            pass

        class ProviderUnavailableError(LLMError):
            def __init__(self, message="", *, retryable=True, **kw):
                super().__init__(message, retryable=retryable, **kw)

        class LLMTimeoutError(LLMError):
            def __init__(self, message="", *, retryable=True, **kw):
                super().__init__(message, retryable=retryable, **kw)

        class NotFoundError(LLMError):
            pass

        class StreamError(LLMError):
            def __init__(self, message="", *, retryable=True, **kw):
                super().__init__(message, retryable=retryable, **kw)

        class AbortError(LLMError):
            pass

        class InvalidToolCallError(LLMError):
            def __init__(self, message="", *, tool_name=None, raw_arguments=None, **kw):
                super().__init__(message, **kw)
                self.tool_name = tool_name
                self.raw_arguments = raw_arguments

        class ConfigurationError(LLMError):
            pass

        class AccessDeniedError(AuthenticationError):
            pass

        class NetworkError(ProviderUnavailableError):
            pass

        class QuotaExceededError(RateLimitError):
            def __init__(self, message="", *, retryable=False, **kw):
                super().__init__(message, retryable=retryable, **kw)

        # Populate module namespace
        for _cls in [
            LLMError,
            RateLimitError,
            AuthenticationError,
            ContextLengthError,
            ContentFilterError,
            InvalidRequestError,
            ProviderUnavailableError,
            LLMTimeoutError,
            NotFoundError,
            StreamError,
            AbortError,
            InvalidToolCallError,
            ConfigurationError,
            AccessDeniedError,
            NetworkError,
            QuotaExceededError,
        ]:
            setattr(_stub, _cls.__name__, _cls)

        sys.modules["amplifier_core.llm_errors"] = _stub
