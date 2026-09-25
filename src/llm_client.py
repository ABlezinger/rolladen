"""Shared helpers for talking to the LLM/embedding API (OpenAI-compatible base_url).

Centralizes:
- A sane default request timeout (the API sometimes never answers).
- Normalizing connection failures / timeouts / 5xx server errors into a single
  `LLMServiceError`, carrying the standard German user-facing message, so every
  call site can show the same message instead of crashing or failing silently.
"""

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
)
import streamlit as st

# Shown to the user whenever the LLM/embedding server can't be reached or errors out.
LLM_UNAVAILABLE_MESSAGE = (
    "Der Server für die Sprachmodelle ist momentan nicht verfügbar. "
    "Bitte versuchen Sie es später erneut"
)

# No call site previously set a timeout, so requests could hang indefinitely.
# 1.0s was too aggressive for real completions (planning/answering calls routinely
# take 2s+), causing spurious LLMServiceError on nearly every real chat turn.
DEFAULT_TIMEOUT = 60.0  # seconds
DEFAULT_MAX_RETRIES = 1


class LLMServiceError(Exception):
    """Raised when the LLM/embedding backend is unreachable or returns a server error (5xx)."""

    def __init__(self, message=LLM_UNAVAILABLE_MESSAGE):
        super().__init__(message)


def get_client(base_url, api_key, timeout=DEFAULT_TIMEOUT, max_retries=DEFAULT_MAX_RETRIES):
    """Build an OpenAI client with a default timeout/retry policy applied."""
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)


def _is_service_error(exc):
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return True
    return False


def safe_completion(client, **kwargs):
    """Call client.chat.completions.create, normalizing connectivity/5xx errors.

    Raises LLMServiceError (with the standard German message) when the server is
    unreachable, times out, or returns a 5xx status. Other errors (e.g. bad
    request, auth) propagate unchanged.
    """
    try:
        print("MODEL: ", kwargs["model"])
        print("Clients: ", client)
        response = client.chat.completions.create(**kwargs)
        st.session_state["server_available"] = True
        return response
    except (APIConnectionError, APITimeoutError) as e:
        st.session_state["server_available"] = False

        raise LLMServiceError() from e
    except APIStatusError as e:
        if e.status_code >= 500:
            st.session_state["server_available"] = False
            raise LLMServiceError() from e
        raise


def safe_embeddings_create(client, **kwargs):
    """Call client.embeddings.create, normalizing connectivity/5xx errors."""
    try:
        return client.embeddings.create(**kwargs)
    except (APIConnectionError, APITimeoutError) as e:
        raise LLMServiceError() from e
    except APIStatusError as e:
        if e.status_code >= 500:
            raise LLMServiceError() from e
        raise


def safe_stream_iter(stream):
    """Iterate a streaming completion, normalizing connectivity/5xx errors that
    occur while reading chunks (not just at request creation)."""
    try:
        for chunk in stream:
            yield chunk
    except (APIConnectionError, APITimeoutError) as e:
        raise LLMServiceError() from e
    except APIStatusError as e:
        if e.status_code >= 500:
            raise LLMServiceError() from e
        raise
