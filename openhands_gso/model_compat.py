"""Model compatibility fixes for the OpenHands runner."""

import importlib

EXTRA_PATTERNS = ["gpt-6*", "claude-fable-*"]
_LIST_NAMES = (
    "REASONING_EFFORT_PATTERNS",
    "REASONING_EFFORT_MODELS",
    "FUNCTION_CALLING_PATTERNS",
    "PROMPT_CACHE_PATTERNS",
)
# Fallback prices in USD/token from the OpenRouter catalog used for these runs.
_MODELS = {
    "openai/gpt-5.6-sol": (2e-06, 1e-05, 1050000),
    "openai/gpt-6-astra": (1e-05, 5e-05, 1050000),
    "anthropic/claude-fable-5": (1e-05, 5e-05, 1000000),
    "anthropic/claude-fable-5.1": (1e-05, 5e-05, 1000000),
}


def _register_litellm_models(litellm_mod):
    """Register capabilities and pricing missing from the LiteLLM model catalog."""
    entries = {}
    for name, (cin, cout, ctx) in _MODELS.items():
        provider, bare = name.split("/", 1)
        meta = {
            "max_tokens": 64000,
            "max_input_tokens": ctx,
            "max_output_tokens": 64000,
            "input_cost_per_token": cin,
            "output_cost_per_token": cout,
            "litellm_provider": provider,
            "mode": "responses" if provider == "openai" else "chat",
            "supports_reasoning": True,
            "supports_function_calling": True,
            "supports_prompt_caching": True,
        }
        for key in (name, bare):
            entries[key] = dict(meta)
        entries["openrouter/" + name] = dict(
            meta, litellm_provider="openrouter", mode="chat"
        )
        if provider == "openai":
            for key in ("openai/responses/" + bare, "responses/" + bare):
                # A mode on the routing alias stops LiteLLM stripping responses/.
                litellm_mod.model_cost.pop(key, None)
                entries[key] = dict(meta, mode=None)
    litellm_mod.register_model(entries)
    _wrap_completion_for_responses(litellm_mod)
    _merge_responses_bridge_choices(litellm_mod)
    _reasoning_roundtrip(litellm_mod)
    _reasoning_roundtrip_openrouter(litellm_mod)
    litellm_mod._gso_model_compat_installed = True


def _wrap_completion_for_responses(litellm_mod):
    """Preserve reasoning effort when using the Responses API bridge."""
    original = getattr(litellm_mod, "completion", None)
    if original is None or getattr(original, "_gso_responses_patch", False):
        return

    def completion(*args, **kwargs):
        model = kwargs.get("model") or (args[0] if args else "")
        effort = kwargs.get("reasoning_effort")
        if effort and isinstance(model, str) and ("/responses/" in model):
            kwargs.pop("reasoning_effort", None)
            kwargs.setdefault("reasoning", {"effort": effort})
        return original(*args, **kwargs)

    completion._gso_responses_patch = True
    litellm_mod.completion = completion


def _widen_retry_exceptions(retry_mixin_mod):
    """Retry provider API errors that LiteLLM does not classify more specifically."""
    try:
        import litellm
    except Exception:
        return
    cls = getattr(retry_mixin_mod, "RetryMixin", None)
    orig = getattr(cls, "retry_decorator", None) if cls else None
    if orig is None or getattr(orig, "_gso_widened", False):
        return

    def retry_decorator(self, **kwargs):
        exc = tuple(kwargs.get("retry_exceptions") or ())
        if litellm.APIError not in exc:
            kwargs["retry_exceptions"] = exc + (litellm.APIError,)
        return orig(self, **kwargs)

    retry_decorator._gso_widened = True
    cls.retry_decorator = retry_decorator


def _merge_responses_bridge_choices(litellm_mod):
    """Combine assistant text and tool calls into the single choice OpenHands expects."""
    try:
        from litellm.completion_extras.litellm_responses_transformation.transformation import (
            LiteLLMResponsesTransformationHandler as _H,
        )
    except Exception:
        return
    raw = _H.__dict__.get("_convert_response_output_to_choices")
    if raw is None:
        return
    orig = raw.__func__ if isinstance(raw, staticmethod) else raw
    if getattr(orig, "_gso_single_choice", False):
        return

    def _single_choice(*args, **kwargs):
        choices = orig(*args, **kwargs)
        if not choices or len(choices) == 1:
            return choices
        keep = next(
            (c for c in choices if getattr(c.message, "tool_calls", None)), choices[0]
        )
        texts, items = ([], [])
        for c in choices:
            t = getattr(c.message, "content", None)
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
            items.extend(getattr(c.message, "reasoning_items", None) or [])
        if texts:
            keep.message.content = "\n\n".join(texts)
        if items:
            keep.message.reasoning_items = items
        keep.index = 0
        return [keep]

    _single_choice._gso_single_choice = True
    _H._convert_response_output_to_choices = staticmethod(_single_choice)


_REASONING_BY_TOOLCALL = {}
_REASONING_CACHE_MAX = 4000


def _extract_reasoning_items(message):
    items = getattr(message, "reasoning_items", None)
    if items is None and isinstance(message, dict):
        items = message.get("reasoning_items")
    return items or []


def _tool_call_ids(message):
    tcs = getattr(message, "tool_calls", None)
    if tcs is None and isinstance(message, dict):
        tcs = message.get("tool_calls")
    ids = []
    for tc in tcs or []:
        tid = getattr(tc, "id", None) or (
            tc.get("id") if isinstance(tc, dict) else None
        )
        if tid:
            ids.append(tid)
    return ids


def _reasoning_roundtrip(litellm_mod):
    """Preserve Responses API reasoning items across tool calls."""
    original = getattr(litellm_mod, "completion", None)
    if original is None or getattr(original, "_gso_reasoning_roundtrip", False):
        return

    def completion(*args, **kwargs):
        model = kwargs.get("model") or (args[0] if args else "")
        is_responses = isinstance(model, str) and "/responses/" in model
        if is_responses and _REASONING_BY_TOOLCALL:
            for msg in kwargs.get("messages") or []:
                if not isinstance(msg, dict) or msg.get("role") != "assistant":
                    continue
                if msg.get("reasoning_items"):
                    continue
                for tid in _tool_call_ids(msg):
                    cached = _REASONING_BY_TOOLCALL.get(tid)
                    if cached:
                        msg["reasoning_items"] = cached
                        break
        resp = original(*args, **kwargs)
        if is_responses:
            try:
                for choice in getattr(resp, "choices", None) or []:
                    m = getattr(choice, "message", None)
                    items = _extract_reasoning_items(m)
                    if not items:
                        continue
                    for tid in _tool_call_ids(m):
                        _REASONING_BY_TOOLCALL[tid] = items
                if len(_REASONING_BY_TOOLCALL) > _REASONING_CACHE_MAX:
                    for k in list(_REASONING_BY_TOOLCALL)[
                        : len(_REASONING_BY_TOOLCALL) // 2
                    ]:
                        _REASONING_BY_TOOLCALL.pop(k, None)
            except Exception:
                pass
        return resp

    completion._gso_reasoning_roundtrip = True
    litellm_mod.completion = completion


_OR_REASONING_BY_TOOLCALL = {}


def _extract_reasoning_details(message):
    psf = getattr(message, "provider_specific_fields", None)
    if psf is None and isinstance(message, dict):
        psf = message.get("provider_specific_fields")
    details = psf.get("reasoning_details") if isinstance(psf, dict) else None
    if not details:
        details = getattr(message, "reasoning_details", None)
        if details is None and isinstance(message, dict):
            details = message.get("reasoning_details")
    return details or []


def _reasoning_roundtrip_openrouter(litellm_mod):
    """Preserve signed OpenRouter reasoning details across tool calls."""
    original = getattr(litellm_mod, "completion", None)
    if original is None or getattr(original, "_gso_or_reasoning_roundtrip", False):
        return

    def completion(*args, **kwargs):
        model = kwargs.get("model") or (args[0] if args else "")
        is_or = isinstance(model, str) and model.startswith("openrouter/")
        if is_or and _OR_REASONING_BY_TOOLCALL:
            for msg in kwargs.get("messages") or []:
                if not isinstance(msg, dict) or msg.get("role") != "assistant":
                    continue
                if msg.get("reasoning_details"):
                    continue
                for tid in _tool_call_ids(msg):
                    cached = _OR_REASONING_BY_TOOLCALL.get(tid)
                    if cached:
                        msg["reasoning_details"] = cached
                        break
        resp = original(*args, **kwargs)
        if is_or:
            try:
                for choice in getattr(resp, "choices", None) or []:
                    m = getattr(choice, "message", None)
                    details = _extract_reasoning_details(m)
                    if not details:
                        continue
                    for tid in _tool_call_ids(m):
                        _OR_REASONING_BY_TOOLCALL[tid] = details
                if len(_OR_REASONING_BY_TOOLCALL) > _REASONING_CACHE_MAX:
                    for k in list(_OR_REASONING_BY_TOOLCALL)[
                        : len(_OR_REASONING_BY_TOOLCALL) // 2
                    ]:
                        _OR_REASONING_BY_TOOLCALL.pop(k, None)
            except Exception:
                pass
        return resp

    completion._gso_or_reasoning_roundtrip = True
    litellm_mod.completion = completion


def install_model_compatibility():
    """Apply the compatibility fixes before creating an OpenHands LLM."""
    import litellm

    # OpenHands binds litellm.completion during import, so wrap it first.
    if not getattr(litellm, "_gso_model_compat_installed", False):
        _register_litellm_models(litellm)

    for module_name in (
        "openhands.llm.model_features",
        "openhands.sdk.llm.utils.model_features",
    ):
        try:
            features = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in _LIST_NAMES:
            patterns = getattr(features, name, None)
            if isinstance(patterns, list):
                for pattern in EXTRA_PATTERNS:
                    if pattern not in patterns:
                        patterns.append(pattern)

    try:
        retry_mixin = importlib.import_module("openhands.llm.retry_mixin")
    except ImportError:
        pass
    else:
        _widen_retry_exceptions(retry_mixin)
