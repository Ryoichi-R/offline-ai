"""Observational smoke tracing: fixed codes, field types and timings; never text.

Install only in the disposable single-case subprocess. Wrappers preserve call
arguments, return values, exceptions and inspect.signature via functools.wraps.
"""

from contextlib import contextmanager
from functools import wraps
import time

ERROR_CODES = frozenset(
    {
        "timeout",
        "stall",
        "cancelled",
        "internal_error",
        "service_unavailable",
        "prompt_budget",
        "model_error",
        "extraction_unsupported",
        "relevance_disputed",
        "relevance_uncertain",
        "invalid_request",
        "evidence_budget",
    }
)
ERROR_TYPES = frozenset(
    {
        "ValueError",
        "TypeError",
        "KeyError",
        "JSONDecodeError",
        "TimeoutError",
        "HTTPError",
        "URLError",
        "CancelledError",
        "DeepWorkerError",
        "ExtractionRejected",
        "DeepBudgetExpired",
        "DeepExplorationFinished",
        "ModelConfigurationError",
        "PromptBudgetError",
        "RuntimeError",
        "OSError",
    }
)
FIELDS = (
    "subject",
    "scope",
    "relevance",
    "conditions",
    "exceptions",
    "references",
    "supported",
    "contradictions",
    "missing_conditions",
    "unsupported_claims",
    "queries",
    "unresolved",
    "source_kind",
    "source_actor",
    "claim_actor",
)


def safe_code(value):
    return value if isinstance(value, str) and value in ERROR_CODES else "other"


def safe_failure(exc):
    name = type(exc).__name__
    return {
        "exceptionType": name if name in ERROR_TYPES else "OtherError",
        "errorCode": safe_code(getattr(exc, "code", None)),
    }


def structure(data):
    """Only fixed schema keys; neither arbitrary keys nor values are retained."""
    result = {}
    if not isinstance(data, dict):
        return {"object": False}
    for key in FIELDS:
        if key not in data:
            result[key] = {"type": "missing"}
            continue
        value = data[key]
        kind = type(value).__name__
        row = {
            "type": kind
            if kind in {"str", "list", "bool", "int", "float", "dict", "NoneType"}
            else "other"
        }
        if isinstance(value, (str, list, dict)):
            row["length"] = len(value)
        if type(value) is bool:
            row["value"] = value
        if (
            key == "relevance"
            and isinstance(value, str)
            and value in {"relevant", "irrelevant", "uncertain"}
        ):
            row["value"] = value
        result[key] = row
    return {"object": True, "fields": result}


@contextmanager
def observe(records):
    import deep_research
    import search
    import web_services

    originals = []

    def replace(module, name, wrapper):
        original = getattr(module, name)
        originals.append((module, name, original))
        setattr(module, name, wrapper(original))

    def timed(label):
        def decorate(original):
            @wraps(original)
            def wrapper(*args, **kwargs):
                row = {
                    "stage": label,
                    "startedOffsetSeconds": round(time.monotonic() - started, 3),
                }
                records.append(row)
                begin = time.monotonic()
                if isinstance(kwargs.get("timeout"), (int, float)):
                    row["timeoutSeconds"] = round(kwargs["timeout"], 3)
                try:
                    value = original(*args, **kwargs)
                    row["outcome"] = "returned"
                    if label == "deep.model_call" and isinstance(value, str):
                        row["responseChars"] = len(value)
                    return value
                except Exception as exc:
                    row.update(outcome="raised", **safe_failure(exc))
                    raise
                finally:
                    row["elapsedSeconds"] = round(time.monotonic() - begin, 3)

            return wrapper

        return decorate

    def json_stage(original):
        @wraps(original)
        def wrapper(model, system, user, budget, validator, *, reserve):
            names = {
                "_validate_section_json": "section",
                "_verification": "verification",
                "_gaps": "gaps",
                "_line_verdict": "line_verdict",
                "_relevance_verdict": "relevance",
            }
            row = {
                "stage": "deep.json_stage",
                "validator": names.get(validator.__name__, "other"),
                "attempts": [],
            }
            records.append(row)

            @wraps(validator)
            def checked(data):
                attempt = structure(data)
                row["attempts"].append(attempt)
                try:
                    value = validator(data)
                    attempt["valid"] = True
                    if type(value) is bool:
                        attempt["verdict"] = value
                    return value
                except Exception as exc:
                    attempt.update(valid=False, **safe_failure(exc))
                    raise

            try:
                return original(model, system, user, budget, checked, reserve=reserve)
            except Exception as exc:
                row.update(**safe_failure(exc))
                raise

        return wrapper

    started = time.monotonic()
    try:
        replace(deep_research, "_json_stage", json_stage)
        replace(deep_research, "_call_ollama_content", timed("deep.model_call"))
        for name in (
            "create_search_plan",
            "extract_keywords_via_llm",
            "verify_evidence_support",
        ):
            replace(search, name, timed("search." + name))
        replace(search, "run_retrieval_pipeline", timed("cli.retrieval"))
        replace(web_services, "run_retrieval_pipeline", timed("web.retrieval"))
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)
