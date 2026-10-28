"""
Part C — Support Ticket Triage with Guardrails
Run from project root:
    python triage.py

Required input files:
    data/raw/tickets.csv
    outputs/customer_segments.csv
    mock_llm.py
"""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import mock_llm
from src.audit_data import RAW_DIR

# Configuration
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

TICKETS_FILE = Path(RAW_DIR / "tickets.csv")
SEGMENTS_FILE = Path("outputs/customer_segments.csv")
OUTPUT_DIR = Path("outputs")

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")

ZERO_SHOT_PROMPT = """CLASSIFY the support ticket into exactly one allowed category.
Allowed categories: billing, delivery, technical, fraud.
Reply with JSON only, using exactly this shape:
{"category": "...", "reason": "..."}
The category must be one of the four allowed categories. The reason should be brief.
Do not include markdown or any text outside the JSON object."""

FEW_SHOT_PROMPT = """CLASSIFY the support ticket into exactly one allowed category.
Allowed categories: billing, delivery, technical, fraud.
Reply with JSON only, using exactly this shape:
{"category": "...", "reason": "..."}
The category must be one of the four allowed categories. The reason should be brief.
Do not include markdown or any text outside the JSON object.
EXAMPLES:
Example 1 — Billing: "I was charged twice for my order." ->
{"category": "billing", "reason": "The customer reports a duplicate charge."}
Example 2 — Delivery: "My package has not arrived." ->
{"category": "delivery", "reason": "The customer reports a missing delivery."}
Example 3 — Fraud: "I did not make this purchase and suspect account fraud." ->
{"category": "fraud", "reason": "The customer disputes an unauthorized purchase."}"""

def est_tokens(text: str) -> int:
    """Use the estimator from mock_llm if available; otherwise use a fallback."""
    estimator = getattr(mock_llm, "est_tokens", None)
    if callable(estimator):
        return int(estimator(text))
    # Fallback only: replace this with your assignment's exact estimator if supplied.
    return max(1, (len(text) + 3) // 4) if text else 0

def redact_text(text: str) -> tuple[str, dict[str, int]]:
    """Redact in the required order and return substitution counts."""
    counts: dict[str, int] = {}
    text, counts["email"] = EMAIL_PATTERN.subn("[EMAIL]", text)
    text, counts["phone"] = PHONE_PATTERN.subn("[PHONE]", text)
    text, counts["order_id"] = ORDER_PATTERN.subn("[ORDER_ID]", text)
    return text, counts

def residual_sensitive_matches(text: str) -> dict[str, int]:
    """Count any sensitive patterns left after redaction."""
    return {
        "email": len(EMAIL_PATTERN.findall(text)),
        "phone": len(PHONE_PATTERN.findall(text)),
        "order_id": len(ORDER_PATTERN.findall(text)),
    }

def get_text(value: Any) -> str:
    return value if isinstance(value, str) else ""

def call_mock(system_prompt: str, redacted_text: str) -> Any:
    """
    Adapter for the course mock. The user message content is exactly redacted_text.
    If mock_llm uses a different public function/signature, update this adapter only.
    """
    # Common OpenAI-style mock interface:
    if callable(getattr(mock_llm, "chat", None)):
        return mock_llm.chat(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": redacted_text}],
        )
    if callable(getattr(mock_llm, "complete", None)):
        return mock_llm.complete(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": redacted_text}],
        )
    if callable(getattr(mock_llm, "call", None)):
        return mock_llm.call(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": redacted_text}],
        )
    raise AttributeError(
        "Could not find mock_llm.chat, mock_llm.complete, or mock_llm.call. "
        "Inspect mock_llm.py and update call_mock() to match its API."
    )

def response_text(response: Any) -> str:
    """Extract text from common mock response formats."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "content" in response:
            return str(response["content"])
        if "text" in response:
            return str(response["text"])
        choices = response.get("choices")
        if choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message", {})
                if isinstance(message, dict) and "content" in message:
                    return str(message["content"])
                if "text" in choice:
                    return str(choice["text"])
    for attr in ("content", "text", "output_text"):
        if hasattr(response, attr):
            return str(getattr(response, attr))
    choices = getattr(response, "choices", None)
    if choices:
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is not None and hasattr(message, "content"):
            return str(message.content)
        if hasattr(choice, "text"):
            return str(choice.text)
    return str(response)

def response_tokens(response: Any) -> int:
    """Extract response.usage.total_tokens from dict- or object-style responses."""
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get("total_tokens", 0) or 0)
    return int(getattr(usage, "total_tokens", 0) or 0)

def get_transient_error_type():
    """Resolve the mock's transient exception without requiring a fixed class location."""
    candidate = getattr(mock_llm, "TransientLLMError", None)
    if isinstance(candidate, type) and issubclass(candidate, Exception):
        return candidate
    for value in vars(mock_llm).values():
        if isinstance(value, type) and issubclass(value, Exception) and value.__name__ == "TransientLLMError":
            return value
    return None

def call_with_retries(ticket_id: Any, system_prompt: str, redacted_text: str) -> Any:
    """Call model with at most 3 retries, jitter, and one log line per retry."""
    transient_error = get_transient_error_type()
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return call_mock(system_prompt, redacted_text)
        except Exception as exc:
            is_transient = (
                (transient_error is not None and isinstance(exc, transient_error))
                or exc.__class__.__name__ == "TransientLLMError"
            )
            if not is_transient or retry_number >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * (2 ** retry_number) * random.uniform(0.9, 1.15)
            print(
                f"RETRY ticket_id={ticket_id} retry_number={retry_number} "
                f"wait_seconds={wait:.4f}"
            )
            time.sleep(wait)
    raise RuntimeError("Unreachable retry state")

def validate_answer(raw_text: str) -> tuple[str, str | None, str]:
    """
    Returns (status, category, reason).
    Valid JSON with an unsupported category is invalid; do not keyword-fallback it.
    """
    try:
        payload = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", raw_text.lower())
        found = [word for word in words if word in ALLOWED_CATEGORIES]
        unique_found = set(found)
        if len(unique_found) == 1:
            return "classified_keyword", next(iter(unique_found)), ""
        return "flagged_invalid_output", None, ""

    if not isinstance(payload, dict):
        return "flagged_invalid_output", None, ""
    category = payload.get("category")
    if category not in ALLOWED_CATEGORIES:
        return "flagged_invalid_output", None, ""
    reason = payload.get("reason", "")
    return "classified_json", category, str(reason)

def read_segments() -> dict[str, dict[str, Any]]:
    """Return customer_id -> segment row; missing customers are non-premium."""
    if not SEGMENTS_FILE.exists():
        print(f"WARNING: {SEGMENTS_FILE} not found; treating all customers as not premium.")
        return {}
    segments = pd.read_csv(SEGMENTS_FILE, keep_default_na=False)
    if "customer_id" not in segments.columns:
        raise ValueError("customer_segments.csv must contain customer_id")
    # Normalize ID strings to avoid CSV numeric/string mismatches.
    return {str(row["customer_id"]): row.to_dict() for _, row in segments.iterrows()}

def is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}

def route_ticket(status: str, category: str | None, segment: dict[str, Any] | None) -> tuple[str, str]:
    if status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if segment and is_truthy(segment.get("is_premium", False)):
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"

def run_pipeline(system_prompt: str, *, token_budget: int | None = None, reset_mock: bool = True, label: str = "run",) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the triage pipeline end-to-end and return results and metrics."""
    tickets = pd.read_csv(TICKETS_FILE, keep_default_na=False)
    required = {"ticket_id", "customer_id", "text", "true_category"}
    missing = required - set(tickets.columns)
    if missing:
        raise ValueError(f"tickets.csv missing columns: {sorted(missing)}")

    if reset_mock:
        mock_llm.reset()  # Required immediately before each full run.

    segments = read_segments()
    rows: list[dict[str, Any]] = []
    total_tokens = 0
    redaction_totals = {"email": 0, "phone": 0, "order_id": 0}
    eligible_model_calls = 0
    transient_retries = 0

    for _, ticket in tickets.iterrows():
        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]
        original_text = get_text(ticket["text"])
        category: str | None = None
        segment = segments.get(str(customer_id))
        tokens_used = 0
        reason = ""

        # Input guardrails always run before redaction or any model call.
        if original_text.strip() == "":
            status = "skipped_empty"
        elif est_tokens(original_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            status = "budget_exhausted"
        else:
            redacted_text, counts = redact_text(original_text)
            for key, value in counts.items():
                redaction_totals[key] += value

            # Assertion: no specified sensitive pattern remains in prompt content.
            leftovers = residual_sensitive_matches(redacted_text)
            assert sum(leftovers.values()) == 0, (
                f"Sensitive information remains for ticket {ticket_id}: {leftovers}"
            )
            # The user message content passed to the model is exactly redacted_text.
            assert isinstance(redacted_text, str)

            before_retries = 0
            try:
                response = call_with_retries(ticket_id, system_prompt, redacted_text)
                eligible_model_calls += 1
            except Exception:
                # Transient failures are retried up to 3 times. Unexpected/permanent
                # failures should not silently be classified.
                status = "flagged_invalid_output"
                response = None

            if response is not None:
                tokens_used = response_tokens(response)
                total_tokens += tokens_used
                raw = response_text(response)
                status, category, reason = validate_answer(raw)
            else:
                tokens_used = 0

        route, priority = route_ticket(status, category, segment)
        rows.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category or "",
            "segment_name": (segment or {}).get("segment_name", ""),
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used,
            # Internal analysis columns used for metrics, removed from final CSV.
            "true_category": ticket["true_category"],
            "reason": reason,
        })

    results = pd.DataFrame(rows)
    classified = results[results["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((classified["category"] == classified["true_category"]).sum())
    classified_count = len(classified)
    metrics = {
        "label": label,
        "tickets_total": len(results),
        "classified": classified_count,
        "correct": correct,
        "accuracy": (correct / classified_count) if classified_count else 0.0,
        "total_tokens": total_tokens,
        "tokens_per_classified_ticket": (total_tokens / classified_count) if classified_count else 0.0,
        "status_counts": results["status"].value_counts().to_dict(),
        "redaction_totals": redaction_totals,
        "model_calls": eligible_model_calls,
    }
    return results, metrics

def print_metrics(metrics: dict[str, Any]) -> None:
    print(f"\n=== {metrics['label']} metrics ===")
    print(f"Tickets total: {metrics['tickets_total']}")
    print(f"Classified: {metrics['classified']}")
    print(f"Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['classified']})")
    print(f"Total tokens: {metrics['total_tokens']}")
    print(f"Tokens per classified ticket: {metrics['tokens_per_classified_ticket']:.2f}")
    print("Status counts:", metrics["status_counts"])
    print("Redaction substitutions:", metrics["redaction_totals"])

def save_final_csv(results: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "ticket_id", "customer_id", "status", "category",
        "segment_name", "priority", "route", "tokens_used"
    ]
    results[columns].to_csv(path, index=False)

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run A: zero-shot, no token budget.
    zero_results, zero_metrics = run_pipeline(
        ZERO_SHOT_PROMPT, token_budget=None, reset_mock=True, label="zero-shot"
    )
    print_metrics(zero_metrics)
    zero_results.to_csv(OUTPUT_DIR / "triage_zero_shot_analysis.csv", index=False)

    # Confusion matrix for classified zero-shot predictions.
    zero_classified = zero_results[
        zero_results["status"].isin(["classified_json", "classified_keyword"])
    ]
    confusion = pd.crosstab(
        zero_classified["true_category"],
        zero_classified["category"],
        rownames=["true_category"],
        colnames=["predicted_category"],
        dropna=False,
    )
    print("\nZero-shot confusion table:")
    print(confusion)

    # Run B: few-shot, no token budget.
    few_results, few_metrics = run_pipeline(
        FEW_SHOT_PROMPT, token_budget=None, reset_mock=True, label="few-shot"
    )
    print_metrics(few_metrics)
    few_results.to_csv(OUTPUT_DIR / "triage_few_shot_analysis.csv", index=False)

    # Run C: zero-shot budget demo. If 6000 does not exhaust budget, rerun at 5000.
    budget_demo, budget_metrics = run_pipeline(
        ZERO_SHOT_PROMPT, token_budget=6000, reset_mock=True, label="zero-shot-budget-6000"
    )
    exhausted = int((budget_demo["status"] == "budget_exhausted").sum())
    budget_used = 6000
    if exhausted == 0:
        budget_used = 5000
        budget_demo, budget_metrics = run_pipeline(
            ZERO_SHOT_PROMPT, token_budget=budget_used, reset_mock=True,
            label="zero-shot-budget-5000"
        )
        exhausted = int((budget_demo["status"] == "budget_exhausted").sum())
    save_final_csv(budget_demo, OUTPUT_DIR / "triage_budget_demo.csv")
    print_metrics(budget_metrics)
    print(f"Budget used: {budget_used}; budget_exhausted tickets: {exhausted}")
    assert exhausted >= 1, "Budget demo must exhaust at least one ticket."

    # Run D: final few-shot run with high budget and required routing.
    final_results, final_metrics = run_pipeline(
        FEW_SHOT_PROMPT, token_budget=100000, reset_mock=True, label="few-shot-final"
    )
    save_final_csv(final_results, OUTPUT_DIR / "triage_results.csv")
    print_metrics(final_metrics)
    assert len(final_results) == 164
    print("\nSaved outputs/triage_results.csv and outputs/triage_budget_demo.csv")

if __name__ == "__main__":
    main()
