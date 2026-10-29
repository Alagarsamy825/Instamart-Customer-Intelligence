"""
Part D — Refund Agent
Tasks 5–7: tool definitions, seven scenarios, safety demos, and crash/resume.

Run from the project root:
    python Refund_Agent_Tasks_5_to_7.py

This script uses the shared implementation in src/part_d.py.
Scenario runs append records to outputs/agent_runs.jsonl and may append to
outputs/refund_ledger.jsonl. Checkpoint files are written under state/.
"""
from pathlib import Path
import sys
import re
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
if not (ROOT / "src" / "part_d.py").exists():
    raise FileNotFoundError("Expected src/part_d.py beside this script's project folder.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import part_d


def main():
    print("=" * 72)
    print("PART D — REFUND AGENT (TASKS 5–7)")
    print("=" * 72)

    # Task 5: Inspect tool specifications and demonstrate read/calculation tools.
    print("\nTASK 5 — REFUND TOOLS AND TOOL MAP")
    print("Tool specifications:")
    print(json.dumps(part_d.TOOLS_SPEC, indent=2, ensure_ascii=False))
    print("\nTool names:", list(part_d.TOOL_MAP.keys()))

    scenarios_path = ROOT / "data" / "raw" / "agent_scenarios.csv"
    scenarios_df = pd.read_csv(scenarios_path, keep_default_na=False)
    print("\nScenario input preview:")
    print(scenarios_df.head().to_string(index=False))

    sample_order_id = None
    for message in scenarios_df["message"].astype(str):
        match = re.search(r"ORD-\d+", message)
        if match:
            sample_order_id = match.group(0)
            break
    if sample_order_id:
        print("\nSample order lookup:", sample_order_id)
        print(part_d.get_order(sample_order_id))

    print("\nRefund calculation example (INR 128.25, 52 minutes):",
          part_d.calculate_refund(128.25, 52))
    print("Refund calculation at 45 minutes:",
          part_d.calculate_refund(128.25, 45))

    # Tasks 6–7: Run all seven scenarios, safeguard demos, and checkpoint recovery.
    print("\nTASK 6 — RUN THE SEVEN SCENARIOS")
    part_d.build_index(overlap=10)
    scenario_results = part_d.run_scenarios()
    scenario_table = pd.DataFrame([
        {
            "scenario_id": result["scenario_id"],
            "status": result["status"],
            "refund_inr": result.get("refund_amount_inr", 0.0),
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
        }
        for result in scenario_results
    ])
    print(scenario_table.to_string(index=False))
    print("\nAgent run log:", ROOT / "outputs" / "agent_runs.jsonl")

    print("\nTASK 7 — SAFETY-LIMIT DEMONSTRATIONS")
    safety_results = part_d.run_safety_demos()
    print("S1 with token_limit=200:",
          safety_results["token_demo"]["status"])
    print("Token demo calls:", safety_results["token_demo"]["llm_calls"])
    print("Token demo tokens:", safety_results["token_demo"]["tokens_used"])
    print("\nS1 with time_limit_seconds=0:",
          safety_results["timeout_demo"]["status"])
    print("Timeout demo calls:", safety_results["timeout_demo"]["llm_calls"])
    print("Timeout demo tokens:", safety_results["timeout_demo"]["tokens_used"])

    print("\nCRASH AND RESUME DEMONSTRATION")
    resume_demo = part_d.run_resume_demo()
    print("Tools before resume:", resume_demo["tools_before_resume"])
    print("Tools across the full run:", resume_demo["all_tools"])
    print("Final status:", resume_demo["result"]["status"])
    print("Final answer:", resume_demo["result"]["final_answer"])
    print("Model calls:", resume_demo["result"]["llm_calls"])
    print("Tokens used:", resume_demo["result"]["tokens_used"])

    print("\nGenerated artifacts:")
    for path in [
        ROOT / "outputs" / "agent_runs.jsonl",
        ROOT / "outputs" / "refund_ledger.jsonl",
        ROOT / "state",
    ]:
        print(f" - {path.relative_to(ROOT)}: {'exists' if path.exists() else 'MISSING'}")

    ledger_path = ROOT / "outputs" / "refund_ledger.jsonl"
    if ledger_path.exists():
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        print("\nLatest refund ledger entries:")
        for line in lines[-5:]:
            print(json.dumps(json.loads(line), ensure_ascii=False))

    print("\nRefund agent tasks 5–7 complete.")


if __name__ == "__main__":
    main()
