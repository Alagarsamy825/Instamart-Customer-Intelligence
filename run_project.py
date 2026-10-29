#!/usr/bin/env python3
"""Single entry point for the Instamart Customer Intelligence project.

Run from any directory with: python run_project.py
The script executes the existing Part A, B, C and Part D policy-RAG modules
in order without rewriting their implementation.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

RAW_DIR = ROOT / "data" / "raw"

STAGES = [
    ("Part A — audit and clean raw data", [PYTHON, "src/audit_data.py"]),
    ("Part A — validate cleaned data", [PYTHON, "src/validate.py"]),
    ("Part A — run SQL queries", [PYTHON, "src/run_queries.py"]),
    ("Part A — pandas exploratory analysis", [PYTHON, "src/explore.py"]),
    ("Part B — customer segmentation", [PYTHON, "src/rfm_segments.py"]),
    ("Part B — lapse prediction and retention list", [PYTHON, "src/lapse_model.py"]),
    ("Part C — support ticket triage", [PYTHON, "src/triage.py"]),
    ("Part D — policy RAG and evaluation", [PYTHON, "src/policy_rag.py"]),
]


def main():
    print("Instamart Customer Intelligence — full project runner", flush=True)
    print(f"Project directory: {ROOT}", flush=True)

    required = [
        RAW_DIR / "customers.csv",
        RAW_DIR / "products.csv",
        RAW_DIR / "orders.csv",
        RAW_DIR / "tickets.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        print("Required input files are missing:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print("If raw data is not committed, run: python generate_data.py", file=sys.stderr)
        return 1

    for title, command in STAGES:
        print("\n" + "=" * 76, flush=True)
        print(title, flush=True)
        print("Command:", " ".join(command), flush=True)
        print("=" * 76, flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            print(f"\nFAILED: {title} (exit code {result.returncode})", file=sys.stderr)
            print("No later stages were run. Fix the error above and run this command again.", file=sys.stderr)
            return result.returncode

    # The supplied project refers to src/policy_rag.py from src/refund_agent.py,
    # but that implementation is absent from the uploaded repository snapshot.
    if not (ROOT / "src" / "policy_rag.py").exists():
        print("\nPart D refund-agent scenarios were not run: src/policy_rag.py is missing.", file=sys.stderr)
        print("The provided src/refund_agent.py imports this module, so running the refund-agent tasks", file=sys.stderr)
        print("requires restoring/committing src/policy_rag.py. The Part D policy-RAG tasks above have run.", file=sys.stderr)
        return 2

    result = subprocess.run([PYTHON, "src/refund_agent.py"], cwd=ROOT)
    if result.returncode:
        print("Refund-agent stage failed. See the error above.", file=sys.stderr)
        return result.returncode

    print("\nAll project stages completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
