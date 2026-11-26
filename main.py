"""Command-line interface for the Autonomous AI Data Analyst.

Examples
  python main.py ask "What are total revenue and profit by region?"
  python main.py chat --csv data/sample_sales.csv
  python main.py ask "Monthly revenue trend" --provider anthropic --report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_data_analyst import DataAnalystAgent, Database, build_llm, load_settings
from ai_data_analyst.llm import LLMError

DEFAULT_CSV = Path(__file__).parent / "data" / "sample_sales.csv"


def build_agent(args) -> tuple[DataAnalystAgent, Database]:
    settings = load_settings(args.provider)
    db = Database(max_rows=settings.max_rows)
    csvs = args.csv or [str(DEFAULT_CSV)]
    for path in csvs:
        if not Path(path).exists():
            sys.exit(f"CSV not found: {path}")
        table = db.load_csv(path)
        print(f"Loaded {path} -> table '{table}'")
    try:
        llm = build_llm(settings)
    except LLMError as exc:
        sys.exit(f"LLM setup error: {exc}")
    print(f"Provider: {llm.name}\n")
    return DataAnalystAgent(db, llm, settings), db


def show(result, settings, report: bool) -> None:
    print(f"\n=== {result.question} ===")
    if not result.ok:
        print(f"Could not answer: {result.error}")
        for i, a in enumerate(result.attempts, 1):
            print(f"  attempt {i}: {a.error}\n    {a.sql}")
        return
    if len(result.attempts) > 1:
        print(f"(self-corrected after {len(result.attempts) - 1} failed attempt(s))")
    print(f"\nSQL:\n{result.sql}\n")
    print(result.data.head(15).to_string(index=False))
    if len(result.data) > 15:
        print(f"... {len(result.data) - 15} more rows")
    print(f"\n{result.summary}\n")
    if result.chart_path:
        print(f"Chart saved: {result.chart_path}")
    if report:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(result.chart_path).stem if result.chart_path else "report"
        md = settings.output_dir / f"{stem}.md"
        md.write_text(result.to_markdown(), encoding="utf-8")
        print(f"Report saved: {md}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask questions about your data in plain English.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (("ask", "answer one question"), ("chat", "interactive session")):
        p = sub.add_parser(name, help=help_)
        if name == "ask":
            p.add_argument("question")
        p.add_argument("--csv", action="append", help="CSV file to load (repeatable). Default: sample sales data")
        p.add_argument("--provider", choices=["anthropic", "openai", "offline"], help="LLM backend (default: auto-detect from env)")
        p.add_argument("--report", action="store_true", help="also save a Markdown report next to the chart")
    args = parser.parse_args(argv)

    agent, _ = build_agent(args)

    if args.command == "ask":
        show(agent.ask(args.question), agent.settings, args.report)
        return

    print("Ask a question about your data. Type 'schema' to see the tables, 'exit' to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "schema":
            from ai_data_analyst.database import schema_to_text
            print(schema_to_text(agent.db.describe()))
            continue
        show(agent.ask(q), agent.settings, args.report)


if __name__ == "__main__":
    main()
