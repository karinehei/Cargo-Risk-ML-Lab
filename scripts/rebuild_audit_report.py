"""Rebuild docs/methodological_audit.md from a saved audit payload (no refit)."""

from __future__ import annotations

import json

from src.audit.report import build_audit_markdown
from src.config import resolve_path


def main() -> None:
    payload_path = resolve_path("artifacts/audit/audit_payload.json")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    report = build_audit_markdown(payload)
    resolve_path("artifacts/audit/REPORT.md").write_text(report, encoding="utf-8")
    resolve_path("docs/methodological_audit.md").write_text(report, encoding="utf-8")
    print(f"Wrote report ({len(report)} characters)")


if __name__ == "__main__":
    main()
