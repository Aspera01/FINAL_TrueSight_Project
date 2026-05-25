"""
TrueSight — JSON History Store
-------------------------------
Replaces the old SQLite backend with a single flat JSON file
(truesight_history.json) at the project root.

Public API is identical to the old Database class so no callers need changes:
    from utils.database import Database
    db = Database()
    file_id = db.save_media_file("video.mp4", "Video", "/path/to/video.mp4")
    db.save_analysis(file_id, aggregated)
    history = db.get_history()
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

HISTORY_PATH = Path(__file__).parent.parent / "truesight_history.json"


class Database:
    def __init__(self, history_path: Path = HISTORY_PATH):
        self._path = history_path
        self._pending: dict[int, dict] = {}
        if not self._path.exists():
            self._write([])

    # ---------------------------------------------------------------- I/O

    def _read(self) -> list[dict]:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write(self, records: list[dict]):
        self._path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _next_id(self, records: list[dict]) -> int:
        if not records and not self._pending:
            return 1
        existing = [r["Report_ID"] for r in records]
        pending = list(self._pending.keys())
        return max(existing + pending, default=0) + 1

    # ---------------------------------------------------------------- Write

    def save_media_file(self, file_name: str, file_type: str, file_path: str) -> int:
        """Reserve an ID and stash file metadata until save_analysis is called."""
        records = self._read()
        new_id = self._next_id(records)
        self._pending[new_id] = {
            "File_Name": file_name,
            "File_Type": file_type,
            "File_Path": file_path,
            "Upload_Date": datetime.now().isoformat(),
        }
        return new_id

    def save_analysis(self, file_id: int, aggregated) -> int:
        """Persist all module results and the overall report. Returns the report ID."""
        timestamp = datetime.now().isoformat()
        file_meta = self._pending.pop(file_id, {})

        module_results = []
        for result in aggregated.module_results:
            discrepancy = (
                result.label
                if result.supported and not result.error
                else (result.error or "N/A")
            )
            module_results.append(
                {
                    "Module_Name": result.module_name,
                    "Probability_Score": round(result.score, 4),
                    "Confidence": round(result.confidence, 4),
                    "Discrepancy_Detected": discrepancy,
                    "Details_JSON": json.dumps(result.details) if result.details else None,
                }
            )

        key_findings = aggregated.key_findings or []
        record = {
            "Report_ID": file_id,
            **file_meta,
            "Overall_Score": round(aggregated.overall_score, 4),
            "Confidence": round(getattr(aggregated, "overall_confidence", 0.0), 4),
            "Verdict": aggregated.verdict,
            "Risk_Level": aggregated.risk_level,
            "Flagged": bool(aggregated.flagged),
            "Threshold_Used": aggregated.threshold,
            "Summary": key_findings[0] if key_findings else aggregated.verdict,
            "Key_Findings": json.dumps(key_findings),
            "Generated_Date": timestamp,
            "module_results": module_results,
        }

        records = self._read()
        records.append(record)
        self._write(records)
        return file_id

    # ---------------------------------------------------------------- Read

    def get_history(self, limit: int = 50) -> list[dict]:
        """Return recent analyses sorted newest-first, ready for HistoryPanel."""
        records = sorted(
            self._read(),
            key=lambda r: r.get("Generated_Date", ""),
            reverse=True,
        )
        return [
            {
                "Report_ID":      r["Report_ID"],
                "Overall_Score":  r.get("Overall_Score", 0),
                "Verdict":        r.get("Verdict", ""),
                "Risk_Level":     r.get("Risk_Level", "low"),
                "Flagged":        r.get("Flagged", False),
                "Generated_Date": r.get("Generated_Date", ""),
                "File_Name":      r.get("File_Name", ""),
                "File_Type":      r.get("File_Type", ""),
                "File_Path":      r.get("File_Path", ""),
            }
            for r in records[:limit]
        ]

    def get_report_detail(self, report_id: int) -> dict[str, Any] | None:
        """Return the full record split into report metadata and module results."""
        for r in self._read():
            if r["Report_ID"] == report_id:
                return {
                    "report": {k: v for k, v in r.items() if k != "module_results"},
                    "module_results": r.get("module_results", []),
                }
        return None

    # ---------------------------------------------------------------- Delete

    def delete_record(self, report_id: int):
        """Remove a record by its Report_ID."""
        records = [r for r in self._read() if r["Report_ID"] != report_id]
        self._write(records)
