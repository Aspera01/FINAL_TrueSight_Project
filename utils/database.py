"""
Local SQLite Database
----------------------
Stores analysis history locally — no cloud, no accounts.
Schema matches the Data Dictionary from the TrueSight capstone paper:
  - Media_File
  - Analysis_Result (one row per module per analysis)
  - Report (one row per analysis)

Usage:
    from utils.database import Database
    db = Database()
    file_id = db.save_media_file("video.mp4", "Video", "/path/to/video.mp4")
    db.save_analysis(file_id, results, aggregated)
    history = db.get_history()
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "truesight.db"


class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        """Create tables if they don't exist yet."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS Media_File (
                    File_ID     INTEGER PRIMARY KEY AUTOINCREMENT,
                    File_Type   TEXT NOT NULL,
                    File_Name   TEXT NOT NULL,
                    Upload_Date TEXT NOT NULL,
                    File_Path   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS Detection_Module (
                    Module_ID       INTEGER PRIMARY KEY AUTOINCREMENT,
                    Module_Name     TEXT NOT NULL UNIQUE,
                    Description     TEXT,
                    Algorithm_Type  TEXT
                );

                CREATE TABLE IF NOT EXISTS Analysis_Result (
                    Result_ID               INTEGER PRIMARY KEY AUTOINCREMENT,
                    File_ID                 INTEGER NOT NULL,
                    Module_ID               INTEGER NOT NULL,
                    Probability_Score       REAL NOT NULL,
                    Confidence              REAL NOT NULL,
                    Discrepancy_Detected    TEXT,
                    Details_JSON            TEXT,
                    Timestamp               TEXT NOT NULL,
                    FOREIGN KEY (File_ID)   REFERENCES Media_File(File_ID),
                    FOREIGN KEY (Module_ID) REFERENCES Detection_Module(Module_ID)
                );

                CREATE TABLE IF NOT EXISTS Report (
                    Report_ID       INTEGER PRIMARY KEY AUTOINCREMENT,
                    File_ID         INTEGER NOT NULL,
                    Overall_Score   REAL NOT NULL,
                    Verdict         TEXT NOT NULL,
                    Risk_Level      TEXT NOT NULL,
                    Flagged         INTEGER NOT NULL,
                    Threshold_Used  REAL NOT NULL,
                    Summary         TEXT,
                    Key_Findings    TEXT,
                    Generated_Date  TEXT NOT NULL,
                    FOREIGN KEY (File_ID) REFERENCES Media_File(File_ID)
                );
            """)
            self._seed_modules(conn)

    def _seed_modules(self, conn: sqlite3.Connection):
        """Insert the known detection modules if not already present."""
        modules = [
            ("Error Level Analysis",       "Detects pixel-level tampering via JPEG re-compression comparison",        "Signal Processing"),
            ("Frequency Analysis",         "Detects GAN artifacts in DCT/FFT frequency domain",                       "Signal Processing"),
            ("Face CNN (EfficientNet-B4)", "Deep CNN face deepfake classifier trained on FaceForensics++",             "Pre-trained CNN"),
            ("Temporal Consistency",       "Detects frame-to-frame inconsistencies via optical flow analysis",        "Computer Vision"),
            ("Lip-Sync Analysis",          "Detects audio-visual desynchronization using facial landmark tracking",   "Multi-modal Analysis"),
            ("Audio Spectrogram (LCNN)",   "Mel-spectrogram CNN classifier for synthetic speech detection",           "Pre-trained CNN"),
            ("Noise Floor Consistency",    "Detects audio splicing via background noise profile analysis",            "Signal Processing"),
        ]
        for name, desc, algo_type in modules:
            conn.execute(
                "INSERT OR IGNORE INTO Detection_Module (Module_Name, Description, Algorithm_Type) VALUES (?, ?, ?)",
                (name, desc, algo_type)
            )

    # ------------------------------------------------------------------ Write

    def save_media_file(self, file_name: str, file_type: str, file_path: str) -> int:
        """Insert a media file record and return its File_ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO Media_File (File_Type, File_Name, Upload_Date, File_Path) VALUES (?, ?, ?, ?)",
                (file_type, file_name, datetime.now().isoformat(), file_path)
            )
            return cursor.lastrowid

    def save_analysis(self, file_id: int, aggregated) -> int:
        """
        Save all module results and the overall report for one analysis session.
        Returns the Report_ID.
        """
        timestamp = datetime.now().isoformat()

        with self._connect() as conn:
            # Fetch module ID map
            rows = conn.execute("SELECT Module_ID, Module_Name FROM Detection_Module").fetchall()
            module_map = {r["Module_Name"]: r["Module_ID"] for r in rows}

            # Insert one Analysis_Result per module
            for result in aggregated.module_results:
                module_id = module_map.get(result.module_name)
                if module_id is None:
                    continue

                discrepancy = result.label if result.supported and not result.error else (result.error or "N/A")
                details_json = json.dumps(result.details) if result.details else None

                conn.execute(
                    """INSERT INTO Analysis_Result
                       (File_ID, Module_ID, Probability_Score, Confidence,
                        Discrepancy_Detected, Details_JSON, Timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        file_id,
                        module_id,
                        round(result.score, 4),
                        round(result.confidence, 4),
                        discrepancy,
                        details_json,
                        timestamp,
                    )
                )

            # Insert overall Report
            key_findings_json = json.dumps(aggregated.key_findings)
            summary = aggregated.key_findings[0] if aggregated.key_findings else aggregated.verdict

            cursor = conn.execute(
                """INSERT INTO Report
                   (File_ID, Overall_Score, Verdict, Risk_Level, Flagged,
                    Threshold_Used, Summary, Key_Findings, Generated_Date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_id,
                    round(aggregated.overall_score, 4),
                    aggregated.verdict,
                    aggregated.risk_level,
                    1 if aggregated.flagged else 0,
                    aggregated.threshold,
                    summary,
                    key_findings_json,
                    timestamp,
                )
            )
            return cursor.lastrowid

    # ------------------------------------------------------------------ Read

    def get_history(self, limit: int = 50) -> list[dict]:
        """Return recent analysis history for display."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT r.Report_ID, r.Overall_Score, r.Verdict, r.Risk_Level,
                          r.Flagged, r.Generated_Date,
                          mf.File_Name, mf.File_Type, mf.File_Path
                   FROM Report r
                   JOIN Media_File mf ON r.File_ID = mf.File_ID
                   ORDER BY r.Generated_Date DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_report_detail(self, report_id: int) -> dict[str, Any] | None:
        """Return a full report with all module results."""
        with self._connect() as conn:
            report = conn.execute(
                "SELECT * FROM Report WHERE Report_ID = ?", (report_id,)
            ).fetchone()
            if not report:
                return None

            results = conn.execute(
                """SELECT ar.*, dm.Module_Name, dm.Algorithm_Type
                   FROM Analysis_Result ar
                   JOIN Detection_Module dm ON ar.Module_ID = dm.Module_ID
                   WHERE ar.File_ID = ?
                   ORDER BY ar.Probability_Score DESC""",
                (report["File_ID"],)
            ).fetchall()

            return {
                "report": dict(report),
                "module_results": [dict(r) for r in results],
            }

    def delete_record(self, report_id: int):
        """Delete a report and its associated data."""
        with self._connect() as conn:
            file_id = conn.execute(
                "SELECT File_ID FROM Report WHERE Report_ID = ?", (report_id,)
            ).fetchone()
            if file_id:
                conn.execute("DELETE FROM Analysis_Result WHERE File_ID = ?", (file_id[0],))
                conn.execute("DELETE FROM Report WHERE File_ID = ?", (file_id[0],))
                conn.execute("DELETE FROM Media_File WHERE File_ID = ?", (file_id[0],))
