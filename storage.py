"""Local SQLite persistence for the EduPredict student roster."""

from __future__ import annotations

import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


DATABASE_PATH = Path(
	os.environ.get("EDUPREDICT_DB_PATH", str(Path(__file__).with_name("edupredict.db")))
).expanduser()
RECORD_FIELDS = (
	"student_id",
	"student_name",
	"grade_level",
	"study_hours",
	"attendance",
	"previous_score",
	"assignment_score",
	"sleep_hours",
	"final_score",
)


def _connect(database_path: str | Path = DATABASE_PATH):
	database_url = os.environ.get("DATABASE_URL")
	if database_url:
		import psycopg
		from psycopg.rows import dict_row

		return psycopg.connect(
			database_url,
			connect_timeout=10,
			prepare_threshold=None,
			row_factory=dict_row,
			sslmode="require",
		)
	connection = sqlite3.connect(database_path, timeout=30)
	connection.row_factory = sqlite3.Row
	return connection


def _execute(connection, query: str, parameters: tuple = ()):
	if not isinstance(connection, sqlite3.Connection):
		query = query.replace("?", "%s")
	return connection.execute(query, parameters)


def initialize_database(database_path: str | Path = DATABASE_PATH) -> None:
	"""Create the roster table if it does not exist yet."""
	if not os.environ.get("DATABASE_URL") and str(database_path) != ":memory:":
		Path(database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
	with closing(_connect(database_path)) as connection, connection:
		_execute(
			connection,
			"""
			CREATE TABLE IF NOT EXISTS students (
				student_id TEXT PRIMARY KEY,
				student_name TEXT NOT NULL,
				grade_level TEXT NOT NULL,
				study_hours REAL NOT NULL CHECK (study_hours BETWEEN 0 AND 24),
				attendance REAL NOT NULL CHECK (attendance BETWEEN 0 AND 100),
				previous_score REAL NOT NULL CHECK (previous_score BETWEEN 0 AND 100),
				assignment_score REAL NOT NULL CHECK (assignment_score BETWEEN 0 AND 100),
				sleep_hours REAL NOT NULL CHECK (sleep_hours BETWEEN 0 AND 24),
				final_score REAL NOT NULL CHECK (final_score BETWEEN 0 AND 100),
				updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
			)
			"""
		)


def validate_student(record: dict[str, Any]) -> dict[str, Any]:
	"""Normalize and validate a student record before it reaches SQLite."""
	missing = [field for field in RECORD_FIELDS if field not in record]
	if missing:
		raise ValueError("Missing required fields: " + ", ".join(missing))

	cleaned: dict[str, Any] = {
		"student_id": str(record["student_id"]).strip(),
		"student_name": str(record["student_name"]).strip(),
		"grade_level": str(record["grade_level"]).strip(),
	}
	if not cleaned["student_id"] or not cleaned["student_name"] or not cleaned["grade_level"]:
		raise ValueError("Student ID, name, and grade level are required.")

	for field in RECORD_FIELDS[3:]:
		try:
			value = float(record[field])
		except (TypeError, ValueError) as exc:
			raise ValueError(f"{field.replace('_', ' ').title()} must be a number.") from exc
		if not math.isfinite(value):
			raise ValueError(f"{field.replace('_', ' ').title()} must be finite.")
		upper_bound = 24 if field in ("study_hours", "sleep_hours") else 100
		if not 0 <= value <= upper_bound:
			raise ValueError(f"{field.replace('_', ' ').title()} must be between 0 and {upper_bound}.")
		cleaned[field] = value
	return cleaned


def list_students(database_path: str | Path = DATABASE_PATH) -> list[dict[str, Any]]:
	"""Return roster records in a stable student-ID order."""
	with closing(_connect(database_path)) as connection, connection:
		rows = _execute(connection, "SELECT * FROM students ORDER BY student_id").fetchall()
	return [dict(row) for row in rows]


def get_student(student_id: str, database_path: str | Path = DATABASE_PATH) -> dict[str, Any] | None:
	with closing(_connect(database_path)) as connection, connection:
		row = _execute(connection, "SELECT * FROM students WHERE student_id = ?", (student_id,)).fetchone()
	return dict(row) if row else None


def save_student(record: dict[str, Any], database_path: str | Path = DATABASE_PATH) -> None:
	"""Insert a student or replace the existing record with the same ID."""
	student = validate_student(record)
	columns = ", ".join(RECORD_FIELDS)
	placeholders = ", ".join("?" for _ in RECORD_FIELDS)
	updates = ", ".join(f"{field} = excluded.{field}" for field in RECORD_FIELDS[1:])
	with closing(_connect(database_path)) as connection, connection:
		_execute(
			connection,
			f"INSERT INTO students ({columns}) VALUES ({placeholders}) "
			f"ON CONFLICT(student_id) DO UPDATE SET {updates}, updated_at = CURRENT_TIMESTAMP",
			tuple(student[field] for field in RECORD_FIELDS),
		)


def delete_student(student_id: str, database_path: str | Path = DATABASE_PATH) -> None:
	with closing(_connect(database_path)) as connection, connection:
		_execute(connection, "DELETE FROM students WHERE student_id = ?", (student_id,))