"""EduPredict: an interactive student performance analysis dashboard.

Run with: streamlit run main.py
"""

from __future__ import annotations

import numpy as np
import os
import pandas as pd
import sqlite3
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	f1_score,
	mean_squared_error,
	precision_score,
	r2_score,
	recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from storage import (
	DATABASE_PATH,
	delete_student,
	get_student,
	initialize_database,
	list_students,
	save_student,
)


FEATURES = [
	"study_hours",
	"attendance",
	"previous_score",
	"assignment_score",
	"sleep_hours",
]
TARGET = "final_score"
GRADE_ORDER = ["Needs support", "Developing", "Proficient", "Excellent"]

st.set_page_config(page_title="EduPredict | Student Analytics", page_icon="📚", layout="wide")

try:
	database_url = st.secrets.get("DATABASE_URL")
except FileNotFoundError:
	database_url = None
if database_url:
	os.environ["DATABASE_URL"] = str(database_url)


@st.cache_data(show_spinner=False)
def make_demo_data(rows: int = 800, seed: int = 42) -> pd.DataFrame:
	"""Create a reproducible demo dataset when no school data is uploaded."""
	rng = np.random.default_rng(seed)
	study = rng.uniform(0.5, 8.0, rows)
	attendance = rng.uniform(55, 100, rows)
	previous = rng.uniform(35, 100, rows)
	assignments = rng.uniform(40, 100, rows)
	sleep = np.clip(rng.normal(7.2, 1.1, rows), 4, 10)
	score = (
		3.5 * study
		+ 0.19 * attendance
		+ 0.28 * previous
		+ 0.22 * assignments
		+ 1.4 * sleep
		+ rng.normal(0, 7, rows)
	)
	frame = pd.DataFrame(
		{
			"study_hours": study,
			"attendance": attendance,
			"previous_score": previous,
			"assignment_score": assignments,
			"sleep_hours": sleep,
			TARGET: np.clip(score, 0, 100),
		}
	)
	return frame.round(1)


def prepare_data(uploaded_file: object | None) -> tuple[pd.DataFrame, str | None]:
	empty_data = pd.DataFrame(columns=FEATURES + [TARGET])
	if uploaded_file is None:
		return make_demo_data(), None
	try:
		data = pd.read_csv(uploaded_file)
	except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
		return empty_data, f"Could not read the uploaded CSV ({exc}). No analysis was run."

	missing = [column for column in FEATURES + [TARGET] if column not in data.columns]
	if missing:
		return empty_data, "Uploaded CSV is missing required columns: " + ", ".join(missing) + ". No analysis was run."
	data = data[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
	data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
	bounds = {
		"study_hours": (0, 24),
		"attendance": (0, 100),
		"previous_score": (0, 100),
		"assignment_score": (0, 100),
		"sleep_hours": (0, 24),
		"final_score": (0, 100),
	}
	out_of_range_fields = [
		field
		for field, (minimum, maximum) in bounds.items()
		if ((data[field] < minimum) | (data[field] > maximum)).any()
	]
	if out_of_range_fields:
		return empty_data, (
			"Uploaded CSV has values outside the allowed ranges for "
			+ ", ".join(field.replace("_", " ") for field in out_of_range_fields)
			+ ". Correct the data and upload it again. No analysis was run."
		)
	if len(data) < 12:
		return data.reset_index(drop=True), "CSV needs at least 12 rows with a numeric final_score to run analysis."
	return data.reset_index(drop=True), None


def grade_scores(scores: pd.Series | np.ndarray) -> np.ndarray:
	return pd.cut(
		np.asarray(scores),
		bins=[-np.inf, 49, 64, 79, np.inf],
		labels=GRADE_ORDER,
	).astype(str)


def build_regressors(seed: int) -> dict[str, object]:
	return {
		"Linear Regression": Pipeline(
			[("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LinearRegression())]
		),
		"Decision Tree": Pipeline(
			[("imputer", SimpleImputer(strategy="median")), ("model", DecisionTreeRegressor(max_depth=6, random_state=seed))]
		),
		"Random Forest": Pipeline(
			[("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=seed, n_jobs=-1))]
		),
		"K-Nearest Neighbors": Pipeline(
			[("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=7))]
		),
	}


def build_classifiers(seed: int) -> dict[str, object]:
	return {
		"Logistic Regression": Pipeline(
			[("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1500, class_weight="balanced"))]
		),
		"Decision Tree": Pipeline(
			[("imputer", SimpleImputer(strategy="median")), ("model", DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=seed))]
		),
		"Random Forest": Pipeline(
			[("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=160, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1))]
		),
		"K-Nearest Neighbors": Pipeline(
			[("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7))]
		),
	}


def train_models(data: pd.DataFrame, test_fraction: float, seed: int):
	x = data[FEATURES]
	y = data[TARGET].clip(0, 100)
	stratify = grade_scores(y)
	# Stratification is useful for grade classification, but very small classes
	# can make a stratified split impossible; fall back to a normal split then.
	counts = pd.Series(stratify).value_counts()
	stratify_arg = stratify if len(counts) > 1 and counts.min() >= 2 else None
	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=test_fraction, random_state=seed, stratify=stratify_arg
	)
	reg_models = build_regressors(seed)
	reg_rows = []
	for name, model in reg_models.items():
		model.fit(x_train, y_train)
		pred = np.clip(model.predict(x_test), 0, 100)
		reg_rows.append({"Model": name, "RMSE": mean_squared_error(y_test, pred) ** 0.5, "R²": r2_score(y_test, pred)})

	y_train_grade = grade_scores(y_train)
	y_test_grade = grade_scores(y_test)
	if len(np.unique(y_train_grade)) < 2:
		clf_models = {}
		clf_rows = []
	else:
		clf_models = build_classifiers(seed)
		clf_rows = []
		for name, model in clf_models.items():
			model.fit(x_train, y_train_grade)
			pred = model.predict(x_test)
			clf_rows.append(
				{
					"Model": name,
					"Accuracy": accuracy_score(y_test_grade, pred),
					"Precision": precision_score(y_test_grade, pred, average="weighted", zero_division=0),
					"Recall": recall_score(y_test_grade, pred, average="weighted", zero_division=0),
					"F1": f1_score(y_test_grade, pred, average="weighted", zero_division=0),
				}
			)
	return x_test, y_test, reg_models, clf_models, pd.DataFrame(reg_rows), pd.DataFrame(clf_rows)


def fit_projection_model(data: pd.DataFrame, seed: int = 42) -> tuple[object, str, float]:
	"""Fit and evaluate regression models for the student exam-planning scenario."""
	x_train, x_test, y_train, y_test = train_test_split(
		data[FEATURES], data[TARGET].clip(0, 100), test_size=0.25, random_state=seed
	)
	models = build_regressors(seed)
	results = []
	for name, model in models.items():
		model.fit(x_train, y_train)
		prediction = np.clip(model.predict(x_test), 0, 100)
		results.append((mean_squared_error(y_test, prediction) ** 0.5, name))
	rmse, best_name = min(results)
	return models[best_name], best_name, float(rmse)


def seed_roster() -> None:
	"""Create an editable demo roster the first time the local database is opened."""
	existing_students = list_students()
	if existing_students:
		for student in existing_students:
			student_id = str(student["student_id"])
			seed_name = f"Student {student_id.removeprefix('STU-')}"
			legacy_year = str(student["grade_level"])
			if student["student_name"] == seed_name and legacy_year in {"9", "10", "11", "12"}:
				student["grade_level"] = f"Year {int(legacy_year) - 8}"
				save_student(student)
		return
	for index, row in make_demo_data(rows=48, seed=19).iterrows():
		save_student(
			{
				"student_id": f"STU-{index + 1:03}",
				"student_name": f"Student {index + 1:03}",
				"grade_level": f"Year {1 + index % 4}",
				**row.to_dict(),
			}
		)


def show_student_workspace(student: dict[str, object], cohort: pd.DataFrame) -> None:
	"""Render the selected student's private performance view."""
	score = float(student[TARGET])
	grade = str(grade_scores(np.array([score]))[0])
	st.markdown(f"### Welcome, {student['student_name']}")
	st.caption(f"Academic level {student['grade_level']} · Demo ID {student['student_id']}")
	first, second, third, fourth = st.columns(4)
	first.metric("Current score", f"{score:.1f} / 100")
	second.metric("Attendance", f"{float(student['attendance']):.0f}%")
	third.metric("Internal assessment", f"{float(student['assignment_score']):.1f} / 100")
	fourth.metric("Performance band", grade)

	st.subheader("Academic record")
	factors = pd.DataFrame(
		[
			{"Measure": "Previous exam", "Recorded result": f"{float(student['previous_score']):.1f} / 100"},
			{"Measure": "Internal assessment", "Recorded result": f"{float(student['assignment_score']):.1f} / 100"},
			{"Measure": "Attendance", "Recorded result": f"{float(student['attendance']):.1f}%"},
			{"Measure": "Study routine", "Recorded result": f"{float(student['study_hours']):.1f} hours/day"},
			{"Measure": "Sleep routine", "Recorded result": f"{float(student['sleep_hours']):.1f} hours/day"},
		]
	)
	st.dataframe(factors, use_container_width=True, hide_index=True)

	st.subheader("Upcoming internal exam planner")
	if len(cohort) >= 12:
		model, model_name, rmse = fit_projection_model(cohort)
		current = pd.DataFrame([{field: float(student[field]) for field in FEATURES}], columns=FEATURES)
		baseline = float(np.clip(model.predict(current)[0], 0, 100))
		goal_column, hours_column = st.columns(2)
		target_score = goal_column.number_input(
			"Your target score", min_value=0.0, max_value=100.0,
			value=min(100.0, round(baseline + 5, 1)), step=1.0, key=f"goal_{student['student_id']}"
		)
		planned_hours = hours_column.slider(
			"Planned study hours per day", min_value=0.5, max_value=12.0,
			value=min(12.0, max(0.5, float(student["study_hours"]) + 1.0)), step=0.5,
			key=f"hours_{student['student_id']}"
		)
		scenario = current.copy()
		scenario.loc[0, "study_hours"] = planned_hours
		estimate = float(np.clip(model.predict(scenario)[0], 0, 100))
		base_metric, scenario_metric, target_metric = st.columns(3)
		base_metric.metric("Baseline estimate", f"{baseline:.1f} / 100")
		scenario_metric.metric("Study-plan estimate", f"{estimate:.1f} / 100", f"{estimate - baseline:+.1f} points")
		target_metric.metric("Distance to target", f"{estimate - target_score:+.1f} points")
		st.caption(
			f"Model: {model_name} · held-out RMSE: {rmse:.1f} points. "
			"The model uses fictional cohort records and final_score as an exam-score proxy. "
			"Changing study hours is a what-if scenario, not a causal guarantee."
		)
	else:
		st.info("The planner needs at least 12 roster records to evaluate a model. Ask your teacher to add or import more records.")

	if score < 50:
		st.warning("Your current score suggests that extra support could help. Talk with your teacher about a short, manageable study plan.")
	elif score < 65:
		st.info("You are building your skills. Choose one topic to review this week and ask for feedback on your next assignment.")
	elif score < 80:
		st.success("You are making steady progress. Keep your study routine consistent and look for one area to strengthen.")
	else:
		st.success("Strong progress. Keep challenging yourself and share the strategies that are working for you.")
	st.caption("This view is a learning aid, not a diagnosis or a prediction of a student's potential.")


def show_project_status() -> None:
	"""Show a transparent, milestone-based snapshot of project progress."""
	milestones = [
		{"Milestone": "Teacher cohort analytics", "Status": "Complete", "Evidence": "Score distribution, performance bands, and factor correlations"},
		{"Milestone": "Editable student records", "Status": "Complete", "Evidence": "SQLite roster with range checks and add/edit/delete"},
		{"Milestone": "CSV analysis and export", "Status": "Complete", "Evidence": "Required-column validation, preview, and download"},
		{"Milestone": "Model comparison", "Status": "Complete", "Evidence": "Regression and grade-band metrics on held-out data"},
		{"Milestone": "Student academic view", "Status": "Complete", "Evidence": "Attendance, internal marks, previous result, and study factors"},
		{"Milestone": "Exam improvement planner", "Status": "Complete", "Evidence": "What-if projection and held-out RMSE disclosure"},
		{"Milestone": "Professor verification checks", "Status": "Complete", "Evidence": "Automated storage and Streamlit workflow tests"},
		{"Milestone": "Campus identity integration", "Status": "In progress", "Evidence": "Demo role switch is present; university OIDC/SAML connection is not configured"},
		{"Milestone": "Production deployment and privacy review", "Status": "Not started", "Evidence": "Requires institution hosting, access policy, and approved data source"},
	]
	completed = sum(item["Status"] == "Complete" for item in milestones)
	first, second, third = st.columns(3)
	first.metric("Milestones complete", f"{completed} / {len(milestones)}")
	second.metric("In progress", str(sum(item["Status"] == "In progress" for item in milestones)))
	third.metric("Not started", str(sum(item["Status"] == "Not started" for item in milestones)))
	st.caption("This is a count of defined milestones, not an estimate of hours worked or effort remaining.")
	st.dataframe(pd.DataFrame(milestones), use_container_width=True, hide_index=True)
	st.subheader("How to verify the demo")
	st.code("python -m pytest -q\nstreamlit run main.py", language="bash")
	st.write("Expected checks: roster validation passes; teacher dashboard loads; student profile shows attendance and internal marks; planner reports an estimate and RMSE; CSV mode asks for a file instead of substituting sample results.")
	st.warning("The demo selector is not authentication. Never enter a campus password here. Production login must redirect to the university identity provider and enforce server-side roles.")


def show_roster_workspace(students: list[dict[str, object]]) -> None:
	"""Provide teacher tools for adding, correcting, and removing local records."""
	st.subheader("Student roster")
	st.dataframe(
		pd.DataFrame(students)[["student_id", "student_name", "grade_level", "attendance", "final_score", "updated_at"]],
		use_container_width=True,
		hide_index=True,
	)
	st.caption("Changes are saved to the local SQLite database. Student/teacher role switching is a demo control, not authentication.")

	with st.expander("Add a student", expanded=False):
		with st.form("add_student"):
			columns = st.columns(3)
			student_id = columns[0].text_input("Student ID", placeholder="e.g. STU-049")
			student_name = columns[1].text_input("Student name")
			grade_level = columns[2].text_input("College year / level", value="Year 1")
			values = {}
			for column, (field, label, maximum, default) in zip(
				st.columns(5),
				[
					("study_hours", "Study hours/day", 24.0, 3.0),
					("attendance", "Attendance %", 100.0, 85.0),
					("previous_score", "Previous score", 100.0, 70.0),
					("assignment_score", "Assignment score", 100.0, 75.0),
					("sleep_hours", "Sleep hours/day", 24.0, 7.0),
				],
			):
				values[field] = column.number_input(label, min_value=0.0, max_value=maximum, value=default, step=1.0)
			final_score = st.number_input("Final score", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
			if st.form_submit_button("Add student", type="primary"):
				try:
					save_student({"student_id": student_id, "student_name": student_name, "grade_level": grade_level, **values, "final_score": final_score})
				except (ValueError, sqlite3.IntegrityError) as error:
					st.error(f"Student was not saved: {error}")
				else:
					st.success("Student record added.")
					st.rerun()

	if students:
		student_ids = [str(student["student_id"]) for student in students]
		selected_id = st.selectbox("Select a record to correct", student_ids, key="edit_student_id")
		selected = get_student(selected_id)
		if selected:
			with st.form("edit_student"):
				columns = st.columns(3)
				student_name = columns[0].text_input("Student name", value=str(selected["student_name"]))
				grade_level = columns[1].text_input("College year / level", value=str(selected["grade_level"]))
				columns[2].text_input("Student ID", value=selected_id, disabled=True)
				values = {}
				for column, field, label, maximum in zip(
					st.columns(6),
					["study_hours", "attendance", "previous_score", "assignment_score", "sleep_hours", "final_score"],
					["Study hours/day", "Attendance %", "Previous score", "Assignment score", "Sleep hours/day", "Final score"],
					[24.0, 100.0, 100.0, 100.0, 24.0, 100.0],
				):
					values[field] = column.number_input(label, min_value=0.0, max_value=maximum, value=float(selected[field]), step=1.0, key=f"edit_{field}")
				if st.form_submit_button("Save corrections", type="primary"):
					try:
						save_student({"student_id": selected_id, "student_name": student_name, "grade_level": grade_level, **values})
					except ValueError as error:
						st.error(f"Student was not saved: {error}")
					else:
						st.success("Corrections saved.")
						st.rerun()
			if st.checkbox("Enable record deletion", key="confirm_delete"):
				if st.button("Delete selected student", type="secondary"):
					delete_student(selected_id)
					st.success("Student record deleted.")
					st.rerun()


initialize_database(DATABASE_PATH)
seed_roster()
students = list_students()

st.markdown(
	"""
	<style>
	:root {
		color-scheme: dark;
		--canvas: #0b1120;
		--surface: #111a2b;
		--surface-raised: #172338;
		--line: #26354d;
		--ink: #e6edf7;
		--muted: #94a3b8;
		--accent: #38bdf8;
		--accent-strong: #0ea5e9;
		--positive: #34d399;
	}
	html, body, [data-testid="stAppViewContainer"], .stApp {
		background: var(--canvas);
		color: var(--ink);
		font-family: Inter, "Segoe UI", sans-serif;
	}
	[data-testid="stHeader"] { background: rgba(11, 17, 32, .92); }
	.block-container { max-width: 1480px; padding: 1.6rem 2.2rem 3rem; }
	h1, h2, h3, h4 { color: var(--ink); letter-spacing: -.025em; }
	h1, h2, h3 { font-weight: 650; }
	p, label, [data-testid="stMarkdownContainer"] { color: var(--ink); }
	[data-testid="stCaptionContainer"], [data-testid="stMetricLabel"] { color: var(--muted); }
	[data-testid="stSidebar"] { background: #0e1727; border-right: 1px solid var(--line); }
	[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--ink); }
	.brand {
		padding: 1.2rem 1.5rem;
		border: 1px solid #263b57;
		border-left: 3px solid var(--accent);
		border-radius: 12px;
		background: linear-gradient(115deg, #16243a, #111a2b 68%);
		box-shadow: 0 12px 30px rgba(0, 0, 0, .16);
		margin: 0 0 1.5rem;
	}
	.brand h1 { color: #f8fafc; margin: 0; font-size: 1.85rem; }
	.brand p { margin: .35rem 0 0; color: #a9bad0; }
	[data-testid="stMetric"] {
		background: var(--surface);
		padding: 1rem 1.1rem;
		border: 1px solid var(--line);
		border-top: 2px solid var(--accent);
		border-radius: 10px;
		box-shadow: 0 8px 20px rgba(0, 0, 0, .12);
	}
	[data-testid="stMetricValue"] { color: #f1f5f9; font-weight: 650; }
	[data-testid="stMetricDelta"] svg { fill: currentColor; }
	.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
		border-radius: 8px;
		border-color: var(--line);
	}
	.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
		background: var(--accent-strong);
		border-color: var(--accent-strong);
		color: #06121f;
		font-weight: 650;
	}
	.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
		background: var(--accent);
		border-color: var(--accent);
		color: #06121f;
	}
	.stTabs [data-baseweb="tab-list"] { gap: .45rem; border-bottom: 1px solid var(--line); }
	.stTabs [data-baseweb="tab"] { height: 3rem; color: var(--muted); }
	.stTabs [data-baseweb="tab"]:hover { color: var(--ink); }
	.stTabs [aria-selected="true"] { color: var(--accent); font-weight: 650; }
	[data-testid="stExpander"], [data-testid="stForm"] {
		border-color: var(--line);
		border-radius: 10px;
		background: rgba(17, 26, 43, .48);
	}
	input, textarea, [data-baseweb="select"] > div {
		background-color: var(--surface-raised) !important;
		border-color: var(--line) !important;
		color: var(--ink) !important;
	}
	[data-baseweb="popover"], [role="listbox"] { background: var(--surface-raised); }
	[role="option"] { color: var(--ink); }
	[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 9px; overflow: hidden; }
	[data-testid="stAlert"] { border-radius: 9px; }
	a { color: var(--accent); }
	:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
	@media (max-width: 700px) {
		.block-container { padding: 1rem .8rem 2rem; }
		.brand { padding: 1rem; }
		.brand h1 { font-size: 1.6rem; }
	}
	</style>
	<div class="brand"><h1>EduPredict</h1>
	<p>Student progress, made understandable · Learning analytics demo</p></div>
	""",
	unsafe_allow_html=True,
)

with st.sidebar:
	st.header("Your workspace")
	role = st.radio("Demo role", ["Teacher", "Student"], horizontal=True)
	st.divider()
	student_ids = [str(student["student_id"]) for student in students]
	if role == "Student":
		selected_student_id = st.selectbox("Student profile", student_ids)
		st.caption("Demo profile selector. Production access needs verified sign-in and permissions.")
	else:
		st.caption(f"{len(students)} student records · stored locally")

if role == "Student":
	selected_student = get_student(selected_student_id)
	if selected_student:
		cohort = pd.DataFrame([{field: student[field] for field in FEATURES + [TARGET]} for student in students])
		show_student_workspace(selected_student, cohort)
else:
	with st.sidebar:
		st.header("Analysis settings")
		source = st.selectbox("Analysis data", ["Saved student roster", "Generated sample", "Upload CSV"])
		uploaded = st.file_uploader("Student data CSV", type=["csv"], disabled=source != "Upload CSV")
		st.caption("CSV fields: study_hours, attendance, previous_score, assignment_score, sleep_hours, final_score")
		test_fraction = st.slider("Test set size", min_value=0.15, max_value=0.40, value=0.25, step=0.05)
		seed = st.number_input("Random seed", min_value=0, max_value=10000, value=42, step=1)

	if source == "Saved student roster":
		data = pd.DataFrame([{field: student[field] for field in FEATURES + [TARGET]} for student in students])
		warning = None if len(data) >= 12 else "Add at least 12 student records to train the models."
	elif source == "Generated sample":
		data, warning = make_demo_data(), None
	elif uploaded is None:
		data, warning = pd.DataFrame(columns=FEATURES + [TARGET]), "Upload a CSV with the required numeric columns to analyze it."
	else:
		data, warning = prepare_data(uploaded)
	if warning:
		st.warning(warning)
	if len(data) < 12:
		st.info("The teacher roster workspace is available below. Model analysis needs at least 12 valid rows.")
		with st.expander("Manage student records"):
			show_roster_workspace(students)
	else:
		x_test, y_test, regressors, classifiers, regression_results, classification_results = train_models(
			data, test_fraction, int(seed)
		)
		grade_distribution = pd.Series(grade_scores(data[TARGET])).value_counts()
		metric_cols = st.columns(4)
		metric_cols[0].metric("Students analyzed", f"{len(data):,}")
		metric_cols[1].metric("Average final score", f"{data[TARGET].mean():.1f} / 100")
		metric_cols[2].metric("Average attendance", f"{data['attendance'].mean():.1f}%")
		metric_cols[3].metric("Top regression RMSE", f"{regression_results['RMSE'].min():.2f}")

		overview, models_tab, roster_tab, data_tab, status_tab = st.tabs(
			["Overview", "Model lab", "Student roster", "Data", "Project status"]
		)
		with overview:
			left, right = st.columns(2)
			with left:
				st.subheader("Final score distribution")
				hist = np.histogram(data[TARGET].clip(0, 100), bins=10, range=(0, 100))
				st.bar_chart(pd.DataFrame({"Students": hist[0]}, index=[f"{hist[1][i]:.0f}–{hist[1][i + 1]:.0f}" for i in range(len(hist[0]))]))
			with right:
				st.subheader("Performance bands")
				st.bar_chart(grade_distribution.reindex(GRADE_ORDER, fill_value=0))
			st.subheader("Factors associated with final score")
			correlations = data[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
			st.bar_chart(correlations.rename("Correlation with final score"))
			st.caption("Correlations describe association, not cause. Grade bands: below 50, 50–64, 65–79, and 80+.")

		with models_tab:
			st.subheader("Score estimation")
			st.caption("RMSE is measured in score points; lower is better. R² closer to 1 is better.")
			st.dataframe(regression_results.sort_values("RMSE").style.format({"RMSE": "{:.2f}", "R²": "{:.3f}"}), use_container_width=True, hide_index=True)
			st.subheader("Performance-band classification")
			if classification_results.empty:
				st.info("Classification comparison is unavailable because the training split contains only one performance band.")
			else:
				st.caption("Grade bands are derived from final_score. Precision, recall, and F1 use weighted averages.")
				st.dataframe(classification_results.sort_values("F1", ascending=False).style.format({column: "{:.3f}" for column in ["Accuracy", "Precision", "Recall", "F1"]}), use_container_width=True, hide_index=True)
			best_name = regression_results.sort_values("RMSE").iloc[0]["Model"]
			st.subheader(f"Actual vs predicted: {best_name}")
			actual_predicted = pd.DataFrame({"Actual score": y_test.to_numpy(), "Predicted score": np.clip(regressors[best_name].predict(x_test), 0, 100)})
			st.scatter_chart(actual_predicted, x="Actual score", y="Predicted score")

		with roster_tab:
			show_roster_workspace(students)

		with data_tab:
			st.subheader("Dataset preview")
			st.dataframe(data.head(100), use_container_width=True, hide_index=True)
			st.caption(f"Showing up to 100 of {len(data):,} rows. Missing feature values are imputed from training data.")
			csv = data.to_csv(index=False).encode("utf-8")
			st.download_button("Download analyzed data", data=csv, file_name="edupredict_student_data.csv", mime="text/csv")

		with status_tab:
			show_project_status()
