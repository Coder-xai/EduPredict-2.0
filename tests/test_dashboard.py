from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "main.py"


def test_teacher_dashboard_exposes_project_status():
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Overview",
        "Model lab",
        "Student roster",
        "Data",
        "Project status",
    ]
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Milestones complete"] == "7 / 9"
    assert metrics["In progress"] == "1"
    assert metrics["Not started"] == "1"


def test_student_dashboard_shows_exam_planner():
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    app.radio[0].set_value("Student")
    app.run()

    assert not app.exception
    assert any("Upcoming internal exam planner" in item.value for item in app.subheader)
    assert {item.label for item in app.metric} >= {
        "Attendance",
        "Internal assessment",
        "Baseline estimate",
        "Study-plan estimate",
        "Distance to target",
    }


def test_upload_mode_does_not_fall_back_to_demo_data():
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    source = next(widget for widget in app.selectbox if widget.label == "Analysis data")
    source.set_value("Upload CSV")
    app.run()

    assert not app.exception
    assert any("Upload a CSV" in warning.value for warning in app.warning)
    assert not app.metric


def test_invalid_csv_does_not_show_demo_analysis():
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    source = next(widget for widget in app.selectbox if widget.label == "Analysis data")
    source.set_value("Upload CSV")
    app.run()
    uploader = next(widget for widget in app.file_uploader if widget.label == "Student data CSV")
    uploader.set_value(("invalid.csv", b"study_hours,final_score\n2,70\n", "text/csv"))
    app.run()

    assert not app.exception
    assert not app.metric
    assert any("missing required columns" in warning.value for warning in app.warning)
    assert any("No analysis was run" in warning.value for warning in app.warning)


def test_out_of_range_csv_is_rejected():
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    source = next(widget for widget in app.selectbox if widget.label == "Analysis data")
    source.set_value("Upload CSV")
    app.run()
    uploader = next(widget for widget in app.file_uploader if widget.label == "Student data CSV")
    rows = [
        "study_hours,attendance,previous_score,assignment_score,sleep_hours,final_score",
        *[f"3,140,60,65,7,60" for _ in range(12)],
    ]
    uploader.set_value(("out-of-range.csv", "\n".join(rows).encode(), "text/csv"))
    app.run()

    assert not app.exception
    assert not app.metric
    assert any("outside the allowed ranges" in warning.value for warning in app.warning)


def test_valid_csv_uses_uploaded_records():
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    source = next(widget for widget in app.selectbox if widget.label == "Analysis data")
    source.set_value("Upload CSV")
    app.run()
    uploader = next(widget for widget in app.file_uploader if widget.label == "Student data CSV")
    sample_csv = APP_PATH.parent / "examples" / "sample_students.csv"
    uploader.set_value(("sample_students.csv", sample_csv.read_bytes(), "text/csv"))
    app.run()

    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Students analyzed"] == "16"


def test_single_band_csv_skips_unavailable_classification():
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    source = next(widget for widget in app.selectbox if widget.label == "Analysis data")
    source.set_value("Upload CSV")
    app.run()
    uploader = next(widget for widget in app.file_uploader if widget.label == "Student data CSV")
    rows = [
        "study_hours,attendance,previous_score,assignment_score,sleep_hours,final_score",
        *[f"{index / 2},80,60,65,7,60" for index in range(12)],
    ]
    uploader.set_value(("single-band.csv", "\n".join(rows).encode(), "text/csv"))
    app.run()

    assert not app.exception
    assert any("only one performance band" in item.value for item in app.info)
