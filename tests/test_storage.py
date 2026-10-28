import pytest

from storage import _execute, delete_student, get_student, initialize_database, list_students, save_student


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "students.db"
    initialize_database(path)
    return path


def sample_student(**changes):
    return {
        "student_id": "STU-001",
        "student_name": "Jordan Lee",
        "grade_level": "College 3rd Year",
        "study_hours": 13.5,
        "attendance": 82,
        "previous_score": 72,
        "assignment_score": 81,
        "sleep_hours": 7.5,
        "final_score": 78,
        **changes,
    }


def test_save_update_and_delete_student(database):
    save_student(sample_student(), database)
    assert get_student("STU-001", database)["final_score"] == 78

    save_student(sample_student(final_score=84), database)
    assert len(list_students(database)) == 1
    assert get_student("STU-001", database)["final_score"] == 84

    delete_student("STU-001", database)
    assert get_student("STU-001", database) is None


@pytest.mark.parametrize("field,value", [("attendance", 101), ("study_hours", -1), ("sleep_hours", float("inf"))])
def test_rejects_invalid_measurements(database, field, value):
    with pytest.raises(ValueError):
        save_student(sample_student(**{field: value}), database)


def test_rejects_blank_identity(database):
    with pytest.raises(ValueError, match="required"):
        save_student(sample_student(student_name="   "), database)


def test_initialize_database_creates_parent_directory(tmp_path):
    database_path = tmp_path / "persistent" / "data" / "students.db"

    initialize_database(database_path)

    assert database_path.is_file()


def test_postgres_queries_use_psycopg_placeholders():
    class RecordingConnection:
        def execute(self, query, parameters):
            self.query = query
            self.parameters = parameters

    connection = RecordingConnection()
    _execute(connection, "SELECT * FROM students WHERE student_id = ?", ("STU-001",))

    assert connection.query == "SELECT * FROM students WHERE student_id = %s"
    assert connection.parameters == ("STU-001",)
