# EduPredict

A web-based student performance analysis prototype for teachers and students. The app combines a local student roster, performance summaries, model evaluation, and an individual student view.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run main.py
```

Open the browser URL printed by Streamlit (usually `http://localhost:8501`). The app listens on port `8501` and on network interfaces, so it can also be opened from another device that can reach the host. For a hosted demo, deploy this repository to a Streamlit-compatible service and expose port `8501`.

The first run seeds fictional college-year records. By default, teacher changes are saved in `edupredict.db` beside `main.py`. For a hosted deployment, set a `DATABASE_URL` secret containing a PostgreSQL connection URI; the app creates its roster table and uses PostgreSQL instead of SQLite. The app requires SSL for PostgreSQL connections. Keep credentials in the hosting provider's secret settings, never in source control. For local PostgreSQL testing, put `DATABASE_URL = "postgresql://..."` in the ignored `.streamlit/secrets.toml`. Without that secret, the app continues to use SQLite; configure `EDUPREDICT_DB_PATH` for a persistent mounted volume if you host SQLite. This demo is not approved for real student data.

## Deploy on Streamlit Community Cloud

1. Push the app source to a GitHub repository you control.
2. Sign in at [share.streamlit.io](https://share.streamlit.io), choose **Create app**, and select the repository, `main` branch, and `main.py`.
3. In the app's **Advanced settings → Secrets**, add the PostgreSQL connection URI from your managed database provider:

   ```toml
   DATABASE_URL = "postgresql://USER:PASSWORD@HOST:PORT/DATABASE?sslmode=require"
   ```

4. Deploy and check the app logs if startup fails. Streamlit Community Cloud assigns a `*.streamlit.app` address; you can choose its subdomain during setup. It does not directly attach arbitrary domains you own. To use a custom domain such as `edupredict.demo.com`, choose a host that supports custom domains (for example, Render) and configure its DNS instructions with the domain owner.

## Workspaces

- **Teacher:** review cohort charts, compare score-estimation and performance-band models, upload a CSV for analysis, manage roster records, and inspect the project milestone tracker.
- **Student:** use the demo role selector to inspect a sample profile, attendance, internal marks, and an upcoming-exam what-if planner with model error disclosure.
- **Data:** preview and export the dataset used by the current analysis.
- **Interface:** a dark, responsive dashboard theme with high-contrast metrics, accessible focus states, and no external font dependency.
- **Browser access:** the Streamlit server is configured to listen on `0.0.0.0:8501`; Streamlit renders the interface in a regular web browser without a separate desktop client.

The role and student selectors are demonstration controls, not authentication. Do not enter a real NETID password into this app. A production college login must redirect to the university's OIDC/SAML identity provider; this demo intentionally does not collect credentials or pretend that a local password/CAPTCHA is secure.

## Professor demo checklist

1. Run `python -m pytest -q`. Expected: all tests pass, including the teacher dashboard and student planner checks.
2. Start the app with `streamlit run main.py`. Expected: Streamlit prints a local URL and the teacher overview loads with a cohort of 48 fictional students.
3. In the sidebar, switch **Demo role** to **Student**. Expected: attendance, internal assessment, performance band, baseline estimate, study-plan estimate, target distance, and held-out RMSE appear.
4. Switch back to **Teacher** and open **Project status**. Expected: 7 complete milestones, 1 in progress (campus identity integration), and 1 not started (production deployment/privacy review).
5. In **Analysis data**, select **Upload CSV** without uploading a file. Expected: an upload prompt appears and no analysis metrics are shown. Upload [examples/sample_students.csv](examples/sample_students.csv) to test analysis with 16 fictional records across all four performance bands.
6. Open **Student roster** to add or correct a record. Expected: saved changes persist after a rerun; out-of-range values are rejected.

For a short live presentation, show the cohort metrics and charts, compare the regression/classification models, change one student record and show it persists, then run the student exam planner and point out its held-out RMSE disclosure. Finish with the project-status tab and test output. This demonstrates working analytics, storage, what-if interaction, and automated checks while making clear that role switching is only a demo and campus sign-in is not implemented.

Example student planner output is data-dependent. It shows a **Baseline estimate**, a **Study-plan estimate** with the point difference, **Distance to target**, and the selected model's held-out **RMSE**. It uses fictional cohort data and a final-score proxy, not actual exam records or guaranteed improvements.

## Project status

The in-app milestone tracker reports completed, in-progress, and not-started deliverables. It tracks implementation milestones, not hours worked or an objective percentage of effort. Current production blockers are institutional SSO, authorization, privacy review, and deployment configuration.

## Data fields

CSV analysis expects numeric `study_hours`, `attendance`, `previous_score`, `assignment_score`, `sleep_hours`, and `final_score` columns. Uploaded values outside valid score, attendance, or daily-hour ranges are rejected instead of being clipped. Malformed files and files missing required columns do not fall back to demo data or display analysis metrics. Rows without a numeric final score are excluded; missing feature values are imputed by the training pipeline. Classification comparison is unavailable when the training split contains only one performance band.

## Tests

```bash
python -m pytest
```

Model output is exploratory, not a diagnosis or a high-stakes decision. Correlations do not establish cause, and all predictions require educator review.
