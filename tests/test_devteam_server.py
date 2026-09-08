"""Unit tests for the devteam MCP server tools.

start_dev_pipeline is tested with subprocess.Popen mocked out — actually
invoking the `claude` CLI in a unit test would be slow, non-deterministic,
and would perform real unsupervised work. check_dev_pipeline/kill_dev_pipeline
are tested against real short-lived stand-in processes (`sleep`) seeded
directly into the job store, the same approach used to manually verify this
module before it had tests.
"""
import subprocess
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from integrations.servers.devteam_server import (
    _MAX_RUNTIME_SECONDS,
    _load,
    _save,
    check_dev_pipeline,
    kill_dev_pipeline,
    list_dev_pipelines,
    start_dev_pipeline,
)

_FAKE_PROJECT = {"name": "dann-of-thursday", "path": "/tmp"}


class TestStartDevPipeline:
    def test_unknown_project_returns_not_found(self, tmp_dann_home):
        with patch("integrations.servers.devteam_server.resolve_project", return_value=None), \
             patch("integrations.servers.devteam_server.find_projects", return_value=[]):
            result = start_dev_pipeline("nonexistent", "do something")
        assert "not found" in result

    def test_launches_and_records_running_job(self, tmp_dann_home):
        mock_proc = MagicMock(pid=99999)
        with patch("integrations.servers.devteam_server.resolve_project", return_value=_FAKE_PROJECT), \
             patch("integrations.servers.devteam_server.subprocess.Popen", return_value=mock_proc) as mock_popen:
            result = start_dev_pipeline("dann-of-thursday", "add a caching layer")

        assert "Started dev pipeline" in result
        jobs = _load()
        assert len(jobs) == 1
        assert jobs[0]["project"] == "dann-of-thursday"
        assert jobs[0]["status"] == "running"
        assert jobs[0]["pid"] == 99999

        # --permission-mode acceptEdits is required for a headless run to be
        # able to do anything — regression guard for that fix.
        cmd = mock_popen.call_args[0][0]
        assert "--permission-mode" in cmd
        assert "acceptEdits" in cmd


class TestCheckAndListDevPipelines:
    def test_check_reports_done_after_process_exits(self, tmp_dann_home):
        proc = subprocess.Popen(["sleep", "0.2"])
        job_id = "realproc1"
        _save([{
            "id": job_id, "project": "dann-of-thursday", "task": "smoke test",
            "pid": proc.pid, "log_path": str(tmp_dann_home / "nope.log"),
            "status": "running", "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None, "exit_code": None,
        }])

        proc.wait(timeout=5)
        # No in-memory Popen handle for this job (as if a fresh process
        # queried it) — exercises the PID-liveness fallback path.
        result = check_dev_pipeline(job_id)
        assert "done" in result

    def test_list_filters_by_project(self, tmp_dann_home):
        _save([
            {"id": "a", "project": "dann-of-thursday", "task": "x", "pid": 1,
             "log_path": "", "status": "done", "started_at": "2026-01-01T00:00:00",
             "finished_at": "2026-01-01T00:01:00", "exit_code": 0},
            {"id": "b", "project": "other-project", "task": "y", "pid": 2,
             "log_path": "", "status": "done", "started_at": "2026-01-01T00:00:00",
             "finished_at": "2026-01-01T00:01:00", "exit_code": 0},
        ])
        result = list_dev_pipelines(project_name="other")
        assert "other-project" in result
        assert "dann-of-thursday" not in result

    def test_empty_list_message(self, tmp_dann_home):
        assert list_dev_pipelines() == "No dev pipeline jobs found."


class TestKillDevPipeline:
    def test_kills_running_job(self, tmp_dann_home):
        proc = subprocess.Popen(["sleep", "5"])
        job_id = "killme1"
        _save([{
            "id": job_id, "project": "dann-of-thursday", "task": "smoke test",
            "pid": proc.pid, "log_path": str(tmp_dann_home / "nope.log"),
            "status": "running", "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None, "exit_code": None,
        }])

        result = kill_dev_pipeline(job_id)
        assert "Stopped" in result

        proc.wait(timeout=5)
        jobs = _load()
        assert jobs[0]["status"] == "killed"

    def test_unknown_job_returns_message(self, tmp_dann_home):
        assert "No job with id" in kill_dev_pipeline("nope")


class TestTimeout:
    def test_long_running_job_marked_timed_out(self, tmp_dann_home):
        proc = subprocess.Popen(["sleep", "5"])
        job_id = "stuck1"
        stale_start = datetime.now() - timedelta(seconds=_MAX_RUNTIME_SECONDS + 60)
        _save([{
            "id": job_id, "project": "dann-of-thursday", "task": "hung pipeline",
            "pid": proc.pid, "log_path": str(tmp_dann_home / "nope.log"),
            "status": "running", "started_at": stale_start.isoformat(timespec="seconds"),
            "finished_at": None, "exit_code": None,
        }])

        result = check_dev_pipeline(job_id)
        assert "timed_out" in result

        # _refresh_status should have SIGTERM'd the real process too
        proc.wait(timeout=5)
        assert proc.returncode is not None
