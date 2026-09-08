#!/usr/bin/env python3
"""MCP server that runs multi-phase Claude Code dev pipelines in the background.

Unlike ask_claude_code (a single blocking Q&A call with a 120s timeout), a
research -> plan -> implement -> test -> devops pipeline can run for many
minutes — too long to hold a voice turn open. This module launches the
pipeline as a background `claude -p` process (real, unsupervised code
changes — not a Q&A call) and lets you check back on it in a later turn
instead of making Dann wait.

Job state and logs live locally (~/.dann/devteam_jobs.json and
~/.dann/devteam_logs/<job_id>.log) — no external account needed. Project
resolution is delegated to claude_code_server (the canonical source of
Dann's project list) so the two modules can't drift out of sync.

Exposed tools:
  start_dev_pipeline — launch a multi-phase Claude Code pipeline in the background
  check_dev_pipeline — check status/output of a running or finished pipeline
  list_dev_pipelines — list recent pipeline jobs
  kill_dev_pipeline  — stop a running pipeline

Limitation: like notes_server's reminders, Dann does not proactively
announce when a background pipeline finishes — there's no scheduler wired
into the orchestrator yet. check_dev_pipeline is how you'd follow up.
"""
import os
import signal
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.mcp_servers._store import dann_home, load_json, save_json
from src.mcp_servers.claude_code_server import find_projects, resolve_project

mcp = FastMCP("devteam")

_FILE = "devteam_jobs.json"
_LOG_DIR_NAME = "devteam_logs"

# A pipeline stuck this long is almost certainly hung (e.g. waiting on a
# permission prompt that will never come) rather than doing real work —
# _refresh_status kills and marks it "timed_out" rather than leaving it to
# run indefinitely with only kill_dev_pipeline as a manual escape hatch.
_MAX_RUNTIME_SECONDS = 30 * 60

# Popen handles for jobs launched by *this* server process — gives
# check_dev_pipeline/kill_dev_pipeline an exact exit code without re-parsing
# the log. Lost if the module is disabled/re-enabled mid-job; the PID-liveness
# check below is the fallback that still works then (just without an exit code).
_processes: dict[str, subprocess.Popen] = {}


def _load() -> list[dict]:
    return load_json(_FILE, [])


def _save(jobs: list[dict]) -> None:
    save_json(_FILE, jobs)


def _log_dir() -> Path:
    d = dann_home() / _LOG_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _build_pipeline_prompt(task: str) -> str:
    return (
        f"{task}\n\n"
        "Run this as a structured multi-phase engineering task:\n"
        "1. Research relevant approaches and current best practices.\n"
        "2. Consolidate findings into a concrete implementation plan.\n"
        "3. Implement the plan.\n"
        "4. Test the implementation.\n"
        "5. Handle any deployment/CI/devops changes the work requires.\n\n"
        "Use your own subagents for research and planning where it helps "
        "quality. Work through all phases autonomously without stopping to "
        "ask questions unless genuinely blocked."
    )


@mcp.tool()
def start_dev_pipeline(project_name: str, task: str) -> str:
    """Launch a multi-phase Claude Code pipeline (research, plan, implement,
    test, devops) in the background and return immediately.

    This performs real, unsupervised code changes and can run for many
    minutes — only use this for substantial build/implement requests you've
    been explicitly asked to carry out, never for quick questions (use
    ask_claude_code for those instead). Use check_dev_pipeline in a later
    turn to see how it's going, not this same turn.

    Args:
        project_name: Name (or partial name) of the project to work in.
        task: The engineering task to carry out end to end.
    """
    project = resolve_project(project_name)
    if not project:
        available = ", ".join(p["name"] for p in find_projects())
        return f"Project '{project_name}' not found. Available projects: {available or 'none'}"

    job_id = uuid.uuid4().hex[:8]
    log_path = _log_dir() / f"{job_id}.log"
    prompt = _build_pipeline_prompt(task)

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            # --permission-mode acceptEdits: this runs headless with nobody
            # available to approve a permission prompt, so without this every
            # pipeline stalls out the instant it needs to write a file (as
            # tested live — see devteam_logs/83d730bd.log). acceptEdits keeps
            # file edits auto-approved but still gates anything more
            # sensitive (e.g. arbitrary bash); it's scoped to this one
            # subprocess/cwd, not a global setting.
            ["claude", "-p", "--permission-mode", "acceptEdits", prompt],
            cwd=project["path"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    _processes[job_id] = proc

    jobs = _load()
    jobs.append({
        "id": job_id,
        "project": project["name"],
        "task": task,
        "pid": proc.pid,
        "log_path": str(log_path),
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "exit_code": None,
    })
    _save(jobs)
    return (
        f"Started dev pipeline {job_id} on {project['name']}. "
        "This will take a while — ask me to check on it later."
    )


def _refresh_status(job: dict) -> dict:
    if job["status"] != "running":
        return job

    started = datetime.fromisoformat(job["started_at"])
    if (datetime.now() - started).total_seconds() > _MAX_RUNTIME_SECONDS:
        try:
            os.kill(job["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        job["status"] = "timed_out"
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _processes.pop(job["id"], None)
        return job

    proc = _processes.get(job["id"])
    if proc is not None:
        code = proc.poll()
        if code is not None:
            job["status"] = "done" if code == 0 else "failed"
            job["exit_code"] = code
            job["finished_at"] = datetime.now().isoformat(timespec="seconds")
    elif not _pid_alive(job["pid"]):
        # Handle lost (module restarted) — process is gone so it's finished,
        # but we can't recover the real exit code from here.
        job["status"] = "done"
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return job


@mcp.tool()
def check_dev_pipeline(job_id: str = "") -> str:
    """Check the status and recent output of a dev pipeline job.

    Args:
        job_id: The job id from start_dev_pipeline. Omit to check the most
            recently started job.
    """
    jobs = _load()
    if not jobs:
        return "No dev pipeline jobs found."
    if job_id:
        job = next((j for j in jobs if j["id"] == job_id), None)
        if not job:
            return f"No job with id '{job_id}'."
    else:
        job = sorted(jobs, key=lambda j: j["started_at"])[-1]

    job = _refresh_status(job)
    _save([job if j["id"] == job["id"] else j for j in jobs])

    tail = ""
    log_path = Path(job["log_path"])
    if log_path.exists():
        lines = log_path.read_text(errors="replace").splitlines()
        tail = "\n".join(lines[-15:])

    header = f"[{job['id']}] {job['project']} — {job['status']}"
    if job["exit_code"] is not None:
        header += f" (exit {job['exit_code']})"
    return f"{header}\n\n{tail or '(no output yet)'}"


@mcp.tool()
def list_dev_pipelines(project_name: str = "", limit: int = 10) -> str:
    """List recent dev pipeline jobs, most recent first.

    Args:
        project_name: Optional project to filter by.
        limit: Max jobs to return.
    """
    jobs = [_refresh_status(j) for j in _load()]
    _save(jobs)
    if project_name:
        jobs = [j for j in jobs if project_name.lower() in j["project"].lower()]
    jobs = sorted(jobs, key=lambda j: j["started_at"], reverse=True)[:limit]
    if not jobs:
        return "No dev pipeline jobs found."
    return "\n".join(f"[{j['id']}] {j['project']} — {j['status']} ({j['started_at']})" for j in jobs)


@mcp.tool()
def kill_dev_pipeline(job_id: str) -> str:
    """Stop a running dev pipeline job.

    Args:
        job_id: The job id to stop.
    """
    jobs = _load()
    job = next((j for j in jobs if j["id"] == job_id), None)
    if not job:
        return f"No job with id '{job_id}'."
    if job["status"] != "running":
        return f"Job {job_id} isn't running (status: {job['status']})."

    try:
        os.kill(job["pid"], signal.SIGTERM)
    except ProcessLookupError:
        pass
    job["status"] = "killed"
    job["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _save(jobs)
    _processes.pop(job_id, None)
    return f"Stopped job {job_id}."


if __name__ == "__main__":
    mcp.run()
