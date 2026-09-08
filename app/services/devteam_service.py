"""Read-only access to devteam pipeline job records for the dashboard.

Jobs are written by integrations/servers/devteam_server.py's own tool calls
(start_dev_pipeline/check_dev_pipeline/etc.) into ~/.dann/devteam_jobs.json.
This module only reads that file — it does not attempt to refresh a job's
status (Popen polling / PID-liveness), since that logic only works inside
the MCP server's own process. Status here is whatever was last persisted.
"""
from integrations.servers._store import load_json

_FILE = "devteam_jobs.json"


def list_jobs(project: str | None = None, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    jobs = load_json(_FILE, [])
    if project:
        jobs = [j for j in jobs if project.lower() in j.get("project", "").lower()]
    jobs.sort(key=lambda j: j.get("started_at", ""), reverse=True)
    return jobs[offset: offset + limit], len(jobs)
