import { useEffect, useState } from 'react'
import { api } from '../../lib/api'
import type { DevTeamJob } from '../../types'

const POLL_MS = 5000

const STATUS_COLOR: Record<DevTeamJob['status'], string> = {
  running: 'text-neon-blue',
  done: 'text-neon-green',
  failed: 'text-neon-red',
  killed: 'text-zinc-500',
  timed_out: 'text-neon-orange',
}

function relTime(iso: string): string {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000
  if (diff < 60) return `${Math.floor(diff)}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  return `${Math.floor(diff / 3600)}h ago`
}

export function DevTeamJobs() {
  const [jobs, setJobs] = useState<DevTeamJob[]>([])

  useEffect(() => {
    let cancelled = false
    const load = () => {
      api.getDevTeamJobs({ limit: 10 })
        .then(r => { if (!cancelled) setJobs(r.jobs) })
        .catch(() => {})
    }
    load()
    const id = setInterval(load, POLL_MS)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  if (jobs.length === 0) return null

  return (
    <div className="flex flex-col gap-1.5 w-full max-w-xl px-4">
      <span className="text-[9px] tracking-widest uppercase text-zinc-600">Devteam pipelines</span>
      {jobs.map(job => (
        <div key={job.id} className="flex items-center gap-2 text-[11px] border-b border-zinc-800/50 pb-1.5">
          <span className={`font-mono uppercase text-[9px] ${STATUS_COLOR[job.status]}`}>{job.status}</span>
          <span className="text-zinc-300 truncate">{job.project}</span>
          <span className="text-zinc-600 truncate flex-1">{job.task}</span>
          <span className="text-zinc-700 shrink-0">{relTime(job.started_at)}</span>
        </div>
      ))}
    </div>
  )
}
