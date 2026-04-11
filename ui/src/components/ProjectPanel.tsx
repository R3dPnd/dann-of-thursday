import { useEffect, useState } from 'react'
import type { Project, RunStatus } from '../types'
import { useDannStore } from '../hooks/useDannState'
import { TerminalOutput } from './TerminalOutput'
import { api } from '../lib/api'

function RunButton({
  project,
  onRunOpened,
}: {
  project: Project
  onRunOpened: (name: string) => void
}) {
  const [status, setStatus] = useState<RunStatus | null>(null)
  const [busy, setBusy] = useState(false)

  // Poll status every 3 s while mounted
  useEffect(() => {
    let alive = true
    const poll = async () => {
      try {
        const s = await api.getRunStatus(project.name)
        if (alive) setStatus(s)
      } catch {
        // no run config — component shouldn't even render, but be safe
      }
    }
    poll()
    const id = setInterval(poll, 3000)
    return () => { alive = false; clearInterval(id) }
  }, [project.name])

  const handleStart = async () => {
    setBusy(true)
    try {
      await api.startRun(project.name)
      setStatus(await api.getRunStatus(project.name))
      onRunOpened(project.name)
    } catch (e) {
      console.error(e)
    } finally {
      setBusy(false)
    }
  }

  const handleStop = async () => {
    setBusy(true)
    try {
      await api.stopRun(project.name)
      setStatus(await api.getRunStatus(project.name))
    } catch (e) {
      console.error(e)
    } finally {
      setBusy(false)
    }
  }

  const running = status?.alive ?? false

  return (
    <div className="flex items-center gap-1.5 flex-shrink-0">
      {/* Status dot */}
      <span
        className={`inline-block h-2 w-2 rounded-full flex-shrink-0 ${
          running ? 'bg-green-400' : 'bg-zinc-600'
        }`}
        title={running ? 'Running' : 'Stopped'}
      />

      {running ? (
        <>
          <button
            onClick={() => onRunOpened(project.name)}
            className="rounded bg-zinc-800 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-700"
            title="View output"
          >
            Logs
          </button>
          <button
            onClick={handleStop}
            disabled={busy}
            className="rounded bg-red-900/60 px-2 py-1 text-xs text-red-300 hover:bg-red-800/60 disabled:opacity-50"
            title="Stop process"
          >
            Stop
          </button>
        </>
      ) : (
        <button
          onClick={handleStart}
          disabled={busy}
          className="rounded bg-green-900/60 px-2 py-1 text-xs text-green-300 hover:bg-green-800/60 disabled:opacity-50"
          title={`Run: ${project.run}`}
        >
          Run
        </button>
      )}
    </div>
  )
}

function ProjectCard({
  project,
  isActive,
  onOpenTerminal,
  onRunOpened,
}: {
  project: Project
  isActive: boolean
  onOpenTerminal: (name: string) => void
  onRunOpened: (name: string) => void
}) {
  const codeHistory = useDannStore((s) => s.codeHistory)
  const turns = codeHistory[project.name] ?? []
  const [expanded, setExpanded] = useState(false)

  const lastTurn = turns[turns.length - 1]

  return (
    <div
      id={`project-${project.name}`}
      className={`rounded-lg border transition-colors ${
        isActive
          ? 'border-blue-600 bg-blue-950/30'
          : 'border-zinc-800 bg-zinc-900 hover:border-zinc-700'
      }`}
    >
      {/* Card header */}
      <div className="flex items-center gap-3 px-4 py-3">
        {/* Active indicator */}
        <span
          className={`inline-block h-2 w-2 flex-shrink-0 rounded-full ${
            isActive ? 'bg-blue-400 pulse-dot' : 'bg-zinc-700'
          }`}
        />

        {/* Name + path */}
        <div className="min-w-0 flex-1">
          <p className="truncate font-medium text-zinc-100">{project.name}</p>
          <p className="truncate text-xs text-zinc-500" title={project.path}>
            {project.path}
          </p>
        </div>

        {/* Turn count */}
        {turns.length > 0 && (
          <span className="flex-shrink-0 rounded-full bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400">
            {turns.length}
          </span>
        )}

        {/* Last active */}
        {lastTurn && (
          <span className="hidden flex-shrink-0 text-xs text-zinc-600 sm:block">
            {lastTurn.timestamp.toLocaleTimeString()}
          </span>
        )}

        {/* Run button (only if project has a run config) */}
        {project.run && (
          <RunButton project={project} onRunOpened={onRunOpened} />
        )}

        {/* Open terminal tab */}
        <button
          onClick={() => onOpenTerminal(project.name)}
          className="flex-shrink-0 rounded bg-zinc-800 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-700"
          title="Open Claude Code in a browser terminal tab"
        >
          Open
        </button>

        {/* Expand toggle */}
        <button
          onClick={() => setExpanded((v) => !v)}
          className="flex-shrink-0 text-zinc-500 hover:text-zinc-300"
          aria-label={expanded ? 'Collapse' : 'Expand'}
        >
          <svg
            className={`h-4 w-4 transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {/* Terminal output (expanded) */}
      {expanded && <TerminalOutput turns={turns} />}
    </div>
  )
}

export function ProjectPanel({
  onOpenTerminal,
  onRunOpened,
}: {
  onOpenTerminal: (name: string) => void
  onRunOpened: (name: string) => void
}) {
  const projects = useDannStore((s) => s.projects)
  const activeProject = useDannStore((s) => s.project)

  if (projects.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-zinc-600">
        <p className="text-sm">No projects found.</p>
        <p className="mt-1 text-xs">Projects are configured in config.yaml.</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      {projects.map((project) => (
        <ProjectCard
          key={project.name}
          project={project}
          isActive={activeProject === project.name}
          onOpenTerminal={onOpenTerminal}
          onRunOpened={onRunOpened}
        />
      ))}
    </div>
  )
}
