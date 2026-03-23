import { useState } from 'react'
import type { Project } from '../types'
import { useDannStore } from '../hooks/useDannState'
import { TerminalOutput } from './TerminalOutput'

function ProjectCard({
  project,
  isActive,
  onOpenTerminal,
}: {
  project: Project
  isActive: boolean
  onOpenTerminal: (name: string) => void
}) {
  const codeHistory = useDannStore((s) => s.codeHistory)
  const turns = codeHistory[project.name] ?? []
  const [expanded, setExpanded] = useState(false)

  const lastTurn = turns[turns.length - 1]

  const handleOpen = () => onOpenTerminal(project.name)

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
          <p
            className="truncate text-xs text-zinc-500"
            title={project.path}
          >
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

        {/* Open terminal tab */}
        <button
          onClick={handleOpen}
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

export function ProjectPanel({ onOpenTerminal }: { onOpenTerminal: (name: string) => void }) {
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
        />
      ))}
    </div>
  )
}
