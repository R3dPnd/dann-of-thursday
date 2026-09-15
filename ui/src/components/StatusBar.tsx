import { useState } from 'react'
import { useDannStore } from '../hooks/useDannState'
import { api } from '../lib/api'
import type { Mode, PipelineStage } from '../types'

interface ModeConfig {
  label: string
  pillClass: string
  dot: boolean
}

function modeConfig(mode: Mode, project: string | null): ModeConfig {
  switch (mode) {
    case 'normal':
      return { label: 'Listening', pillClass: 'bg-green-600 text-green-50', dot: true }
    case 'code':
      return {
        label: `Code${project ? `: ${project}` : ''}`,
        pillClass: 'bg-violet-600 text-violet-100',
        dot: false,
      }
    case 'idle':
    default:
      return { label: 'Idle', pillClass: 'bg-zinc-700 text-zinc-300', dot: false }
  }
}

const STAGE_CONFIG: Record<PipelineStage, { label: string; color: string }> = {
  idle:        { label: '',            color: '' },
  wake:        { label: '⚡ Wake',     color: 'text-neon-orange' },
  recording:   { label: '● REC',       color: 'text-neon-red' },
  thinking:    { label: '… Thinking',  color: 'text-neon-blue' },
  speaking:    { label: '▶ Speaking',  color: 'text-neon-green' },
}

export function StatusBar() {
  const mode = useDannStore((s) => s.mode)
  const project = useDannStore((s) => s.project)
  const wsConnected = useDannStore((s) => s.wsConnected)
  const pipelineStage = useDannStore((s) => s.pipelineStage)
  const modules = useDannStore((s) => s.modules)
  const cfg = modeConfig(mode, project)
  const stage = STAGE_CONFIG[pipelineStage]
  const [restarting, setRestarting] = useState(false)

  // Always-on modules (currently just "projects") are foundational, not
  // interesting to call out here — this is specifically "what did Dann
  // start up on demand for this conversation."
  const activeModules = modules.filter((m) => m.enabled && !m.always_on)

  const handleRestart = async () => {
    if (!window.confirm('Restart Dann? This will drop any active voice session and reconnect the dashboard.')) return
    setRestarting(true)
    try {
      await api.restartDann()
    } catch {
      // The process is replacing itself — a failed fetch here is expected, not an error.
    }
  }

  return (
    <header className="status-bar sticky top-0 z-30 flex items-center justify-between bg-zinc-950/90 px-4 py-2 backdrop-blur">
      {/* Left: app name */}
      <span className="text-sm font-semibold tracking-wide text-neon-blue">Dann</span>

      {/* Centre: mode pill + pipeline stage */}
      <div className="flex items-center gap-3">
        <span
          className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${cfg.pillClass}`}
        >
          {cfg.dot && (
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-300 pulse-dot" />
          )}
          {cfg.label}
        </span>

        {stage.label && (
          <span className={`text-xs font-medium ${stage.color} animate-pulse`}>
            {stage.label}
          </span>
        )}
      </div>

      {/* Right: active modules + restart + WS connection indicator */}
      <div className="flex items-center gap-3">
        {activeModules.length > 0 && (
          <div className="flex items-center gap-1.5" title="Modules Dann currently has active">
            {activeModules.map((m) => (
              <span
                key={m.name}
                className="flex items-center gap-1 rounded-full bg-blue-950/40 px-2 py-0.5 text-[10px] text-blue-300"
              >
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-blue-400 pulse-dot" />
                {m.name}
              </span>
            ))}
          </div>
        )}

        <button
          onClick={handleRestart}
          disabled={restarting}
          title="Restart Dann"
          className="text-xs text-zinc-600 hover:text-zinc-300 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {restarting ? 'restarting…' : '⟳'}
        </button>

        {wsConnected ? (
          <div className="flex items-center gap-1.5 text-xs text-zinc-600">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-500/70" />
            connected
          </div>
        ) : (
          <div className="flex items-center gap-1.5 text-xs text-neon-red animate-pulse">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-current" />
            offline
          </div>
        )}
      </div>
    </header>
  )
}
