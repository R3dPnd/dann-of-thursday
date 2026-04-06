import { useDannStore } from '../hooks/useDannState'
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
  const cfg = modeConfig(mode, project)
  const stage = STAGE_CONFIG[pipelineStage]

  const scrollToActive = () => {
    if (project) {
      document.getElementById(`project-${project}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }

  return (
    <header className="status-bar sticky top-0 z-30 flex items-center justify-between bg-zinc-950/90 px-4 py-2 backdrop-blur">
      {/* Left: app name */}
      <span className="text-sm font-semibold tracking-wide text-neon-blue">Dann</span>

      {/* Centre: mode pill + pipeline stage */}
      <div className="flex items-center gap-3">
        <button
          onClick={scrollToActive}
          className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors ${cfg.pillClass} ${mode === 'code' ? 'cursor-pointer hover:opacity-80' : 'cursor-default'}`}
        >
          {cfg.dot && (
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-300 pulse-dot" />
          )}
          {cfg.label}
        </button>

        {stage.label && (
          <span className={`text-xs font-medium ${stage.color} animate-pulse`}>
            {stage.label}
          </span>
        )}
      </div>

      {/* Right: WS connection indicator */}
      <div className="flex items-center gap-1.5 text-xs text-zinc-500">
        <span
          className={`inline-block h-1.5 w-1.5 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-zinc-600'}`}
        />
        {wsConnected ? 'connected' : 'disconnected'}
      </div>
    </header>
  )
}
