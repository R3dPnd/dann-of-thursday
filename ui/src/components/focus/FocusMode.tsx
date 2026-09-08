import { useEffect } from 'react'
import { useDannStore } from '../../hooks/useDannState'
import LogPanel from '../LogPanel'
import { StateOrb } from '../voice/StateOrb'
import { ActivityFeed } from './ActivityFeed'
import { DevTeamJobs } from './DevTeamJobs'

export function FocusMode({ active, onExit }: { active: boolean; onExit: () => void }) {
  const pipelineStage = useDannStore(s => s.pipelineStage)
  const voiceListening = useDannStore(s => s.voiceListening)
  const running = useDannStore(s => s.running)

  useEffect(() => {
    if (!active) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onExit() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [active, onExit])

  if (!active) return null

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-zinc-950">
      <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800/60 shrink-0">
        <span className="text-[10px] tracking-widest uppercase text-zinc-500">Focus Mode</span>
        <button
          onClick={onExit}
          className="text-[10px] tracking-widest uppercase text-zinc-500 hover:text-zinc-300 transition-colors"
        >
          Esc to exit
        </button>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center gap-8 overflow-y-auto py-8 min-h-0">
        <StateOrb stage={pipelineStage} active={running && voiceListening} size={320} />
        <ActivityFeed />
        <DevTeamJobs />
      </div>

      <div className="shrink-0">
        <LogPanel />
      </div>
    </div>
  )
}
