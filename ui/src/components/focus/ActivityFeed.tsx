import { useDannStore } from '../../hooks/useDannState'
import { TraceBar } from '../VoicePanel'

export function ActivityFeed() {
  const mode        = useDannStore(s => s.mode)
  const project     = useDannStore(s => s.project)
  const pendingStt  = useDannStore(s => s._pendingStt)
  const liveResponse = useDannStore(s => s.liveResponse)
  const voiceTurns  = useDannStore(s => s.voiceTurns)

  const lastTurn = voiceTurns[voiceTurns.length - 1]

  return (
    <div className="flex flex-col items-center gap-3 w-full max-w-xl px-4">
      <div className="flex items-center gap-2 text-[10px] tracking-widest uppercase text-zinc-500">
        <span>Mode: <span className="text-zinc-300">{mode}</span></span>
        {project && (
          <>
            <span className="text-zinc-700">|</span>
            <span>Project: <span className="text-zinc-300">{project}</span></span>
          </>
        )}
      </div>

      {(pendingStt || liveResponse) && (
        <div className="flex flex-col items-center gap-1.5 text-center">
          {pendingStt && (
            <p className="text-sm text-zinc-400">"{pendingStt}"</p>
          )}
          {liveResponse && (
            <p className="text-base text-zinc-100">
              {liveResponse}
              <span className="inline-block ml-0.5 animate-pulse text-teal-400">▋</span>
            </p>
          )}
        </div>
      )}

      {!pendingStt && !liveResponse && lastTurn && (
        <div className="flex flex-col items-center gap-1.5 text-center opacity-70">
          {lastTurn.userText && <p className="text-sm text-zinc-500">"{lastTurn.userText}"</p>}
          {lastTurn.dannText && <p className="text-base text-zinc-300">{lastTurn.dannText}</p>}
        </div>
      )}

      {lastTurn?.trace && <TraceBar trace={lastTurn.trace} />}

      {!lastTurn && !pendingStt && !liveResponse && (
        <p className="text-xs text-zinc-700">No activity yet — say "ok Dann" to begin.</p>
      )}
    </div>
  )
}
