import { useEffect, useRef, useState } from 'react'
import { useDannStore } from '../hooks/useDannState'
import { api } from '../lib/api'
import type { PipelineStage, VoiceTurn } from '../types'

// ── Stage config ──────────────────────────────────────────────────────────────

const STAGE_INFO: Record<PipelineStage, { label: string; color: string }> = {
  idle:      { label: 'Ready',                color: 'text-zinc-500' },
  wake:      { label: 'Wake word detected',   color: 'text-neon-orange' },
  recording: { label: 'Listening…',           color: 'text-neon-red' },
  thinking:  { label: 'Thinking…',            color: 'text-neon-blue' },
  speaking:  { label: 'Speaking…',            color: 'text-neon-teal' },
}

// Equalizer bar timings — offset durations give a natural, uneven feel
const EQ_BARS: { delay: string; duration: string }[] = [
  { delay: '0s',     duration: '0.55s' },
  { delay: '0.15s',  duration: '0.80s' },
  { delay: '0.05s',  duration: '0.50s' },
  { delay: '0.25s',  duration: '0.70s' },
  { delay: '0.10s',  duration: '0.90s' },
]

// ── Sub-components ────────────────────────────────────────────────────────────

function StateOrb({ stage }: { stage: PipelineStage }) {
  if (stage === 'speaking') {
    return (
      <div className="dann-equalizer">
        {EQ_BARS.map((b, i) => (
          <div
            key={i}
            className="dann-eq-bar"
            style={{ animationDelay: b.delay, animationDuration: b.duration }}
          />
        ))}
      </div>
    )
  }
  return <div className={`dann-orb dann-orb-${stage}`} />
}

function relTime(date: Date): string {
  const diff = (Date.now() - date.getTime()) / 1000
  if (diff < 60) return `${Math.floor(diff)}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  return date.toLocaleTimeString()
}

function TurnCard({ turn }: { turn: VoiceTurn }) {
  return (
    <div className="flex flex-col gap-2 border-b border-zinc-800 px-4 py-3 last:border-b-0">
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-zinc-600">{relTime(turn.timestamp)}</span>
        {turn.mode === 'code' && turn.project && (
          <span className="rounded bg-blue-900/60 px-1.5 py-0.5 text-[10px] text-blue-300">
            {turn.project}
          </span>
        )}
      </div>

      {turn.userText && (
        <div className="flex justify-end">
          <div className="voice-user-bubble max-w-[80%] rounded-lg rounded-tr-sm px-3 py-2 text-xs">
            {turn.userText}
          </div>
        </div>
      )}

      {turn.dannText && (
        <div className="flex justify-start">
          <div className="voice-dann-bubble max-w-[80%] rounded-lg rounded-tl-sm px-3 py-2 text-xs">
            {turn.dannText}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ConversationPanel() {
  const voiceTurns    = useDannStore((s) => s.voiceTurns)
  const pipelineStage = useDannStore((s) => s.pipelineStage)
  const pendingStt    = useDannStore((s) => s._pendingStt)
  const bottomRef     = useRef<HTMLDivElement>(null)

  const [triggering, setTriggering]     = useState(false)
  const [triggerError, setTriggerError] = useState<string | null>(null)

  // Clear error once a session starts
  useEffect(() => {
    if (pipelineStage !== 'idle') setTriggerError(null)
  }, [pipelineStage])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [voiceTurns.length, pipelineStage])

  async function handleTrigger() {
    setTriggering(true)
    setTriggerError(null)
    try {
      await api.triggerVoice()
    } catch {
      setTriggerError('Voice not available — is Dann running?')
    } finally {
      setTriggering(false)
    }
  }

  const stageInfo = STAGE_INFO[pipelineStage]

  return (
    <div className="flex flex-col h-full">

      {/* ── Centered state display ─────────────────────────────────────────── */}
      <div className="flex flex-col items-center justify-center gap-4 py-8 border-b border-zinc-800/60 shrink-0">

        <StateOrb stage={pipelineStage} />

        <div className="flex flex-col items-center gap-1">
          <span className={`text-xs tracking-widest uppercase ${stageInfo.color}`}>
            {stageInfo.label}
          </span>
          {pipelineStage === 'thinking' && pendingStt && (
            <span className="text-[10px] text-zinc-500 max-w-xs truncate">
              "{pendingStt}"
            </span>
          )}
        </div>

        {pipelineStage === 'idle' && (
          <div className="flex flex-col items-center gap-2">
            <button
              onClick={handleTrigger}
              disabled={triggering}
              className={`
                px-8 py-2 text-xs tracking-widest border border-teal-500/60 text-neon-teal
                hover:border-teal-400/80 hover:bg-teal-900/20
                transition-all duration-200
                disabled:opacity-40 disabled:cursor-not-allowed
                ${triggering ? 'animate-pulse' : ''}
              `}
            >
              {triggering ? 'Starting…' : 'Talk to Dann'}
            </button>
            {triggerError && (
              <span className="text-[10px] text-neon-red">{triggerError}</span>
            )}
          </div>
        )}
      </div>

      {/* ── Conversation history ───────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        {voiceTurns.length === 0 ? (
          <p className="text-center text-[11px] text-zinc-700 py-8">No conversation yet</p>
        ) : (
          voiceTurns.map((turn) => <TurnCard key={turn.id} turn={turn} />)
        )}
        <div ref={bottomRef} />
      </div>

    </div>
  )
}
