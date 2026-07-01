import { useEffect, useRef, useState } from 'react'
import { useDannStore } from '../hooks/useDannState'
import { api } from '../lib/api'
import type { PipelineStage, VoiceTurn } from '../types'

// ── Stage config ──────────────────────────────────────────────────────────────

const STAGE_INFO: Record<PipelineStage, { label: string; color: string }> = {
  idle:      { label: "Listening for 'ok Dann'", color: 'text-zinc-500' },
  wake:      { label: 'Wake word detected',       color: 'text-neon-orange' },
  recording: { label: 'Listening…',               color: 'text-neon-red' },
  thinking:  { label: 'Thinking…',                color: 'text-neon-blue' },
  speaking:  { label: 'Speaking…',                color: 'text-neon-teal' },
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
  const liveResponse  = useDannStore((s) => s.liveResponse)
  const voiceListening = useDannStore((s) => s.voiceListening)
  const running        = useDannStore((s) => s.running)
  const bottomRef     = useRef<HTMLDivElement>(null)

  const [triggering, setTriggering]     = useState(false)
  const [triggerError, setTriggerError] = useState<string | null>(null)
  const [toggling, setToggling]         = useState(false)

  // Clear error once a session starts
  useEffect(() => {
    if (pipelineStage !== 'idle') setTriggerError(null)
  }, [pipelineStage])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [voiceTurns.length, pipelineStage, liveResponse])

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

  async function handleToggleListening() {
    setToggling(true)
    setTriggerError(null)
    try {
      if (voiceListening) {
        await api.disableVoice()
      } else {
        await api.enableVoice()
      }
    } catch {
      setTriggerError('Could not reach voice assistant — is Dann running?')
    } finally {
      setToggling(false)
    }
  }

  const stageInfo = STAGE_INFO[pipelineStage]
  const inSession = pipelineStage !== 'idle'

  return (
    <div className="flex flex-col h-full">

      {/* ── Centered state display ─────────────────────────────────────────── */}
      <div className="flex flex-col items-center justify-center gap-4 py-8 border-b border-zinc-800/60 shrink-0">

        {/* Animated state orb */}
        <StateOrb stage={voiceListening && running ? pipelineStage : 'idle'} />

        {/* Mic toggle */}
        <button
          onClick={handleToggleListening}
          disabled={toggling || inSession || !running}
          title={!running ? 'Voice assistant not running' : voiceListening ? 'Click to turn off voice assistant' : 'Click to turn on voice assistant'}
          className={`
            relative flex items-center justify-center w-14 h-14 rounded-full border-2 transition-all duration-200
            disabled:opacity-40 disabled:cursor-not-allowed
            ${voiceListening && running
              ? 'border-teal-500/70 bg-teal-900/20 hover:border-teal-400 hover:bg-teal-900/30'
              : 'border-zinc-600 bg-zinc-800/40 hover:border-zinc-500 hover:bg-zinc-800/60'}
          `}
        >
          {voiceListening && running ? (
            /* Mic on */
            <svg viewBox="0 0 24 24" className="w-6 h-6 text-teal-400" fill="currentColor">
              <path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.93V20H9v2h6v-2h-2v-2.07A7 7 0 0 0 19 11h-2z"/>
            </svg>
          ) : (
            /* Mic off */
            <svg viewBox="0 0 24 24" className="w-6 h-6 text-zinc-500" fill="currentColor">
              <path d="M19 11h-1.7c0 .74-.16 1.43-.43 2.05l1.23 1.23c.56-.98.9-2.09.9-3.28zm-4.02.17c0-.06.02-.11.02-.17V5c0-1.66-1.34-3-3-3S9 3.34 9 5v.18l5.98 5.99zM4.27 3 3 4.27l6.01 6.01V11c0 1.66 1.33 3 2.99 3 .22 0 .44-.03.65-.08l1.66 1.66c-.71.33-1.5.52-2.31.52-2.76 0-5.3-2.1-5.3-5.1H5c0 3.41 2.72 6.23 6 6.72V20H9v2h6v-2h-2v-2.28c.91-.13 1.77-.45 2.54-.9L19.73 21 21 19.73 4.27 3z"/>
            </svg>
          )}
        </button>

        <div className="flex flex-col items-center gap-1">
          <span className={`text-xs tracking-widest uppercase ${voiceListening && running ? stageInfo.color : 'text-zinc-600'}`}>
            {!running
              ? 'Not running'
              : !voiceListening
              ? 'Voice off'
              : stageInfo.label}
          </span>
          {pipelineStage === 'thinking' && pendingStt && (
            <span className="text-[10px] text-zinc-500 max-w-xs truncate">
              "{pendingStt}"
            </span>
          )}
        </div>

        {pipelineStage === 'idle' && voiceListening && running && (
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
          </div>
        )}

        {triggerError && (
          <span className="text-[10px] text-neon-red">{triggerError}</span>
        )}
      </div>

      {/* ── Conversation history ───────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        {voiceTurns.length === 0 && !liveResponse ? (
          <p className="text-center text-[11px] text-zinc-700 py-8">No conversation yet</p>
        ) : (
          voiceTurns.map((turn) => <TurnCard key={turn.id} turn={turn} />)
        )}

        {/* Live streaming response — appears sentence-by-sentence as Dann speaks */}
        {liveResponse && (
          <div className="flex flex-col gap-2 border-b border-zinc-800/60 px-4 py-3">
            {pendingStt && (
              <div className="flex justify-end">
                <div className="voice-user-bubble max-w-[80%] rounded-lg rounded-tr-sm px-3 py-2 text-xs opacity-70">
                  {pendingStt}
                </div>
              </div>
            )}
            <div className="flex justify-start">
              <div className="voice-dann-bubble max-w-[80%] rounded-lg rounded-tl-sm px-3 py-2 text-xs">
                {liveResponse}
                <span className="inline-block ml-0.5 animate-pulse text-teal-400">▋</span>
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

    </div>
  )
}
