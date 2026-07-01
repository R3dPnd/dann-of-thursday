import { useEffect, useRef, useState } from 'react'
import { useDannStore } from '../hooks/useDannState'
import { api } from '../lib/api'
import type { PipelineStage, RawEvent, TurnTrace, VoiceTurn } from '../types'

// ── Stage config ──────────────────────────────────────────────────────────────

const STAGE_INFO: Record<PipelineStage, { label: string; color: string }> = {
  idle:      { label: "Listening for 'ok Dann'", color: 'text-zinc-500' },
  wake:      { label: 'Wake word detected',       color: 'text-neon-orange' },
  recording: { label: 'Listening…',               color: 'text-neon-red' },
  thinking:  { label: 'Thinking…',                color: 'text-neon-blue' },
  speaking:  { label: 'Speaking…',                color: 'text-neon-teal' },
}

const EQ_BARS: { delay: string; duration: string }[] = [
  { delay: '0s',    duration: '0.55s' },
  { delay: '0.15s', duration: '0.80s' },
  { delay: '0.05s', duration: '0.50s' },
  { delay: '0.25s', duration: '0.70s' },
  { delay: '0.10s', duration: '0.90s' },
]

// ── Orb ───────────────────────────────────────────────────────────────────────

function StateOrb({ stage, active }: { stage: PipelineStage; active: boolean }) {
  if (!active) return <div className="dann-orb dann-orb-idle" />
  if (stage === 'speaking') {
    return (
      <div className="dann-equalizer">
        {EQ_BARS.map((b, i) => (
          <div key={i} className="dann-eq-bar" style={{ animationDelay: b.delay, animationDuration: b.duration }} />
        ))}
      </div>
    )
  }
  return <div className={`dann-orb dann-orb-${stage}`} />
}

// ── Stage duration timer ──────────────────────────────────────────────────────

function StageDuration({ stageEnteredAt, stage }: { stageEnteredAt: number; stage: PipelineStage }) {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    if (stage === 'idle') { setElapsed(0); return }
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - stageEnteredAt) / 100) / 10)
    }, 100)
    return () => clearInterval(id)
  }, [stage, stageEnteredAt])

  if (stage === 'idle' || elapsed < 0.1) return null
  return (
    <span className="text-[10px] text-zinc-600 tabular-nums ml-1">{elapsed.toFixed(1)}s</span>
  )
}

// ── Per-turn trace bar ────────────────────────────────────────────────────────

function TraceBar({ trace }: { trace: TurnTrace }) {
  const steps: { label: string; ms: number | null; color: string }[] = [
    { label: 'STT',  ms: trace.stt_ms,  color: 'text-neon-orange' },
    { label: 'LLM',  ms: trace.llm_ms,  color: 'text-neon-blue'   },
    { label: 'TTS',  ms: trace.tts_ms,  color: 'text-neon-teal'   },
    { label: 'CODE', ms: trace.code_ms, color: 'text-neon-violet'  },
  ]
  const visible = steps.filter(s => s.ms != null && s.ms > 0)
  const total = visible.reduce((acc, s) => acc + (s.ms ?? 0), 0)
  const statusColor = trace.status === 'ok' ? 'text-neon-green' : trace.status === 'error' ? 'text-neon-red' : 'text-zinc-500'

  return (
    <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 pl-0.5">
      {visible.map(s => (
        <span key={s.label} className={`text-[9px] font-mono ${s.color}`}>
          {s.label} {s.ms}ms
        </span>
      ))}
      {total > 0 && (
        <span className="text-[9px] font-mono text-zinc-600">∑ {total}ms</span>
      )}
      <span className={`text-[9px] font-mono ${statusColor} ml-auto`}>{trace.status}</span>
    </div>
  )
}

// ── Turn card ─────────────────────────────────────────────────────────────────

function relTime(date: Date): string {
  const diff = (Date.now() - date.getTime()) / 1000
  if (diff < 60) return `${Math.floor(diff)}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function TurnCard({ turn }: { turn: VoiceTurn }) {
  const [showTrace, setShowTrace] = useState(false)

  return (
    <div className="flex flex-col gap-1.5 border-b border-zinc-800/50 px-3 py-2.5 last:border-b-0">
      <div className="flex items-center gap-2">
        <span className="text-[9px] text-zinc-700">{relTime(turn.timestamp)}</span>
        {turn.mode === 'code' && turn.project && (
          <span className="rounded bg-blue-900/50 px-1 py-0.5 text-[9px] text-blue-400">{turn.project}</span>
        )}
        {turn.trace && (
          <button
            onClick={() => setShowTrace(v => !v)}
            className="ml-auto text-[9px] text-zinc-700 hover:text-zinc-400 transition-colors"
          >
            {showTrace ? '▲' : '▼'} trace
          </button>
        )}
      </div>

      {turn.userText && (
        <div className="flex justify-end">
          <div className="voice-user-bubble max-w-[92%] rounded-lg rounded-tr-sm px-2.5 py-1.5 text-[11px]">
            {turn.userText}
          </div>
        </div>
      )}

      {turn.dannText && (
        <div className="flex justify-start">
          <div className="voice-dann-bubble max-w-[92%] rounded-lg rounded-tl-sm px-2.5 py-1.5 text-[11px]">
            {turn.dannText}
          </div>
        </div>
      )}

      {showTrace && turn.trace && <TraceBar trace={turn.trace} />}
    </div>
  )
}

// ── Debug event log ───────────────────────────────────────────────────────────

function DebugLog({ events }: { events: RawEvent[] }) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'instant' })
  }, [events.length])

  return (
    <div className="max-h-44 overflow-y-auto font-mono bg-zinc-950">
      {events.slice(-100).map((e, i) => (
        <div key={i} className="flex gap-2 px-3 py-0.5 hover:bg-zinc-900/60 text-[9px]">
          <span className="text-zinc-700 shrink-0 tabular-nums">
            {new Date(e.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </span>
          <span className="text-zinc-500 shrink-0 min-w-[80px]">{e.type}</span>
          <span className="text-zinc-700 truncate">{e.summary}</span>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}

// ── Mic icons ─────────────────────────────────────────────────────────────────

function MicOn()  {
  return (
    <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="currentColor">
      <path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.93V20H9v2h6v-2h-2v-2.07A7 7 0 0 0 19 11h-2z"/>
    </svg>
  )
}
function MicOff() {
  return (
    <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="currentColor">
      <path d="M19 11h-1.7c0 .74-.16 1.43-.43 2.05l1.23 1.23c.56-.98.9-2.09.9-3.28zm-4.02.17c0-.06.02-.11.02-.17V5c0-1.66-1.34-3-3-3S9 3.34 9 5v.18l5.98 5.99zM4.27 3 3 4.27l6.01 6.01V11c0 1.66 1.33 3 2.99 3 .22 0 .44-.03.65-.08l1.66 1.66c-.71.33-1.5.52-2.31.52-2.76 0-5.3-2.1-5.3-5.1H5c0 3.41 2.72 6.23 6 6.72V20H9v2h6v-2h-2v-2.28c.91-.13 1.77-.45 2.54-.9L19.73 21 21 19.73 4.27 3z"/>
    </svg>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────

export function VoicePanel() {
  const voiceTurns     = useDannStore(s => s.voiceTurns)
  const pipelineStage  = useDannStore(s => s.pipelineStage)
  const stageEnteredAt = useDannStore(s => s.stageEnteredAt)
  const pendingStt     = useDannStore(s => s._pendingStt)
  const liveResponse   = useDannStore(s => s.liveResponse)
  const voiceListening = useDannStore(s => s.voiceListening)
  const running        = useDannStore(s => s.running)
  const wsConnected    = useDannStore(s => s.wsConnected)
  const rawEvents      = useDannStore(s => s.rawEvents)
  const prependTurns   = useDannStore(s => s.prependTurns)

  const [collapsed, setCollapsed]   = useState(false)
  const [debugOpen, setDebugOpen]   = useState(false)
  const [triggering, setTriggering] = useState(false)
  const [toggling, setToggling]     = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  const inSession = pipelineStage !== 'idle'
  const active    = running && voiceListening
  const stageInfo = STAGE_INFO[pipelineStage]

  // Load persisted history once on mount
  useEffect(() => {
    api.getHistory(150).then(records => {
      if (!records.length) return
      const turns: VoiceTurn[] = records
        .filter(r => r['user_text'] || r['dann_text'])
        .map(r => ({
          id: `hist-${r['started_at'] ?? r['timestamp']}-${r['session_id'] ?? Math.random()}`,
          sessionId: (r['session_id'] as string | null) ?? null,
          timestamp: new Date(r['timestamp'] as string),
          userText:  (r['user_text']  as string) ?? '',
          dannText:  (r['dann_text']  as string) ?? '',
          mode:      ((r['mode'] as string) ?? 'normal') as 'normal' | 'code',
          project:   r['project'] as string | undefined,
          trace: (r['stt_ms'] != null || r['llm_ms'] != null) ? {
            stt_ms:  (r['stt_ms']  as number | null) ?? null,
            llm_ms:  (r['llm_ms']  as number | null) ?? null,
            tts_ms:  (r['tts_ms']  as number | null) ?? null,
            code_ms: (r['code_ms'] as number | null) ?? null,
            status:  (r['status']  as string)        ?? 'ok',
          } : undefined,
        }))
      turns.reverse() // API returns newest-first; display oldest-first
      prependTurns(turns)
    }).catch(() => {})
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Scroll to bottom on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [voiceTurns.length, liveResponse])

  async function handleTrigger() {
    setTriggering(true)
    try { await api.triggerVoice() } catch { /* silent */ } finally { setTriggering(false) }
  }

  async function handleToggle() {
    setToggling(true)
    try {
      if (voiceListening) await api.disableVoice()
      else await api.enableVoice()
    } catch { /* silent */ } finally { setToggling(false) }
  }

  // Collapsed — thin strip with status dot and expand button
  if (collapsed) {
    return (
      <div className="flex flex-col items-center py-3 gap-3 w-8 shrink-0 border-l border-zinc-800/60 bg-zinc-950">
        <button
          onClick={() => setCollapsed(false)}
          title="Expand voice panel"
          className="text-zinc-600 hover:text-zinc-400 transition-colors"
        >
          ‹
        </button>
        <div
          className="w-2 h-2 rounded-full"
          style={{
            background:  active ? '#00ffdd' : wsConnected ? '#333' : '#ff3377',
            boxShadow:   active ? '0 0 6px #00ffdd88' : 'none',
          }}
        />
      </div>
    )
  }

  return (
    <div className="flex flex-col w-72 shrink-0 border-l border-zinc-800/60 bg-zinc-950 overflow-hidden">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800/60 shrink-0">
        <span className="text-[9px] text-zinc-500 tracking-widest font-bold uppercase">Voice</span>
        <div className="flex items-center gap-2">
          {!wsConnected && (
            <span className="text-[9px] text-neon-red animate-pulse tracking-widest">OFFLINE</span>
          )}
          <button
            onClick={() => setCollapsed(true)}
            title="Collapse panel"
            className="text-zinc-700 hover:text-zinc-400 transition-colors text-base leading-none"
          >
            ›
          </button>
        </div>
      </div>

      {/* ── Orb + stage ────────────────────────────────────────────────────── */}
      <div className="flex flex-col items-center gap-2 py-5 border-b border-zinc-800/60 shrink-0">
        <StateOrb stage={pipelineStage} active={active} />

        <div className="flex items-center">
          <span className={`text-[10px] tracking-widest uppercase ${active ? stageInfo.color : 'text-zinc-600'}`}>
            {!running ? 'Offline' : !voiceListening ? 'Voice off' : stageInfo.label}
          </span>
          {active && (
            <StageDuration stageEnteredAt={stageEnteredAt} stage={pipelineStage} />
          )}
        </div>

        {pipelineStage === 'thinking' && pendingStt && (
          <span className="text-[10px] text-zinc-500 max-w-[220px] truncate px-2 text-center">
            "{pendingStt}"
          </span>
        )}

        {/* Controls */}
        <div className="flex items-center gap-2 mt-1">
          {pipelineStage === 'idle' && voiceListening && running && (
            <button
              onClick={handleTrigger}
              disabled={triggering}
              className="text-[9px] tracking-widest px-3 py-1 border border-teal-600/50 text-teal-500 hover:border-teal-400/70 hover:text-teal-300 hover:bg-teal-900/10 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {triggering ? '…' : 'Talk'}
            </button>
          )}

          <button
            onClick={handleToggle}
            disabled={toggling || inSession || !running}
            title={!running ? 'Dann not running' : voiceListening ? 'Mute' : 'Unmute'}
            className={`flex items-center justify-center w-7 h-7 rounded-full border transition-all disabled:opacity-30 disabled:cursor-not-allowed ${
              voiceListening && running
                ? 'border-teal-600/60 text-teal-500 hover:border-teal-400 hover:text-teal-300 hover:bg-teal-900/10'
                : 'border-zinc-700 text-zinc-600 hover:border-zinc-500 hover:text-zinc-400'
            }`}
          >
            {voiceListening && running ? <MicOn /> : <MicOff />}
          </button>
        </div>
      </div>

      {/* ── Conversation history ────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto min-h-0">
        {voiceTurns.length === 0 && !liveResponse ? (
          <p className="text-center text-[10px] text-zinc-700 py-8">No conversations yet</p>
        ) : (
          voiceTurns.map(turn => <TurnCard key={turn.id} turn={turn} />)
        )}

        {/* Live streaming response */}
        {liveResponse && (
          <div className="flex flex-col gap-1.5 border-b border-zinc-800/50 px-3 py-2.5">
            {pendingStt && (
              <div className="flex justify-end">
                <div className="voice-user-bubble max-w-[92%] rounded-lg rounded-tr-sm px-2.5 py-1.5 text-[11px] opacity-70">
                  {pendingStt}
                </div>
              </div>
            )}
            <div className="flex justify-start">
              <div className="voice-dann-bubble max-w-[92%] rounded-lg rounded-tl-sm px-2.5 py-1.5 text-[11px]">
                {liveResponse}
                <span className="inline-block ml-0.5 animate-pulse text-teal-400">▋</span>
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── Debug panel (collapsible) ───────────────────────────────────────── */}
      <div className="border-t border-zinc-800/60 shrink-0">
        <button
          onClick={() => setDebugOpen(v => !v)}
          className="w-full flex items-center justify-between px-3 py-1.5 text-[9px] text-zinc-700 hover:text-zinc-500 transition-colors tracking-widest"
        >
          <span>EVENTS</span>
          <span className="font-mono">{debugOpen ? '▼' : '▲'} {rawEvents.length}</span>
        </button>
        {debugOpen && <DebugLog events={rawEvents} />}
      </div>
    </div>
  )
}
