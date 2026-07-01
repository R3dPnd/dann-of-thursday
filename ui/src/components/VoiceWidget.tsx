import { useState } from 'react'
import { useDannStore } from '../hooks/useDannState'
import { api } from '../lib/api'
import type { PipelineStage } from '../types'

// ── Mini orb ──────────────────────────────────────────────────────────────────

const ORB_CONFIG: Record<PipelineStage, { border: string; glow: string; pulse: boolean; spin: boolean }> = {
  idle:      { border: '#00ddff33', glow: 'none',                       pulse: true,  spin: false },
  wake:      { border: '#ff9900cc', glow: '0 0 10px #ff990088',         pulse: true,  spin: false },
  recording: { border: '#ff3377cc', glow: '0 0 10px #ff337788',         pulse: true,  spin: false },
  thinking:  { border: '#4488ff',   glow: '0 0 10px #4488ff88',         pulse: false, spin: true  },
  speaking:  { border: '#00ffddcc', glow: '0 0 10px #00ffdd88',         pulse: true,  spin: false },
}

function MiniOrb({ stage }: { stage: PipelineStage }) {
  const cfg = ORB_CONFIG[stage]
  return (
    <div
      className={`w-5 h-5 rounded-full border-2 shrink-0 transition-all duration-300 ${cfg.pulse ? 'animate-pulse' : ''} ${cfg.spin ? 'animate-spin' : ''}`}
      style={{ borderColor: cfg.border, boxShadow: cfg.glow }}
    />
  )
}

// ── Stage text ────────────────────────────────────────────────────────────────

const STAGE_LABEL: Record<PipelineStage, { text: string; color: string }> = {
  idle:      { text: "Waiting",   color: 'text-zinc-600' },
  wake:      { text: 'Wake!',     color: 'text-neon-orange' },
  recording: { text: 'Listening', color: 'text-neon-red' },
  thinking:  { text: 'Thinking',  color: 'text-neon-blue' },
  speaking:  { text: 'Speaking',  color: 'text-neon-teal' },
}

// ── Icons ─────────────────────────────────────────────────────────────────────

function MicOnIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="currentColor">
      <path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.93V20H9v2h6v-2h-2v-2.07A7 7 0 0 0 19 11h-2z"/>
    </svg>
  )
}

function MicOffIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="currentColor">
      <path d="M19 11h-1.7c0 .74-.16 1.43-.43 2.05l1.23 1.23c.56-.98.9-2.09.9-3.28zm-4.02.17c0-.06.02-.11.02-.17V5c0-1.66-1.34-3-3-3S9 3.34 9 5v.18l5.98 5.99zM4.27 3 3 4.27l6.01 6.01V11c0 1.66 1.33 3 2.99 3 .22 0 .44-.03.65-.08l1.66 1.66c-.71.33-1.5.52-2.31.52-2.76 0-5.3-2.1-5.3-5.1H5c0 3.41 2.72 6.23 6 6.72V20H9v2h6v-2h-2v-2.28c.91-.13 1.77-.45 2.54-.9L19.73 21 21 19.73 4.27 3z"/>
    </svg>
  )
}

// ── Widget ────────────────────────────────────────────────────────────────────

export function VoiceWidget() {
  const pipelineStage  = useDannStore((s) => s.pipelineStage)
  const liveResponse   = useDannStore((s) => s.liveResponse)
  const pendingStt     = useDannStore((s) => s._pendingStt)
  const voiceListening = useDannStore((s) => s.voiceListening)
  const running        = useDannStore((s) => s.running)

  const [triggering, setTriggering] = useState(false)
  const [toggling, setToggling]     = useState(false)

  const inSession = pipelineStage !== 'idle'
  const stageLabel = STAGE_LABEL[pipelineStage]

  // Pick the most informative text to show
  const displayText =
    pipelineStage === 'speaking' && liveResponse  ? liveResponse :
    pipelineStage === 'thinking' && pendingStt     ? `"${pendingStt}"` :
    pipelineStage === 'recording'                  ? '' :
    pipelineStage === 'wake'                       ? 'Session starting…' :
    voiceListening && running                      ? "Say 'ok Dann' to begin" :
    ''

  async function handleTrigger() {
    setTriggering(true)
    try { await api.triggerVoice() } catch { /* silent */ } finally { setTriggering(false) }
  }

  async function handleToggleMic() {
    setToggling(true)
    try {
      if (voiceListening) await api.disableVoice()
      else await api.enableVoice()
    } catch { /* silent */ } finally { setToggling(false) }
  }

  return (
    <div className="voice-widget flex items-center gap-3 px-4 shrink-0" style={{ height: '44px' }}>

      {/* Stage orb */}
      <MiniOrb stage={running && voiceListening ? pipelineStage : 'idle'} />

      {/* Stage label */}
      <span className={`text-[9px] tracking-widest uppercase shrink-0 font-bold ${running && voiceListening ? stageLabel.color : 'text-zinc-700'}`}>
        {running ? (voiceListening ? stageLabel.text : 'Off') : 'Offline'}
      </span>

      {/* Separator */}
      <span className="text-zinc-800 shrink-0">|</span>

      {/* Live text preview */}
      <span className="flex-1 text-[10px] text-zinc-500 truncate min-w-0">
        {displayText}
        {pipelineStage === 'speaking' && liveResponse && (
          <span className="inline-block ml-0.5 animate-pulse text-teal-600">▋</span>
        )}
      </span>

      {/* Controls */}
      <div className="flex items-center gap-2 shrink-0 ml-auto">
        {/* Enable listening — shown when voice is off and Dann is running */}
        {!inSession && !voiceListening && running && (
          <button
            onClick={handleToggleMic}
            disabled={toggling}
            className="text-[9px] tracking-widest px-2.5 py-1 border border-zinc-600/60 text-zinc-400 hover:border-teal-500/60 hover:text-teal-300 hover:bg-teal-900/10 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {toggling ? '…' : 'Enable'}
          </button>
        )}

        {/* Talk button — only shown when idle and listening */}
        {!inSession && voiceListening && running && (
          <button
            onClick={handleTrigger}
            disabled={triggering}
            className="text-[9px] tracking-widest px-2.5 py-1 border border-teal-600/50 text-teal-500 hover:border-teal-400/70 hover:text-teal-300 hover:bg-teal-900/10 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {triggering ? '…' : 'Talk'}
          </button>
        )}

        {/* Mic toggle */}
        <button
          onClick={handleToggleMic}
          disabled={toggling || inSession || !running}
          title={!running ? 'Dann not running' : voiceListening ? 'Mute' : 'Unmute'}
          className={`flex items-center justify-center w-7 h-7 rounded-full border transition-all disabled:opacity-30 disabled:cursor-not-allowed ${
            voiceListening && running
              ? 'border-teal-600/60 text-teal-500 hover:border-teal-400 hover:text-teal-300 hover:bg-teal-900/10'
              : 'border-zinc-700 text-zinc-600 hover:border-zinc-500 hover:text-zinc-400'
          }`}
        >
          {voiceListening && running ? <MicOnIcon /> : <MicOffIcon />}
        </button>
      </div>
    </div>
  )
}
