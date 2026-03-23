export type Mode = 'idle' | 'normal' | 'code'

export interface Project {
  name: string
  path: string
}

export interface CodeTurn {
  id: string
  timestamp: Date
  task: string
  response: string
  status: 'ok' | 'error' | 'empty'
  latencyMs: number
  sessionId: string | null
}

export interface StateSnapshot {
  mode: string
  project: string | null
  session_id: string | null
  running: boolean
  uptime_s: number
}

export interface DannEvent {
  type: string
  payload: Record<string, unknown>
}

export interface MetricRecord {
  session_id?: string
  mode?: string
  blank?: boolean
  status?: string
  stt_ms?: number
  llm_ms?: number
  tts_ms?: number
  total_ms?: number
  recorded_at?: string
}

export interface MetricSummary {
  period: string
  total_turns: number
  blank_turns: number
  code_turns: number
  normal_turns: number
  error_turns: number
  avg_stt_ms: number | null
  avg_llm_ms: number | null
  avg_tts_ms: number | null
  avg_total_ms: number | null
  sessions: string[]
}

export interface LogEntry {
  timestamp: string
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL'
  module: string
  event_type: string
  message: string
  detail?: string
}
