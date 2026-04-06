export type Mode = 'idle' | 'normal' | 'code'
export type PipelineStage = 'idle' | 'wake' | 'recording' | 'thinking' | 'speaking'

export interface VoiceTurn {
  id: string
  sessionId: string | null
  timestamp: Date
  userText: string
  dannText: string
  mode: 'normal' | 'code'
  project?: string
}

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

export interface MetricSummary {
  period: string
  total_calls: number
  successful_calls: number
  error_calls: number
  empty_calls: number
  avg_response_ms: number | null
  projects_used: number
  sessions: string[]
}

export interface ProjectMetric {
  project: string
  total: number
  ok: number
  error: number
  empty: number
  avg_response_ms: number | null
}

export interface LogEntry {
  timestamp: string
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL'
  module: string
  event_type: string
  message: string
  detail?: string
}
