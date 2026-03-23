import { create } from 'zustand'
import type { CodeTurn, LogEntry, MetricSummary, Mode, Project, StateSnapshot } from '../types'

const MAX_LOG_ENTRIES = 2000

interface DannStore {
  // Orchestrator state
  mode: Mode
  project: string | null
  sessionId: string | null
  running: boolean

  // WebSocket connection
  wsConnected: boolean

  // Projects
  projects: Project[]

  // Per-project CODE-mode turn history (project name → turns)
  codeHistory: Record<string, CodeTurn[]>

  // Metrics (session-scope counters updated from metric events)
  metricSummary: MetricSummary | null
  wakeCount: number

  // Live log entries streamed via WebSocket (capped at MAX_LOG_ENTRIES)
  logEntries: LogEntry[]
  unreadErrors: number

  // Actions
  applySnapshot: (snapshot: StateSnapshot) => void
  applyEvent: (type: string, payload: Record<string, unknown>) => void
  setWsConnected: (connected: boolean) => void
  setProjects: (projects: Project[]) => void
  setMetricSummary: (summary: MetricSummary) => void
  clearUnreadErrors: () => void
}

export const useDannStore = create<DannStore>((set) => ({
  mode: 'idle',
  project: null,
  sessionId: null,
  running: false,
  wsConnected: false,
  projects: [],
  codeHistory: {},
  metricSummary: null,
  wakeCount: 0,
  logEntries: [],
  unreadErrors: 0,

  applySnapshot: (snapshot) =>
    set({
      mode: (snapshot.mode as Mode) ?? 'idle',
      project: snapshot.project,
      sessionId: snapshot.session_id,
      running: snapshot.running,
    }),

  applyEvent: (type, payload) => {
    switch (type) {
      case 'state.changed':
        set({
          mode: (payload.mode as Mode) ?? 'idle',
          project: (payload.project as string | null) ?? null,
          sessionId: (payload.session_id as string | null) ?? null,
        })
        break

      case 'session.start':
        set((state) => ({
          sessionId: payload.session_id as string,
          wakeCount: state.wakeCount + 1,
        }))
        break

      case 'session.end':
        set({ sessionId: null, mode: 'idle' })
        break

      case 'turn.code': {
        const project = payload.project as string
        const turn: CodeTurn = {
          id: `${Date.now()}-${Math.random()}`,
          timestamp: new Date(),
          task: (payload.task as string) ?? '',
          response: (payload.response as string) ?? '',
          status: (payload.status as CodeTurn['status']) ?? 'ok',
          latencyMs: (payload.latency_ms as number) ?? 0,
          sessionId: (payload.session_id as string | null) ?? null,
        }
        set((state) => ({
          codeHistory: {
            ...state.codeHistory,
            [project]: [...(state.codeHistory[project] ?? []), turn],
          },
        }))
        break
      }

      case 'metric': {
        // Update running session metric summary from the latest metric event.
        set((state) => {
          const prev = state.metricSummary
          const mode = payload.mode as string | undefined
          const blank = !!(payload.blank)
          const status = payload.status as string | undefined

          const total = (prev?.total_turns ?? 0) + 1
          const blankTurns = (prev?.blank_turns ?? 0) + (blank ? 1 : 0)
          const codeTurns = (prev?.code_turns ?? 0) + (mode === 'CODE' ? 1 : 0)
          const normalTurns = (prev?.normal_turns ?? 0) + (mode === 'NORMAL' ? 1 : 0)
          const errorTurns = (prev?.error_turns ?? 0) + (status === 'error' ? 1 : 0)

          const avg = (prev: number | null, next: unknown, count: number): number | null => {
            if (typeof next !== 'number') return prev
            if (prev === null) return next
            return Math.round((prev * (count - 1) + next) / count)
          }

          return {
            metricSummary: {
              period: 'session',
              total_turns: total,
              blank_turns: blankTurns,
              code_turns: codeTurns,
              normal_turns: normalTurns,
              error_turns: errorTurns,
              avg_stt_ms: avg(prev?.avg_stt_ms ?? null, payload.stt_ms, total),
              avg_llm_ms: avg(prev?.avg_llm_ms ?? null, payload.llm_ms, total),
              avg_tts_ms: avg(prev?.avg_tts_ms ?? null, payload.tts_ms, total),
              avg_total_ms: avg(prev?.avg_total_ms ?? null, payload.total_ms, total),
              sessions: prev?.sessions ?? [],
            },
          }
        })
        break
      }

      case 'error':
      case 'warning': {
        const entry: LogEntry = {
          timestamp: new Date().toISOString(),
          level: type === 'error' ? 'ERROR' : 'WARNING',
          module: (payload.module as string) ?? type,
          event_type: type,
          message: (payload.message as string) ?? '',
          detail: (payload.traceback as string | undefined) ?? (payload.detail as string | undefined),
        }
        set((state) => {
          const entries = [...state.logEntries, entry]
          if (entries.length > MAX_LOG_ENTRIES) entries.splice(0, entries.length - MAX_LOG_ENTRIES)
          return {
            logEntries: entries,
            unreadErrors: type === 'error' ? state.unreadErrors + 1 : state.unreadErrors,
          }
        })
        break
      }

      default:
        // Capture all INFO-level events (session.start/end, state.changed) as log entries.
        if (['session.start', 'session.end', 'state.changed'].includes(type)) {
          const entry: LogEntry = {
            timestamp: new Date().toISOString(),
            level: 'INFO',
            module: 'orchestrator',
            event_type: type,
            message: type,
          }
          set((state) => {
            const entries = [...state.logEntries, entry]
            if (entries.length > MAX_LOG_ENTRIES) entries.splice(0, entries.length - MAX_LOG_ENTRIES)
            return { logEntries: entries }
          })
        }
        break
    }
  },

  setWsConnected: (connected) => set({ wsConnected: connected }),

  setProjects: (projects) => set({ projects }),

  setMetricSummary: (summary) => set({ metricSummary: summary }),

  clearUnreadErrors: () => set({ unreadErrors: 0 }),
}))
