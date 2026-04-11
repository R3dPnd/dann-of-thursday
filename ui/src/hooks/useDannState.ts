import { create } from 'zustand'
import type { CodeTurn, LogEntry, MetricSummary, Mode, PipelineStage, Project, StateSnapshot, VoiceTurn } from '../types'

const MAX_LOG_ENTRIES = 2000
const MAX_VOICE_TURNS = 200

interface DannStore {
  // Orchestrator state
  mode: Mode
  project: string | null
  sessionId: string | null
  running: boolean
  voiceListening: boolean

  // WebSocket connection
  wsConnected: boolean

  // Voice pipeline stage
  pipelineStage: PipelineStage
  // Pending STT text waiting to be paired with an LLM response
  _pendingStt: string | null

  // Projects
  projects: Project[]
  noteProjects: Project[]

  // Voice conversation history (all modes)
  voiceTurns: VoiceTurn[]

  // Per-project CODE-mode turn history (project name → turns)
  codeHistory: Record<string, CodeTurn[]>

  // Metrics
  metricSummary: MetricSummary | null

  // Live log entries streamed via WebSocket (capped at MAX_LOG_ENTRIES)
  logEntries: LogEntry[]
  unreadErrors: number

  // Actions
  applySnapshot: (snapshot: StateSnapshot) => void
  applyEvent: (type: string, payload: Record<string, unknown>) => void
  setWsConnected: (connected: boolean) => void
  setProjects: (projects: Project[]) => void
  setNoteProjects: (notes: Project[]) => void
  setMetricSummary: (summary: MetricSummary) => void
  clearUnreadErrors: () => void
}

export const useDannStore = create<DannStore>((set) => ({
  mode: 'idle',
  project: null,
  sessionId: null,
  running: false,
  voiceListening: true,
  wsConnected: false,
  pipelineStage: 'idle',
  _pendingStt: null,
  projects: [],
  noteProjects: [],
  voiceTurns: [],
  codeHistory: {},
  metricSummary: null,
  logEntries: [],
  unreadErrors: 0,

  applySnapshot: (snapshot) =>
    set({
      mode: (snapshot.mode as Mode) ?? 'idle',
      project: snapshot.project,
      sessionId: snapshot.session_id,
      running: snapshot.running,
      voiceListening: snapshot.listening ?? true,
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

      case 'voice.listening_changed':
        set({ voiceListening: (payload.listening as boolean) ?? true })
        break

      case 'session.start':
        set({ sessionId: payload.session_id as string, pipelineStage: 'wake' })
        break

      case 'session.end':
        set({ sessionId: null, mode: 'idle', pipelineStage: 'idle', _pendingStt: null })
        break

      case 'turn.start':
        set({ pipelineStage: 'recording' })
        break

      case 'turn.stt': {
        const text = (payload.text as string) ?? ''
        if (!text) {
          set({ pipelineStage: 'idle', _pendingStt: null })
        } else {
          set({ pipelineStage: 'thinking', _pendingStt: text })
        }
        break
      }

      case 'turn.llm': {
        const dannText = (payload.text as string) ?? ''
        set((state) => {
          const turn: VoiceTurn = {
            id: `${Date.now()}-${Math.random()}`,
            sessionId: (payload.session_id as string | null) ?? state.sessionId,
            timestamp: new Date(),
            userText: state._pendingStt ?? '',
            dannText,
            mode: 'normal',
          }
          const turns = [...state.voiceTurns, turn]
          if (turns.length > MAX_VOICE_TURNS) turns.splice(0, turns.length - MAX_VOICE_TURNS)
          return { voiceTurns: turns, pipelineStage: 'speaking', _pendingStt: null }
        })
        // After speaking, pipeline returns to idle (approximate — no turn.tts event)
        setTimeout(() => set((s) => s.pipelineStage === 'speaking' ? { pipelineStage: 'idle' } : {}), 3000)
        break
      }

      case 'turn.code': {
        const project = payload.project as string
        const codeTurn: CodeTurn = {
          id: `${Date.now()}-${Math.random()}`,
          timestamp: new Date(),
          task: (payload.task as string) ?? '',
          response: (payload.response as string) ?? '',
          status: (payload.status as CodeTurn['status']) ?? 'ok',
          latencyMs: (payload.latency_ms as number) ?? 0,
          sessionId: (payload.session_id as string | null) ?? null,
        }
        set((state) => {
          const voiceTurn: VoiceTurn = {
            id: codeTurn.id,
            sessionId: codeTurn.sessionId,
            timestamp: codeTurn.timestamp,
            userText: state._pendingStt ?? codeTurn.task,
            dannText: codeTurn.response,
            mode: 'code',
            project,
          }
          const prev = state.metricSummary
          const status = codeTurn.status
          const total = (prev?.total_calls ?? 0) + 1
          const avg = (p: number | null, next: number, n: number) =>
            p === null ? next : Math.round((p * (n - 1) + next) / n)
          const voiceTurns = [...state.voiceTurns, voiceTurn]
          if (voiceTurns.length > MAX_VOICE_TURNS) voiceTurns.splice(0, voiceTurns.length - MAX_VOICE_TURNS)
          return {
            codeHistory: {
              ...state.codeHistory,
              [project]: [...(state.codeHistory[project] ?? []), codeTurn],
            },
            voiceTurns,
            pipelineStage: 'speaking' as PipelineStage,
            _pendingStt: null,
            metricSummary: {
              period: 'session',
              total_calls: total,
              successful_calls: (prev?.successful_calls ?? 0) + (status === 'ok' ? 1 : 0),
              error_calls: (prev?.error_calls ?? 0) + (status === 'error' ? 1 : 0),
              empty_calls: (prev?.empty_calls ?? 0) + (status === 'empty' ? 1 : 0),
              avg_response_ms: avg(prev?.avg_response_ms ?? null, codeTurn.latencyMs, total),
              projects_used: new Set([...Object.keys(state.codeHistory), project]).size,
              sessions: prev?.sessions ?? [],
            },
          }
        })
        setTimeout(() => set((s) => s.pipelineStage === 'speaking' ? { pipelineStage: 'idle' } : {}), 3000)
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
  setNoteProjects: (notes) => set({ noteProjects: notes }),
  setMetricSummary: (summary) => set({ metricSummary: summary }),
  clearUnreadErrors: () => set({ unreadErrors: 0 }),
}))
