import { create } from 'zustand'
import type { CodeTurn, LogEntry, MetricSummary, Mode, PipelineStage, Project, RawEvent, StateSnapshot, VoiceTurn } from '../types'

const MAX_LOG_ENTRIES = 2000
const MAX_VOICE_TURNS = 200
const MAX_RAW_EVENTS  = 200

function summarisePayload(type: string, payload: Record<string, unknown>): string {
  if (type === 'turn.stt')       return `"${String(payload.text ?? '').slice(0, 60)}"`
  if (type === 'turn.llm')       return `${payload.latency_ms}ms — "${String(payload.text ?? '').slice(0, 40)}"`
  if (type === 'turn.llm.chunk') return String(payload.chunk ?? '').slice(0, 40)
  if (type === 'turn.code')      return `[${payload.status}] ${String(payload.task ?? '').slice(0, 40)}`
  if (type === 'metric')         return `stt:${payload.stt_ms} llm:${payload.llm_ms} tts:${payload.tts_ms} → ${payload.status}`
  if (type === 'error')          return String(payload.message ?? '')
  if (type === 'state.changed')  return `mode=${payload.mode} running=${payload.running}`
  if (type === 'session.start')  return String(payload.session_id ?? '').slice(0, 12)
  if (type === 'session.end')    return String(payload.reason ?? '')
  return ''
}

interface DannStore {
  // Orchestrator state
  mode: Mode
  project: string | null
  sessionId: string | null
  running: boolean
  voiceListening: boolean

  // WebSocket connection
  wsConnected: boolean

  // Voice pipeline stage + when it started (for live duration display)
  pipelineStage: PipelineStage
  stageEnteredAt: number
  _pendingStt: string | null
  liveResponse: string

  // Projects
  projects: Project[]
  noteProjects: Project[]

  // Voice conversation history
  voiceTurns: VoiceTurn[]

  // Per-project CODE-mode turn history
  codeHistory: Record<string, CodeTurn[]>

  // Metrics
  metricSummary: MetricSummary | null

  // Log entries (capped)
  logEntries: LogEntry[]
  unreadErrors: number

  // Raw WebSocket events for the debug panel (capped)
  rawEvents: RawEvent[]

  // Actions
  applySnapshot: (snapshot: StateSnapshot) => void
  applyEvent: (type: string, payload: Record<string, unknown>) => void
  setWsConnected: (connected: boolean) => void
  setProjects: (projects: Project[]) => void
  setNoteProjects: (notes: Project[]) => void
  setMetricSummary: (summary: MetricSummary) => void
  clearUnreadErrors: () => void
  prependTurns: (turns: VoiceTurn[]) => void
}

function setStage(stage: PipelineStage): Partial<DannStore> {
  return { pipelineStage: stage, stageEnteredAt: Date.now() }
}

export const useDannStore = create<DannStore>((set) => ({
  mode: 'idle',
  project: null,
  sessionId: null,
  running: false,
  voiceListening: true,
  wsConnected: false,
  pipelineStage: 'idle',
  stageEnteredAt: Date.now(),
  _pendingStt: null,
  liveResponse: '',
  projects: [],
  noteProjects: [],
  voiceTurns: [],
  codeHistory: {},
  metricSummary: null,
  logEntries: [],
  unreadErrors: 0,
  rawEvents: [],

  applySnapshot: (snapshot) =>
    set({
      mode: (snapshot.mode as Mode) ?? 'idle',
      project: snapshot.project,
      sessionId: snapshot.session_id,
      running: snapshot.running,
      voiceListening: snapshot.listening ?? true,
    }),

  applyEvent: (type, payload) => {
    // Push every event to the raw debug log
    set((s) => {
      const entry: RawEvent = { ts: Date.now(), type, summary: summarisePayload(type, payload) }
      const rawEvents = [...s.rawEvents, entry]
      if (rawEvents.length > MAX_RAW_EVENTS) rawEvents.splice(0, rawEvents.length - MAX_RAW_EVENTS)
      return { rawEvents }
    })

    switch (type) {
      case 'state.changed':
        set((s) => ({
          mode: (payload.mode as Mode) ?? 'idle',
          project: (payload.project as string | null) ?? null,
          sessionId: (payload.session_id as string | null) ?? null,
          running: payload.running !== undefined ? (payload.running as boolean) : s.running,
        }))
        break

      case 'voice.listening_changed':
        set({ voiceListening: (payload.listening as boolean) ?? true })
        break

      case 'session.start':
        set({ sessionId: payload.session_id as string, ...setStage('wake') })
        break

      case 'session.end':
        set({ sessionId: null, mode: 'idle', ...setStage('idle'), _pendingStt: null, liveResponse: '' })
        break

      case 'turn.start':
        set({ ...setStage('recording'), liveResponse: '' })
        break

      case 'turn.llm.chunk':
        set((s) => ({
          ...setStage('speaking'),
          liveResponse: s.liveResponse
            ? s.liveResponse + ' ' + ((payload.chunk as string) ?? '')
            : (payload.chunk as string) ?? '',
        }))
        break

      case 'turn.stt': {
        const text = (payload.text as string) ?? ''
        if (!text) {
          set({ ...setStage('idle'), _pendingStt: null })
        } else {
          set({ ...setStage('thinking'), _pendingStt: text })
        }
        break
      }

      case 'turn.llm': {
        const dannText = (payload.text as string) ?? ''
        set((state) => {
          const wasStreaming = !!state.liveResponse
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
          return {
            voiceTurns: turns,
            ...(wasStreaming ? setStage('idle') : setStage('speaking')),
            _pendingStt: null,
            liveResponse: '',
          }
        })
        break
      }

      case 'turn.tts.done':
        set((s) => s.pipelineStage === 'speaking' ? setStage('idle') : {})
        break

      case 'metric': {
        // Attach trace data to the most recent completed turn
        const trace = {
          stt_ms:  (payload.stt_ms  as number | null) ?? null,
          llm_ms:  (payload.llm_ms  as number | null) ?? null,
          tts_ms:  (payload.tts_ms  as number | null) ?? null,
          code_ms: (payload.code_ms as number | null) ?? null,
          status:  (payload.status  as string)        ?? 'ok',
        }
        set((state) => {
          if (state.voiceTurns.length === 0) return {}
          const turns = [...state.voiceTurns]
          turns[turns.length - 1] = { ...turns[turns.length - 1], trace }
          return { voiceTurns: turns }
        })
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
            ...setStage('speaking'),
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
  setProjects:    (projects)  => set({ projects }),
  setNoteProjects:(notes)     => set({ noteProjects: notes }),
  setMetricSummary:(summary)  => set({ metricSummary: summary }),
  clearUnreadErrors: ()       => set({ unreadErrors: 0 }),
  prependTurns: (turns)       => set((s) => ({ voiceTurns: [...turns, ...s.voiceTurns] })),
}))
