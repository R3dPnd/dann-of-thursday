import type { DevTeamJob, FocusArea, FocusAreaNote, LogEntry, MetricSummary, Module, Project, PromptBuilderResult, StateSnapshot, TerminalSession, WorkStream } from '../types'

const BASE = '/api/v1'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`)
  return res.json() as Promise<T>
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`)
  return res.json() as Promise<T>
}

export const api = {
  getState: () => get<StateSnapshot>('/state'),

  getNotes: () =>
    get<{ notes: Project[]; count: number }>('/notes').then(r => r.notes),

  getModules: () =>
    get<{ modules: Module[] }>('/modules').then(r => r.modules),

  enableModule: (name: string) =>
    post<{ module: string; enabled: boolean }>(`/modules/${encodeURIComponent(name)}/enable`),

  disableModule: (name: string) =>
    post<{ module: string; enabled: boolean }>(`/modules/${encodeURIComponent(name)}/disable`),

  getFocusAreas: () =>
    get<{ focus_areas: FocusArea[] }>('/focus-areas').then(r => r.focus_areas),

  getFocusAreaNotes: (name: string) =>
    get<{ notes: FocusAreaNote[] }>(`/focus-areas/${encodeURIComponent(name)}/notes`).then(r => r.notes),

  createFocusAreaNote: (name: string, content: string, title?: string) =>
    post<FocusAreaNote>(`/focus-areas/${encodeURIComponent(name)}/notes`, { content, ...(title ? { title } : {}) }),

  deleteFocusAreaNote: (name: string, filename: string) =>
    fetch(`/api/v1/focus-areas/${encodeURIComponent(name)}/notes/${encodeURIComponent(filename)}`, { method: 'DELETE' }),

  openProject: (name: string) =>
    post<{ result: string }>(`/projects/${encodeURIComponent(name)}/open`),

  getMetrics: (period: 'day' | 'week' | 'all' = 'all') =>
    get<MetricSummary>(`/metrics?period=${period}`),

  getCallsByDay: (days = 7) =>
    get<{ date: string; calls: number }[]>(`/metrics/calls-by-day?days=${days}`),

  getByProject: () =>
    get<{ project: string; total: number; ok: number; error: number; empty: number; avg_response_ms: number | null }[]>('/metrics/by-project'),

  listTerminals: () => get<TerminalSession[]>('/terminals'),

  createTerminal: (focusArea: string, rows = 40, cols = 120, command?: string) =>
    post<TerminalSession>(
      '/terminals',
      { focus_area: focusArea, rows, cols, ...(command ? { command } : {}) },
    ),

  closeTerminal: (sessionId: string) =>
    fetch(`/api/v1/terminals/${sessionId}`, { method: 'DELETE' }).catch(() => {}),

  getHistory: (limit = 100, offset = 0) =>
    get<{ total: number; offset: number; limit: number; records: Record<string, unknown>[] }>(
      `/history?limit=${limit}&offset=${offset}`
    ).then(r => r.records),

  triggerVoice: () => post<{ triggered: boolean }>('/voice/trigger'),
  enableVoice: () => post<{ listening: boolean }>('/voice/enable'),
  disableVoice: () => post<{ listening: boolean }>('/voice/disable'),

  buildPrompt: (body: { thoughts: string; goals: string; notes: string; custom_sections: { title: string; content: string }[] }) =>
    post<PromptBuilderResult>('/prompt-builder', body),

  listStreams: () => get<{ streams: WorkStream[] }>('/chat/streams').then(r => r.streams),

  createStream: (focusArea: string, title?: string) =>
    post<WorkStream>('/chat/streams', { focus_area: focusArea, ...(title ? { title } : {}) }),

  getStream: (id: string) => get<WorkStream>(`/chat/streams/${encodeURIComponent(id)}`),

  sendChatMessage: (id: string, text: string) =>
    post<{ response: string; latency_ms: number }>(`/chat/streams/${encodeURIComponent(id)}/messages`, { text }),

  clearStream: (id: string) =>
    post<WorkStream>(`/chat/streams/${encodeURIComponent(id)}/clear`),

  deleteStream: (id: string) =>
    fetch(`/api/v1/chat/streams/${encodeURIComponent(id)}`, { method: 'DELETE' }),

  getLogs: (params?: { level?: string; module?: string; search?: string; limit?: number; offset?: number }) => {
    const qs = new URLSearchParams()
    if (params?.level) qs.set('level', params.level)
    if (params?.module) qs.set('module', params.module)
    if (params?.search) qs.set('search', params.search)
    if (params?.limit != null) qs.set('limit', String(params.limit))
    if (params?.offset != null) qs.set('offset', String(params.offset))
    return get<{ total: number; offset: number; limit: number; entries: LogEntry[] }>(`/logs?${qs}`)
  },

  getDevTeamJobs: (params?: { project?: string; limit?: number; offset?: number }) => {
    const qs = new URLSearchParams()
    if (params?.project) qs.set('project', params.project)
    if (params?.limit != null) qs.set('limit', String(params.limit))
    if (params?.offset != null) qs.set('offset', String(params.offset))
    return get<{ total: number; offset: number; limit: number; jobs: DevTeamJob[] }>(`/devteam?${qs}`)
  },

  restartDann: () => post<{ status: string }>('/system/restart'),
}
