import type { LogEntry, MetricSummary, Project, StateSnapshot } from '../types'

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

  getProjects: () =>
    get<{ projects: Project[]; count: number }>('/projects').then(r => r.projects),

  openProject: (name: string) =>
    post<{ result: string }>(`/projects/${encodeURIComponent(name)}/open`),

  getMetrics: (period: 'day' | 'week' | 'all' = 'all') =>
    get<MetricSummary>(`/metrics?period=${period}`),

  getCallsByDay: (days = 7) =>
    get<{ date: string; calls: number }[]>(`/metrics/calls-by-day?days=${days}`),

  getByProject: () =>
    get<{ project: string; total: number; ok: number; error: number; empty: number; avg_response_ms: number | null }[]>('/metrics/by-project'),

  createTerminal: (projectName: string, rows = 40, cols = 120) =>
    post<{ session_id: string; project_name: string; project_path: string; alive: boolean }>(
      '/terminals',
      { project_name: projectName, rows, cols },
    ),

  closeTerminal: (sessionId: string) =>
    fetch(`/api/v1/terminals/${sessionId}`, { method: 'DELETE' }).catch(() => {}),

  triggerVoice: () => post<{ triggered: boolean }>('/voice/trigger'),

  getLogs: (params?: { level?: string; module?: string; search?: string; limit?: number; offset?: number }) => {
    const qs = new URLSearchParams()
    if (params?.level) qs.set('level', params.level)
    if (params?.module) qs.set('module', params.module)
    if (params?.search) qs.set('search', params.search)
    if (params?.limit != null) qs.set('limit', String(params.limit))
    if (params?.offset != null) qs.set('offset', String(params.offset))
    return get<{ total: number; offset: number; limit: number; entries: LogEntry[] }>(`/logs?${qs}`)
  },
}
