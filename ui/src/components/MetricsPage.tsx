import { useEffect, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts'
import { api } from '../lib/api'
import { useDannStore } from '../hooks/useDannState'
import type { MetricSummary } from '../types'

interface SessionDay {
  date: string
  sessions: number
}

interface SessionLatency {
  session_id: string
  turns: number
  avg_stt_ms: number | null
  avg_llm_ms: number | null
  avg_tts_ms: number | null
  avg_total_ms: number | null
}

export default function MetricsPage() {
  const { metricSummary } = useDannStore()
  const [allTime, setAllTime] = useState<MetricSummary | null>(null)
  const [byDay, setByDay] = useState<SessionDay[]>([])
  const [bySession, setBySession] = useState<SessionLatency[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      api.getMetrics('all'),
      api.getSessionsByDay(7),
      api.getLatencyBySession(),
    ]).then(([summary, days, sessions]) => {
      setAllTime(summary)
      setByDay(days)
      setBySession(sessions.slice(-20)) // last 20 sessions
    }).catch(() => {/* non-fatal */}).finally(() => setLoading(false))
  }, [])

  const summary = allTime ?? metricSummary

  const pieData = summary
    ? [
        { name: 'Normal', value: summary.normal_turns, color: '#22c55e' },
        { name: 'Code', value: summary.code_turns, color: '#3b82f6' },
        { name: 'Blank', value: summary.blank_turns, color: '#6b7280' },
        { name: 'Error', value: summary.error_turns, color: '#ef4444' },
      ].filter(d => d.value > 0)
    : []

  const latencyData = bySession.map(s => ({
    name: s.session_id.slice(0, 8),
    STT: s.avg_stt_ms ?? 0,
    LLM: s.avg_llm_ms ?? 0,
    TTS: s.avg_tts_ms ?? 0,
  }))

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        Loading metrics…
      </div>
    )
  }

  return (
    <div className="p-4 space-y-6 overflow-y-auto">
      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: 'Total turns', value: summary?.total_turns ?? 0 },
          { label: 'Sessions', value: summary?.sessions?.length ?? 0 },
          { label: 'Code turns', value: summary?.code_turns ?? 0 },
          { label: 'Error turns', value: summary?.error_turns ?? 0 },
        ].map(({ label, value }) => (
          <div key={label} className="bg-gray-800 rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-white">{value}</div>
            <div className="text-xs text-gray-400 mt-1">{label}</div>
          </div>
        ))}
      </div>

      {/* Avg latencies */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: 'Avg STT', value: summary?.avg_stt_ms },
          { label: 'Avg LLM', value: summary?.avg_llm_ms },
          { label: 'Avg TTS', value: summary?.avg_tts_ms },
          { label: 'Avg turn', value: summary?.avg_total_ms },
        ].map(({ label, value }) => (
          <div key={label} className="bg-gray-800 rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-white">
              {value != null ? `${value} ms` : '—'}
            </div>
            <div className="text-xs text-gray-400 mt-1">{label}</div>
          </div>
        ))}
      </div>

      {/* Sessions per day */}
      {byDay.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Sessions per day (last 7 days)</h3>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={byDay} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#9ca3af' }} />
              <YAxis tick={{ fontSize: 11, fill: '#9ca3af' }} allowDecimals={false} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: 6 }}
                labelStyle={{ color: '#e5e7eb' }}
                itemStyle={{ color: '#60a5fa' }}
              />
              <Bar dataKey="sessions" fill="#3b82f6" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Latency breakdown per session */}
      {latencyData.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Avg latency per session (ms)</h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={latencyData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#9ca3af' }} />
              <YAxis tick={{ fontSize: 11, fill: '#9ca3af' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: 6 }}
                labelStyle={{ color: '#e5e7eb' }}
              />
              <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
              <Bar dataKey="STT" stackId="a" fill="#f59e0b" />
              <Bar dataKey="LLM" stackId="a" fill="#3b82f6" />
              <Bar dataKey="TTS" stackId="a" fill="#8b5cf6" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Mode distribution pie */}
      {pieData.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Turn mode distribution</h3>
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={70}
                paddingAngle={3}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                labelLine={false}
              >
                {pieData.map((entry) => (
                  <Cell key={entry.name} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: 6 }}
                itemStyle={{ color: '#e5e7eb' }}
              />
              <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}

      {summary && summary.total_turns === 0 && (
        <div className="text-center text-gray-500 py-12">
          No turns recorded yet. Start a session to see metrics.
        </div>
      )}
    </div>
  )
}
