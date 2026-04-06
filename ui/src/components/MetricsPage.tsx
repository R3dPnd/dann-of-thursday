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
import type { MetricSummary, ProjectMetric } from '../types'

export default function MetricsPage() {
  const { metricSummary } = useDannStore()
  const [allTime, setAllTime] = useState<MetricSummary | null>(null)
  const [byDay, setByDay] = useState<{ date: string; calls: number }[]>([])
  const [byProject, setByProject] = useState<ProjectMetric[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      api.getMetrics('all'),
      api.getCallsByDay(7),
      api.getByProject(),
    ]).then(([summary, days, projects]) => {
      setAllTime(summary)
      setByDay(days)
      setByProject(projects)
    }).catch(() => {}).finally(() => setLoading(false))
  }, [])

  const summary = allTime ?? metricSummary

  const successRate = summary && summary.total_calls > 0
    ? `${Math.round((summary.successful_calls / summary.total_calls) * 100)}%`
    : '—'

  const pieData = summary
    ? [
        { name: 'Success', value: summary.successful_calls, color: '#22c55e' },
        { name: 'Error', value: summary.error_calls, color: '#ef4444' },
        { name: 'Empty', value: summary.empty_calls, color: '#6b7280' },
      ].filter(d => d.value > 0)
    : []

  const tooltipStyle = {
    contentStyle: { backgroundColor: '#1f2937', border: 'none', borderRadius: 6 },
    labelStyle: { color: '#e5e7eb' },
    itemStyle: { color: '#60a5fa' },
  }

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
          { label: 'Total calls', value: summary?.total_calls ?? 0 },
          { label: 'Projects used', value: summary?.projects_used ?? 0 },
          { label: 'Success rate', value: successRate },
          { label: 'Avg response', value: summary?.avg_response_ms != null ? `${summary.avg_response_ms} ms` : '—' },
        ].map(({ label, value }) => (
          <div key={label} className="bg-gray-800 rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-white">{value}</div>
            <div className="text-xs text-gray-400 mt-1">{label}</div>
          </div>
        ))}
      </div>

      {/* Calls per day */}
      {byDay.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Claude calls per day (last 7 days)</h3>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={byDay} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#9ca3af' }} />
              <YAxis tick={{ fontSize: 11, fill: '#9ca3af' }} allowDecimals={false} />
              <Tooltip {...tooltipStyle} />
              <Bar dataKey="calls" fill="#3b82f6" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Calls per project */}
      {byProject.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Calls per project</h3>
          <ResponsiveContainer width="100%" height={Math.max(160, byProject.length * 28)}>
            <BarChart
              data={byProject}
              layout="vertical"
              margin={{ top: 0, right: 16, left: 8, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" tick={{ fontSize: 11, fill: '#9ca3af' }} allowDecimals={false} />
              <YAxis type="category" dataKey="project" tick={{ fontSize: 11, fill: '#9ca3af' }} width={140} />
              <Tooltip {...tooltipStyle} />
              <Bar dataKey="ok" name="Success" stackId="a" fill="#22c55e" />
              <Bar dataKey="error" name="Error" stackId="a" fill="#ef4444" />
              <Bar dataKey="empty" name="Empty" stackId="a" fill="#6b7280" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Status breakdown pie */}
      {pieData.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Response status breakdown</h3>
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
              <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: 6 }} />
              <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}

      {summary && summary.total_calls === 0 && (
        <div className="text-center text-gray-500 py-12">
          No Claude calls recorded yet.
        </div>
      )}
    </div>
  )
}
