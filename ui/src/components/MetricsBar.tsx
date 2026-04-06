import { useEffect } from 'react'
import { api } from '../lib/api'
import { useDannStore } from '../hooks/useDannState'

function Stat({ label, value }: { label: string; value: string | number | null }) {
  return (
    <div className="flex flex-col items-center px-4 border-r border-gray-700 last:border-r-0">
      <span className="text-xs text-gray-400 whitespace-nowrap">{label}</span>
      <span className="stat-value text-sm font-semibold mt-0.5">
        {value ?? '—'}
      </span>
    </div>
  )
}

export default function MetricsBar() {
  const { metricSummary, setMetricSummary } = useDannStore()

  useEffect(() => {
    api.getMetrics('all').then(setMetricSummary).catch(() => {})
  }, [setMetricSummary])

  const m = metricSummary
  const successRate = m && m.total_calls > 0
    ? `${Math.round((m.successful_calls / m.total_calls) * 100)}%`
    : '—'

  return (
    <div className="metrics-bar flex items-center bg-gray-900 px-2 py-1.5 overflow-x-auto">
      <span className="metrics-label text-xs font-medium mr-3 whitespace-nowrap">Claude</span>
      <Stat label="Total calls" value={m?.total_calls ?? '—'} />
      <Stat label="Projects" value={m?.projects_used ?? '—'} />
      <Stat label="Success" value={successRate} />
      <Stat label="Errors" value={m?.error_calls ?? '—'} />
      <Stat label="Avg response" value={m?.avg_response_ms != null ? `${m.avg_response_ms} ms` : '—'} />
    </div>
  )
}
