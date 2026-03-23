import { useEffect } from 'react'
import { api } from '../lib/api'
import { useDannStore } from '../hooks/useDannState'

function Stat({ label, value }: { label: string; value: string | number | null }) {
  return (
    <div className="flex flex-col items-center px-4 border-r border-gray-700 last:border-r-0">
      <span className="text-xs text-gray-400 whitespace-nowrap">{label}</span>
      <span className="text-sm font-semibold text-white mt-0.5">
        {value ?? '—'}
      </span>
    </div>
  )
}

export default function MetricsBar() {
  const { metricSummary, wakeCount, setMetricSummary } = useDannStore()

  // On mount, seed from REST (all-time stats). WebSocket events keep it live after that.
  useEffect(() => {
    api.getMetrics('all').then(setMetricSummary).catch(() => {/* non-fatal */})
  }, [setMetricSummary])

  const avgMs = (v: number | null | undefined) =>
    v != null ? `${v} ms` : '—'

  return (
    <div className="flex items-center bg-gray-900 border-b border-gray-700 px-2 py-1.5 overflow-x-auto">
      <span className="text-xs font-medium text-gray-500 mr-3 whitespace-nowrap">Metrics</span>
      <Stat label="Wake events" value={wakeCount} />
      <Stat label="Turns today" value={metricSummary?.total_turns ?? '—'} />
      <Stat label="Code turns" value={metricSummary?.code_turns ?? '—'} />
      <Stat label="Blank turns" value={metricSummary?.blank_turns ?? '—'} />
      <Stat label="Errors" value={metricSummary?.error_turns ?? '—'} />
      <Stat label="Avg STT" value={avgMs(metricSummary?.avg_stt_ms)} />
      <Stat label="Avg LLM" value={avgMs(metricSummary?.avg_llm_ms)} />
      <Stat label="Avg TTS" value={avgMs(metricSummary?.avg_tts_ms)} />
      <Stat label="Avg turn" value={avgMs(metricSummary?.avg_total_ms)} />
    </div>
  )
}
