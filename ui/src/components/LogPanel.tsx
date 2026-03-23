import { useCallback, useEffect, useRef, useState } from 'react'
import { useDannStore } from '../hooks/useDannState'
import type { LogEntry } from '../types'

const LEVEL_COLORS: Record<string, string> = {
  DEBUG: 'text-gray-400 bg-gray-700',
  INFO: 'text-blue-300 bg-blue-900',
  WARNING: 'text-yellow-300 bg-yellow-900',
  ERROR: 'text-red-300 bg-red-900',
  CRITICAL: 'text-pink-300 bg-pink-900',
}

const LEVEL_ORDER = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

function relTime(iso: string): string {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000
  if (diff < 60) return `${Math.floor(diff)}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  return `${Math.floor(diff / 3600)}h ago`
}

function LogRow({ entry }: { entry: LogEntry }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className="group border-b border-gray-800 px-3 py-1.5 text-xs cursor-pointer hover:bg-gray-800/50"
      onClick={() => entry.detail && setExpanded(e => !e)}
    >
      <div className="flex items-start gap-2">
        {/* Timestamp */}
        <span
          className="text-gray-500 whitespace-nowrap shrink-0"
          title={entry.timestamp}
        >
          {relTime(entry.timestamp)}
        </span>

        {/* Level badge */}
        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase shrink-0 ${LEVEL_COLORS[entry.level] ?? LEVEL_COLORS.DEBUG}`}>
          {entry.level}
        </span>

        {/* Module tag */}
        <span className="text-gray-500 shrink-0 font-mono">{entry.module}</span>

        {/* Message */}
        <span className="text-gray-200 break-all">{entry.message}</span>

        {/* Expand indicator */}
        {entry.detail && (
          <span className="ml-auto text-gray-600 shrink-0">{expanded ? '▲' : '▼'}</span>
        )}
      </div>

      {expanded && entry.detail && (
        <pre className="mt-1.5 text-[11px] text-red-300 bg-gray-900 rounded p-2 overflow-x-auto whitespace-pre-wrap break-all">
          {entry.detail}
        </pre>
      )}
    </div>
  )
}

export default function LogPanel() {
  const { logEntries, unreadErrors, clearUnreadErrors } = useDannStore()
  const [open, setOpen] = useState(false)
  const [minLevel, setMinLevel] = useState(2) // default: WARNING (index 2)
  const [moduleFilter, setModuleFilter] = useState('')
  const [search, setSearch] = useState('')
  const [errorsOnly, setErrorsOnly] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  const effectiveMinLevel = errorsOnly ? 3 : minLevel // ERROR when errors-only

  const filtered = logEntries.filter(e => {
    const rank = LEVEL_ORDER.indexOf(e.level)
    if (rank < effectiveMinLevel) return false
    if (moduleFilter && e.module !== moduleFilter) return false
    if (search && !e.message.toLowerCase().includes(search.toLowerCase())) return false
    return true
  })

  // Auto-scroll to bottom when panel is open
  useEffect(() => {
    if (open) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [filtered.length, open])

  const handleOpen = useCallback(() => {
    setOpen(o => {
      if (!o) clearUnreadErrors()
      return !o
    })
  }, [clearUnreadErrors])

  // Collect unique modules for filter checkboxes
  const modules = [...new Set(logEntries.map(e => e.module))].sort()

  const downloadLog = useCallback(() => {
    const text = filtered
      .map(e => `[${e.timestamp}] [${e.level}] [${e.module}] ${e.message}${e.detail ? '\n' + e.detail : ''}`)
      .join('\n')
    const blob = new Blob([text], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `dann-log-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }, [filtered])

  return (
    <div className="border-t border-gray-700 bg-gray-900 flex flex-col">
      {/* Header tab */}
      <button
        onClick={handleOpen}
        className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-gray-300 hover:text-white hover:bg-gray-800 transition-colors w-full text-left"
      >
        <span>Logs</span>
        {unreadErrors > 0 && (
          <span className="bg-red-600 text-white rounded-full px-1.5 py-0.5 text-[10px] font-bold">
            {unreadErrors}
          </span>
        )}
        <span className="ml-auto text-gray-500">{open ? '▼' : '▲'}</span>
      </button>

      {open && (
        <>
          {/* Filter bar */}
          <div className="flex flex-wrap items-center gap-2 px-3 py-2 border-b border-gray-800 bg-gray-900">
            {/* Level slider */}
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-gray-400">Level:</span>
              <input
                type="range"
                min={0}
                max={4}
                value={minLevel}
                onChange={e => setMinLevel(Number(e.target.value))}
                className="w-20 accent-blue-500"
                disabled={errorsOnly}
              />
              <span className="text-xs text-gray-300 w-16">{LEVEL_ORDER[minLevel]}</span>
            </div>

            {/* Module select */}
            {modules.length > 0 && (
              <select
                value={moduleFilter}
                onChange={e => setModuleFilter(e.target.value)}
                className="text-xs bg-gray-800 text-gray-200 border border-gray-700 rounded px-2 py-0.5"
              >
                <option value="">All modules</option>
                {modules.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            )}

            {/* Search */}
            <input
              type="text"
              placeholder="Search…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="text-xs bg-gray-800 text-gray-200 border border-gray-700 rounded px-2 py-0.5 w-40"
            />

            {/* Errors only toggle */}
            <label className="flex items-center gap-1 text-xs text-gray-300 cursor-pointer">
              <input
                type="checkbox"
                checked={errorsOnly}
                onChange={e => setErrorsOnly(e.target.checked)}
                className="accent-red-500"
              />
              Errors only
            </label>

            <span className="text-xs text-gray-500 ml-auto">{filtered.length} entries</span>

            {/* Download */}
            <button
              onClick={downloadLog}
              className="text-xs text-blue-400 hover:text-blue-300 px-2 py-0.5 border border-gray-700 rounded"
            >
              Download
            </button>
          </div>

          {/* Log entries */}
          <div className="overflow-y-auto" style={{ maxHeight: 280 }}>
            {filtered.length === 0 ? (
              <div className="text-center text-gray-500 py-6 text-xs">No log entries match the current filters.</div>
            ) : (
              filtered.map((entry, i) => <LogRow key={i} entry={entry} />)
            )}
            <div ref={bottomRef} />
          </div>
        </>
      )}
    </div>
  )
}
