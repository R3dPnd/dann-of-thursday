import { Suspense, lazy, useEffect, useState } from 'react'
import { useDannStore } from '../hooks/useDannState'
import type { TerminalSession } from '../types'

const TerminalPane = lazy(() => import('./TerminalPane'))

function TerminalIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2}>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M7 9l3 3-3 3M13 15h4" />
    </svg>
  )
}

function label(t: TerminalSession): string {
  return t.project ? `${t.focus_area} · ${t.project}` : t.focus_area ?? t.session_id.slice(0, 8)
}

export function TerminalIndicator() {
  const terminals = useDannStore((s) => s.terminals)
  const dannTerminals = terminals.filter((t) => t.origin === 'dann' && t.alive)

  const [open, setOpen] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // Default to the most recently listed one whenever the set changes and
  // nothing (or something now-closed) is selected.
  useEffect(() => {
    if (dannTerminals.length === 0) { setSelectedId(null); return }
    if (!selectedId || !dannTerminals.some((t) => t.session_id === selectedId)) {
      setSelectedId(dannTerminals[dannTerminals.length - 1].session_id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dannTerminals.map((t) => t.session_id).join(',')])

  if (dannTerminals.length === 0) return null

  const selected = dannTerminals.find((t) => t.session_id === selectedId) ?? dannTerminals[0]

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        title="Dann is using Claude Code — click to watch"
        className="fixed bottom-4 right-4 z-40 flex items-center gap-1.5 rounded-full border border-emerald-700/60 bg-zinc-900 px-3 py-2 text-emerald-400 shadow-lg hover:bg-zinc-800"
      >
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
        </span>
        <TerminalIcon />
        {dannTerminals.length > 1 && (
          <span className="text-xs font-medium">{dannTerminals.length}</span>
        )}
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={() => setOpen(false)}>
          <div
            className="flex h-full max-h-[80vh] w-full max-w-4xl flex-col overflow-hidden rounded-lg border border-zinc-800 bg-zinc-950 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
              <div className="flex items-center gap-2 overflow-x-auto">
                {dannTerminals.map((t) => (
                  <button
                    key={t.session_id}
                    onClick={() => setSelectedId(t.session_id)}
                    className={`shrink-0 rounded px-2 py-1 text-xs ${
                      t.session_id === selected.session_id
                        ? 'bg-emerald-900/40 text-emerald-300'
                        : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
                    }`}
                  >
                    {label(t)}
                  </button>
                ))}
              </div>
              <button
                onClick={() => setOpen(false)}
                className="shrink-0 text-zinc-500 hover:text-zinc-300"
                title="Close"
              >
                ×
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <Suspense fallback={<div className="p-4 text-xs text-zinc-500">Loading terminal…</div>}>
                <TerminalPane key={selected.session_id} sessionId={selected.session_id} isActive />
              </Suspense>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
