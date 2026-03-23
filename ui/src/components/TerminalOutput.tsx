import { useEffect, useRef } from 'react'
import type { CodeTurn } from '../types'

interface Props {
  turns: CodeTurn[]
}

function StatusBadge({ status }: { status: CodeTurn['status'] }) {
  const styles = {
    ok:    'bg-green-900 text-green-300',
    error: 'bg-red-900 text-red-300',
    empty: 'bg-zinc-700 text-zinc-400',
  }
  return (
    <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${styles[status]}`}>
      {status}
    </span>
  )
}

function relativeTime(date: Date): string {
  const secs = Math.floor((Date.now() - date.getTime()) / 1000)
  if (secs < 60) return `${secs}s ago`
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`
  return `${Math.floor(secs / 3600)}h ago`
}

function CopyButton({ text }: { text: string }) {
  const copy = () => navigator.clipboard.writeText(text).catch(() => {})
  return (
    <button
      onClick={copy}
      className="rounded px-2 py-0.5 text-[10px] text-zinc-500 hover:bg-zinc-700 hover:text-zinc-200"
    >
      copy
    </button>
  )
}

export function TerminalOutput({ turns }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns.length])

  if (turns.length === 0) {
    return (
      <p className="px-3 py-4 text-xs text-zinc-600">
        No turns yet — enter code mode for this project to see output here.
      </p>
    )
  }

  return (
    <div className="max-h-96 overflow-y-auto">
      {turns.map((turn) => (
        <div key={turn.id} className="border-t border-zinc-800 px-3 py-3 text-xs">
          {/* Header row */}
          <div className="mb-1.5 flex items-center gap-2 text-zinc-500">
            <span title={turn.timestamp.toISOString()}>{relativeTime(turn.timestamp)}</span>
            <StatusBadge status={turn.status} />
            <span>{turn.latencyMs}ms</span>
          </div>

          {/* Task (user voice input) */}
          <p className="mb-1 font-mono text-zinc-400">
            <span className="text-zinc-600">› </span>
            {turn.task}
          </p>

          {/* Response */}
          {turn.response ? (
            <div className="group relative rounded bg-zinc-900 p-2">
              <pre className="whitespace-pre-wrap font-mono text-zinc-200">{turn.response}</pre>
              <div className="absolute right-1 top-1 opacity-0 group-hover:opacity-100">
                <CopyButton text={turn.response} />
              </div>
            </div>
          ) : (
            <p className="italic text-zinc-600">
              {turn.status === 'error' ? 'Error — no response returned.' : 'Empty response.'}
            </p>
          )}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
