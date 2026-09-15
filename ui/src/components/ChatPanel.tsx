import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import type { ChatMessage, FocusArea, WorkStream } from '../types'

function relTime(ms: number): string {
  const diff = (Date.now() - ms) / 1000
  if (diff < 60) return `${Math.floor(diff)}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  return new Date(ms).toLocaleTimeString()
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
          isUser ? 'voice-user-bubble rounded-tr-sm' : 'voice-dann-bubble rounded-tl-sm'
        }`}
      >
        {msg.content}
      </div>
    </div>
  )
}

function NewStreamForm({
  focusAreas,
  onCreate,
  onCancel,
}: {
  focusAreas: FocusArea[]
  onCreate: (focusArea: string, title: string) => void
  onCancel: () => void
}) {
  const [focusArea, setFocusArea] = useState(focusAreas[0]?.name ?? '')
  const [title, setTitle] = useState('')

  return (
    <div className="flex flex-col gap-2 border-b border-zinc-800 p-3">
      <select
        value={focusArea}
        onChange={(e) => setFocusArea(e.target.value)}
        className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
      >
        {focusAreas.map((a) => (
          <option key={a.name} value={a.name}>{a.name}</option>
        ))}
      </select>
      <input
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Optional title"
        className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600"
      />
      <div className="flex gap-2">
        <button
          onClick={() => focusArea && onCreate(focusArea, title)}
          disabled={!focusArea}
          className="flex-1 rounded bg-teal-900/40 px-2 py-1 text-xs text-teal-300 hover:bg-teal-900/60 disabled:opacity-40"
        >
          Create
        </button>
        <button
          onClick={onCancel}
          className="rounded px-2 py-1 text-xs text-zinc-500 hover:text-zinc-300"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}

export default function ChatPanel() {
  const [streams, setStreams] = useState<WorkStream[]>([])
  const [focusAreas, setFocusAreas] = useState<FocusArea[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [active, setActive] = useState<WorkStream | null>(null)
  const [showNewForm, setShowNewForm] = useState(false)
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.listStreams().then(setStreams).catch(() => {})
    api.getFocusAreas().then(setFocusAreas).catch(() => {})
  }, [])

  useEffect(() => {
    if (!activeId) { setActive(null); return }
    api.getStream(activeId).then(setActive).catch(() => setActive(null))
  }, [activeId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [active?.messages.length])

  async function handleCreate(focusArea: string, title: string) {
    setError(null)
    try {
      const stream = await api.createStream(focusArea, title)
      setStreams((prev) => [stream, ...prev])
      setActiveId(stream.id)
      setShowNewForm(false)
    } catch {
      setError(`Could not create a stream for '${focusArea}'.`)
    }
  }

  async function handleDelete(id: string) {
    await api.deleteStream(id).catch(() => {})
    setStreams((prev) => prev.filter((s) => s.id !== id))
    if (activeId === id) setActiveId(null)
  }

  async function handleClear() {
    if (!activeId) return
    if (!window.confirm('Clear this conversation? Dann will start fresh with no memory of it.')) return
    setError(null)
    try {
      const cleared = await api.clearStream(activeId)
      setActive(cleared)
      setStreams((prev) => prev.map((s) => (s.id === cleared.id ? cleared : s)))
    } catch {
      setError('Could not clear this conversation.')
    }
  }

  async function handleSend() {
    const text = input.trim()
    if (!text || !activeId || sending) return
    setSending(true)
    setError(null)
    setInput('')
    // Optimistic append so the UI feels responsive during the (possibly
    // multi-second) Ollama round trip.
    setActive((prev) =>
      prev ? { ...prev, messages: [...prev.messages, { role: 'user', content: text, ts: Date.now() / 1000 }] } : prev
    )
    try {
      await api.sendChatMessage(activeId, text)
      const fresh = await api.getStream(activeId)
      setActive(fresh)
      setStreams((prev) => {
        const updated = prev.map((s) => (s.id === fresh.id ? fresh : s))
        return updated.sort((a, b) => b.updated_at - a.updated_at)
      })
    } catch {
      setError('Dann could not respond — is Ollama running?')
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="flex h-full">
      {/* ── Stream list ─────────────────────────────────────────────────── */}
      <div className="flex w-64 shrink-0 flex-col border-r border-zinc-800">
        <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
          <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">Work Streams</span>
          <button
            onClick={() => setShowNewForm((v) => !v)}
            className="rounded bg-teal-900/40 px-2 py-0.5 text-xs text-teal-300 hover:bg-teal-900/60"
            title="Start a new conversation"
          >
            + New conversation
          </button>
        </div>

        {showNewForm && (
          <NewStreamForm
            focusAreas={focusAreas}
            onCreate={handleCreate}
            onCancel={() => setShowNewForm(false)}
          />
        )}

        <div className="flex-1 overflow-y-auto">
          {streams.length === 0 ? (
            <p className="p-4 text-center text-xs text-zinc-600">No work streams yet.</p>
          ) : (
            streams.map((s) => (
              <div
                key={s.id}
                onClick={() => setActiveId(s.id)}
                className={`group flex cursor-pointer items-start justify-between gap-1 border-b border-zinc-900 px-3 py-2 ${
                  s.id === activeId ? 'bg-zinc-900' : 'hover:bg-zinc-900/60'
                }`}
              >
                <div className="min-w-0">
                  <p className="truncate text-xs font-medium text-zinc-200">{s.title}</p>
                  <p className="truncate text-[10px] text-zinc-500">
                    {s.title !== s.focus_area && `${s.focus_area} · `}{relTime(s.updated_at * 1000)}
                  </p>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); handleDelete(s.id) }}
                  className="shrink-0 text-zinc-700 opacity-0 group-hover:opacity-100 hover:text-zinc-400"
                  title="Delete"
                >
                  ×
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {!active ? (
        <div className="flex flex-1 items-center justify-center text-sm text-zinc-600">
          Select or create a work stream to start chatting with Dann.
        </div>
      ) : (
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex items-center justify-between gap-2 border-b border-zinc-800 px-4 py-2">
            <div className="flex min-w-0 items-center gap-2">
              <span className="truncate text-sm font-medium text-zinc-200">{active.title}</span>
              {active.title !== active.focus_area && (
                <span className="shrink-0 rounded bg-blue-900/60 px-1.5 py-0.5 text-[10px] text-blue-300">{active.focus_area}</span>
              )}
            </div>
            <div className="flex shrink-0 items-center gap-1">
              <button
                onClick={() => setShowNewForm((v) => !v)}
                className="rounded px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
                title="Start a new conversation"
              >
                New
              </button>
              <button
                onClick={handleClear}
                disabled={active.messages.length === 0}
                className="rounded px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-30"
                title="Clear this conversation's history — Dann starts fresh"
              >
                Clear
              </button>
            </div>
          </div>

          <div className="flex-1 space-y-2 overflow-y-auto p-4">
            {active.messages.length === 0 && (
              <p className="text-center text-xs text-zinc-700">No messages yet — say hello.</p>
            )}
            {active.messages.map((m, i) => <MessageBubble key={i} msg={m} />)}
            {sending && (
              <div className="flex justify-start">
                <div className="voice-dann-bubble max-w-[80%] rounded-lg rounded-tl-sm px-3 py-2 text-sm opacity-60">
                  thinking…
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {error && <p className="px-4 pb-1 text-[11px] text-neon-red">{error}</p>}

          <div className="flex gap-2 border-t border-zinc-800 p-3">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() } }}
              placeholder="Message Dann…"
              disabled={sending}
              className="flex-1 rounded border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 disabled:opacity-50"
            />
            <button
              onClick={handleSend}
              disabled={sending || !input.trim()}
              className="rounded bg-teal-900/40 px-4 py-2 text-sm text-teal-300 hover:bg-teal-900/60 disabled:opacity-40"
            >
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
