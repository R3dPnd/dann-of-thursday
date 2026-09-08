import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import type { ChatMessage, Project, WorkStream } from '../types'

const TerminalPane = lazy(() => import('./TerminalPane'))

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
  projects,
  onCreate,
  onCancel,
}: {
  projects: Project[]
  onCreate: (project: string, title: string) => void
  onCancel: () => void
}) {
  const [project, setProject] = useState(projects[0]?.name ?? '')
  const [title, setTitle] = useState('')

  return (
    <div className="flex flex-col gap-2 border-b border-zinc-800 p-3">
      <select
        value={project}
        onChange={(e) => setProject(e.target.value)}
        className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
      >
        {projects.map((p) => (
          <option key={p.name} value={p.name}>{p.name}</option>
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
          onClick={() => project && onCreate(project, title)}
          disabled={!project}
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
  const [projects, setProjects] = useState<Project[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [active, setActive] = useState<WorkStream | null>(null)
  const [showNewForm, setShowNewForm] = useState(false)
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [terminalSessionId, setTerminalSessionId] = useState<string | null>(null)
  const [terminalLoading, setTerminalLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.listStreams().then(setStreams).catch(() => {})
    api.getProjects().then(setProjects).catch(() => {})
  }, [])

  useEffect(() => {
    if (!activeId) { setActive(null); return }
    api.getStream(activeId).then(setActive).catch(() => setActive(null))
  }, [activeId])

  // A terminal belongs to a project, not a stream — reused across every
  // stream for that project, same as Dann's own open_claude_code tool does
  // server-side. Look for an existing live session whenever the active
  // project changes; don't auto-create one (that spawns a real `claude`
  // process) — only findExistingTerminal, handleOpenTerminal creates.
  useEffect(() => {
    setTerminalSessionId(null)
    if (!active?.project) return
    findExistingTerminal(active.project)
  }, [active?.project])

  async function findExistingTerminal(project: string) {
    try {
      const sessions = await api.listTerminals()
      const match = sessions.find((s) => s.project_name === project && s.alive)
      setTerminalSessionId(match ? match.session_id : null)
    } catch {
      // dashboard terminal API unreachable — leave as not-found
    }
  }

  async function handleOpenTerminal() {
    if (!active?.project || terminalLoading) return
    setTerminalLoading(true)
    try {
      const session = await api.createTerminal(active.project)
      setTerminalSessionId(session.session_id)
    } catch {
      setError('Could not open a terminal for this project.')
    } finally {
      setTerminalLoading(false)
    }
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [active?.messages.length])

  async function handleCreate(project: string, title: string) {
    setError(null)
    try {
      const stream = await api.createStream(project, title)
      setStreams((prev) => [stream, ...prev])
      setActiveId(stream.id)
      setShowNewForm(false)
    } catch {
      setError(`Could not create a stream for '${project}'.`)
    }
  }

  async function handleDelete(id: string) {
    await api.deleteStream(id).catch(() => {})
    setStreams((prev) => prev.filter((s) => s.id !== id))
    if (activeId === id) setActiveId(null)
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
      if (!terminalSessionId && fresh.project) findExistingTerminal(fresh.project)
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
            className="rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-300 hover:bg-zinc-700"
          >
            + New
          </button>
        </div>

        {showNewForm && (
          <NewStreamForm
            projects={projects}
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
                    {s.title !== s.project && `${s.project} · `}{relTime(s.updated_at * 1000)}
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
        <>
          {/* ── Chat thread ─────────────────────────────────────────────── */}
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-2">
              <span className="text-sm font-medium text-zinc-200">{active.title}</span>
              {active.title !== active.project && (
                <span className="rounded bg-blue-900/60 px-1.5 py-0.5 text-[10px] text-blue-300">{active.project}</span>
              )}
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

          {/* ── Project terminal ────────────────────────────────────────── */}
          <div className="flex w-[45%] min-w-0 flex-col border-l border-zinc-800">
            <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
              <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">
                Terminal — {active.project}
              </span>
              {terminalSessionId && (
                <button
                  onClick={() => findExistingTerminal(active.project)}
                  className="text-zinc-600 hover:text-zinc-400"
                  title="Refresh"
                >
                  ↻
                </button>
              )}
            </div>
            {terminalSessionId ? (
              <div className="flex-1 p-2">
                <Suspense fallback={<div className="p-4 text-xs text-zinc-500">Loading terminal…</div>}>
                  <TerminalPane
                    key={terminalSessionId}
                    sessionId={terminalSessionId}
                    isActive
                    onExit={() => setTerminalSessionId(null)}
                  />
                </Suspense>
              </div>
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center text-xs text-zinc-600">
                <p>No terminal open for '{active.project}' yet.</p>
                <button
                  onClick={handleOpenTerminal}
                  disabled={terminalLoading}
                  className="rounded bg-zinc-800 px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 disabled:opacity-40"
                >
                  {terminalLoading ? 'Opening…' : 'Open Terminal'}
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
