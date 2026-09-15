import { useEffect, useState } from 'react'
import type { FocusArea, FocusAreaNote } from '../types'
import { useDannStore } from '../hooks/useDannState'
import { api } from '../lib/api'

function relTime(sec: number): string {
  const diff = Date.now() / 1000 - sec
  if (diff < 60) return `${Math.floor(diff)}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return new Date(sec * 1000).toLocaleDateString()
}

function NoteCard({ note, onDelete }: { note: FocusAreaNote; onDelete: () => void }) {
  const [expanded, setExpanded] = useState(false)
  const long = note.content.length > 140 || note.content.includes('\n')

  return (
    <div className="rounded border border-zinc-800 bg-zinc-950 px-3 py-2">
      <div className="flex items-start justify-between gap-2">
        <button
          onClick={() => long && setExpanded((v) => !v)}
          className={`min-w-0 flex-1 text-left ${long ? 'cursor-pointer' : 'cursor-default'}`}
        >
          <p className="truncate text-xs font-medium text-zinc-200">{note.title}</p>
          <p className={`text-[11px] text-zinc-500 ${expanded ? 'whitespace-pre-wrap' : 'truncate'}`}>
            {note.content}
          </p>
        </button>
        <div className="flex shrink-0 items-center gap-2">
          <span className="text-[10px] text-zinc-600">{relTime(note.modified_at)}</span>
          <button
            onClick={onDelete}
            className="text-zinc-700 hover:text-neon-red"
            title="Delete note"
          >
            ×
          </button>
        </div>
      </div>
    </div>
  )
}

function AddNoteForm({ onAdd }: { onAdd: (title: string, content: string) => Promise<void> }) {
  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  const [saving, setSaving] = useState(false)

  const submit = async () => {
    if (!content.trim() || saving) return
    setSaving(true)
    try {
      await onAdd(title.trim(), content.trim())
      setTitle('')
      setContent('')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex flex-col gap-1.5 rounded border border-zinc-800 bg-zinc-950 p-2">
      <input
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Optional title"
        className="rounded border border-zinc-800 bg-zinc-900 px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600"
      />
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        placeholder="Note content — remembered in every future conversation about this focus area"
        rows={2}
        className="resize-none rounded border border-zinc-800 bg-zinc-900 px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600"
      />
      <button
        onClick={submit}
        disabled={!content.trim() || saving}
        className="self-end rounded bg-teal-900/40 px-2 py-1 text-xs text-teal-300 hover:bg-teal-900/60 disabled:opacity-40"
      >
        {saving ? 'Saving…' : 'Add note'}
      </button>
    </div>
  )
}

function FocusAreaCard({ area }: { area: FocusArea }) {
  const [expanded, setExpanded] = useState(false)
  const [notes, setNotes] = useState<FocusAreaNote[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const loadNotes = () => {
    api.getFocusAreaNotes(area.name).then(setNotes).catch(() => setError('Could not load notes.'))
  }

  useEffect(() => {
    if (expanded && notes === null) loadNotes()
  }, [expanded]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleAdd = async (title: string, content: string) => {
    setError(null)
    try {
      const note = await api.createFocusAreaNote(area.name, content, title || undefined)
      setNotes((prev) => [note, ...(prev ?? [])])
    } catch {
      setError('Could not save note.')
    }
  }

  const handleDelete = async (filename: string) => {
    setNotes((prev) => prev?.filter((n) => n.filename !== filename) ?? null)
    try {
      await api.deleteFocusAreaNote(area.name, filename)
    } catch {
      setError('Could not delete note — refresh to check.')
    }
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 hover:border-zinc-700 transition-colors">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center gap-3 px-4 py-3 text-left"
      >
        <div className="min-w-0 flex-1">
          <p className="truncate font-medium text-zinc-100">{area.name}</p>
          <p className="text-xs text-zinc-500">{area.description}</p>
        </div>

        {area.module && (
          <span
            className={`flex-shrink-0 rounded-full px-2 py-0.5 text-xs ${
              area.module_enabled ? 'bg-blue-950/40 text-blue-300' : 'bg-zinc-800 text-zinc-500'
            }`}
            title={`Backed by the '${area.module}' module`}
          >
            {area.module_enabled ? `${area.module} active` : area.module}
          </span>
        )}

        <svg
          className={`h-4 w-4 shrink-0 text-zinc-500 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="flex flex-col gap-2 border-t border-zinc-800 px-4 py-3">
          {error && <p className="text-[11px] text-neon-red">{error}</p>}
          {notes === null ? (
            <p className="text-xs text-zinc-600">Loading notes…</p>
          ) : notes.length === 0 ? (
            <p className="text-xs text-zinc-600">No notes yet.</p>
          ) : (
            <div className="flex flex-col gap-1.5">
              {notes.map((note) => (
                <NoteCard key={note.filename} note={note} onDelete={() => handleDelete(note.filename)} />
              ))}
            </div>
          )}
          <AddNoteForm onAdd={handleAdd} />
        </div>
      )}
    </div>
  )
}

export function FocusAreasPanel() {
  const focusAreas = useDannStore((s) => s.focusAreas)

  if (focusAreas.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-zinc-600">
        <p className="text-sm">No focus areas configured.</p>
        <p className="mt-1 text-xs">Add entries under focus_areas in config.yaml.</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs text-zinc-500">
        Topics Dann leans into — asks follow-ups on, connects new information to. A module badge
        means real data backs it; without one, it's just something Dann knows you care about.
        Click a card to see or add its saved notes — permanent context included in every future
        conversation about it.
      </p>
      {focusAreas.map((area) => (
        <FocusAreaCard key={area.name} area={area} />
      ))}
    </div>
  )
}
