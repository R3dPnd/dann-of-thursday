import type { Project } from '../types'
import { useDannStore } from '../hooks/useDannState'

function NoteCard({ note }: { note: Project }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 hover:border-zinc-700 transition-colors">
      <div className="flex items-center gap-3 px-4 py-3">
        <div className="min-w-0 flex-1">
          <p className="truncate font-medium text-zinc-100">{note.name}</p>
          <p className="truncate text-xs text-zinc-500" title={note.path}>
            {note.path}
          </p>
        </div>
      </div>
    </div>
  )
}

export function NotesPanel() {
  const noteProjects = useDannStore((s) => s.noteProjects)

  if (noteProjects.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-zinc-600">
        <p className="text-sm">No notes repositories found.</p>
        <p className="mt-1 text-xs">Add entries under "notes" in config.yaml.</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      {noteProjects.map((note) => (
        <NoteCard key={note.name} note={note} />
      ))}
    </div>
  )
}
