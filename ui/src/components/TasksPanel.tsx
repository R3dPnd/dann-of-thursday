import { useEffect, useRef, useState } from 'react'

interface Task {
  id: string
  text: string
  done: boolean
  createdAt: number
}

const STORAGE_KEY = 'dann-tasks'

function loadTasks(): Task[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveTasks(tasks: Task[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks))
}

export function TasksPanel() {
  const [tasks, setTasks] = useState<Task[]>(loadTasks)
  const [newText, setNewText] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editText, setEditText] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    saveTasks(tasks)
  }, [tasks])

  const addTask = () => {
    const text = newText.trim()
    if (!text) return
    setTasks(prev => [...prev, { id: crypto.randomUUID(), text, done: false, createdAt: Date.now() }])
    setNewText('')
    inputRef.current?.focus()
  }

  const toggleTask = (id: string) => {
    setTasks(prev => prev.map(t => t.id === id ? { ...t, done: !t.done } : t))
  }

  const deleteTask = (id: string) => {
    setTasks(prev => prev.filter(t => t.id !== id))
  }

  const startEdit = (task: Task) => {
    setEditingId(task.id)
    setEditText(task.text)
  }

  const commitEdit = () => {
    if (!editingId) return
    const text = editText.trim()
    if (text) {
      setTasks(prev => prev.map(t => t.id === editingId ? { ...t, text } : t))
    }
    setEditingId(null)
  }

  const clearDone = () => {
    setTasks(prev => prev.filter(t => !t.done))
  }

  const pending = tasks.filter(t => !t.done)
  const done = tasks.filter(t => t.done)

  return (
    <div className="flex flex-col h-full text-xs">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800 shrink-0">
        <span className="text-neon-cyan font-bold tracking-widest uppercase text-[10px]">Tasks</span>
        {done.length > 0 && (
          <button
            onClick={clearDone}
            className="text-zinc-600 hover:text-zinc-400 transition-colors text-[10px]"
          >
            clear done
          </button>
        )}
      </div>

      {/* Add task */}
      <div className="flex gap-1.5 px-3 py-2 border-b border-zinc-800 shrink-0">
        <input
          ref={inputRef}
          value={newText}
          onChange={e => setNewText(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') addTask() }}
          placeholder="new task…"
          className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-zinc-200 placeholder-zinc-700 text-[11px] focus:border-cyan-700"
        />
        <button
          onClick={addTask}
          className="px-2 py-1 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded text-zinc-300 transition-colors"
        >
          +
        </button>
      </div>

      {/* Task list */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
        {tasks.length === 0 && (
          <p className="text-zinc-700 text-center mt-6 text-[10px]">no tasks yet</p>
        )}

        {/* Pending */}
        {pending.map(task => (
          <TaskRow
            key={task.id}
            task={task}
            isEditing={editingId === task.id}
            editText={editText}
            onEditTextChange={setEditText}
            onToggle={() => toggleTask(task.id)}
            onStartEdit={() => startEdit(task)}
            onCommitEdit={commitEdit}
            onDelete={() => deleteTask(task.id)}
          />
        ))}

        {/* Done */}
        {done.length > 0 && (
          <>
            <div className="pt-2 pb-1">
              <span className="text-zinc-700 text-[10px] uppercase tracking-widest">Done</span>
            </div>
            {done.map(task => (
              <TaskRow
                key={task.id}
                task={task}
                isEditing={editingId === task.id}
                editText={editText}
                onEditTextChange={setEditText}
                onToggle={() => toggleTask(task.id)}
                onStartEdit={() => startEdit(task)}
                onCommitEdit={commitEdit}
                onDelete={() => deleteTask(task.id)}
              />
            ))}
          </>
        )}
      </div>

      {/* Footer count */}
      <div className="px-3 py-1.5 border-t border-zinc-800 shrink-0 text-zinc-700 text-[10px]">
        {pending.length} remaining · {done.length} done
      </div>
    </div>
  )
}

interface TaskRowProps {
  task: Task
  isEditing: boolean
  editText: string
  onEditTextChange: (v: string) => void
  onToggle: () => void
  onStartEdit: () => void
  onCommitEdit: () => void
  onDelete: () => void
}

function TaskRow({ task, isEditing, editText, onEditTextChange, onToggle, onStartEdit, onCommitEdit, onDelete }: TaskRowProps) {
  return (
    <div className="group flex items-start gap-2 py-1">
      <button
        onClick={onToggle}
        className={`mt-0.5 w-3.5 h-3.5 shrink-0 border rounded-sm transition-colors flex items-center justify-center ${
          task.done
            ? 'border-cyan-700 bg-cyan-900/40 text-cyan-400'
            : 'border-zinc-700 bg-transparent hover:border-cyan-700'
        }`}
      >
        {task.done && (
          <svg viewBox="0 0 10 10" className="w-2.5 h-2.5" fill="none">
            <path d="M1.5 5L4 7.5L8.5 2.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        )}
      </button>

      {isEditing ? (
        <input
          autoFocus
          value={editText}
          onChange={e => onEditTextChange(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' || e.key === 'Escape') onCommitEdit() }}
          onBlur={onCommitEdit}
          className="flex-1 bg-zinc-900 border border-cyan-700 rounded px-1.5 py-0.5 text-zinc-200 text-[11px]"
        />
      ) : (
        <span
          onDoubleClick={onStartEdit}
          className={`flex-1 leading-snug cursor-default select-none ${
            task.done ? 'line-through text-zinc-600' : 'text-zinc-300'
          }`}
        >
          {task.text}
        </span>
      )}

      <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
        {!isEditing && (
          <button onClick={onStartEdit} className="text-zinc-600 hover:text-zinc-400 transition-colors">
            <svg viewBox="0 0 12 12" className="w-3 h-3" fill="none">
              <path d="M2 8.5L8.5 2 10 3.5 3.5 10H2V8.5Z" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        )}
        <button onClick={onDelete} className="text-zinc-700 hover:text-red-500 transition-colors">
          <svg viewBox="0 0 12 12" className="w-3 h-3" fill="none">
            <path d="M2 2L10 10M10 2L2 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </button>
      </div>
    </div>
  )
}
