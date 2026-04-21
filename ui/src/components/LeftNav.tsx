import { useState } from 'react'
import { TasksPanel } from './TasksPanel'
import { PromptBuilderPanel } from './PromptBuilderPanel'

type Panel = 'tasks' | 'prompt-builder' | null

export function LeftNav() {
  const [activePanel, setActivePanel] = useState<Panel>(null)

  const toggle = (panel: Panel) => {
    setActivePanel(prev => prev === panel ? null : panel)
  }

  return (
    <div className="flex shrink-0 h-full">
      {/* Icon rail */}
      <div className="flex flex-col items-center gap-2 py-3 w-10 bg-zinc-950 border-r border-zinc-800 shrink-0">
        <NavButton
          active={activePanel === 'tasks'}
          title="Tasks"
          onClick={() => toggle('tasks')}
        >
          <TasksIcon />
        </NavButton>

        <NavButton
          active={activePanel === 'prompt-builder'}
          title="Prompt Builder"
          onClick={() => toggle('prompt-builder')}
        >
          <PromptBuilderIcon />
        </NavButton>
      </div>

      {/* Expandable panel */}
      <div
        className={`overflow-hidden border-r border-zinc-800 bg-zinc-950 transition-all duration-200 ease-in-out ${
          activePanel ? 'w-72' : 'w-0'
        }`}
      >
        <div className="w-72 h-full">
          {activePanel === 'tasks' && <TasksPanel />}
          {activePanel === 'prompt-builder' && <PromptBuilderPanel />}
        </div>
      </div>
    </div>
  )
}

interface NavButtonProps {
  active: boolean
  title: string
  onClick: () => void
  children: React.ReactNode
}

function NavButton({ active, title, onClick, children }: NavButtonProps) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={`w-7 h-7 rounded flex items-center justify-center transition-colors ${
        active
          ? 'bg-zinc-800 text-cyan-400'
          : 'text-zinc-600 hover:text-zinc-400 hover:bg-zinc-900'
      }`}
    >
      {children}
    </button>
  )
}

function TasksIcon() {
  return (
    <svg viewBox="0 0 16 16" className="w-4 h-4" fill="none">
      <rect x="1.5" y="2.5" width="4" height="4" rx="0.5" stroke="currentColor" strokeWidth="1.2"/>
      <path d="M2.5 4.5L3.5 5.5L5.5 3.5" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round"/>
      <line x1="8" y1="4.5" x2="14.5" y2="4.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
      <rect x="1.5" y="9.5" width="4" height="4" rx="0.5" stroke="currentColor" strokeWidth="1.2"/>
      <line x1="8" y1="11.5" x2="14.5" y2="11.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
    </svg>
  )
}

function PromptBuilderIcon() {
  return (
    <svg viewBox="0 0 16 16" className="w-4 h-4" fill="none">
      <path d="M8 1L9.2 5.8L14 7L9.2 8.2L8 13L6.8 8.2L2 7L6.8 5.8L8 1Z" stroke="currentColor" strokeWidth="1.1" strokeLinejoin="round"/>
      <path d="M13 11L13.6 13L15 13.6L13.6 14.2L13 16L12.4 14.2L11 13.6L12.4 13L13 11Z" stroke="currentColor" strokeWidth="0.9" strokeLinejoin="round"/>
    </svg>
  )
}
