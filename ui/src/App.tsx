import { Suspense, lazy, useState } from 'react'
import { useDannEvents } from './hooks/useDannEvents'
import { useFocusModeShortcut } from './hooks/useFocusModeShortcut'
import { useDannStore } from './hooks/useDannState'
import { StatusBar } from './components/StatusBar'
import { ModulesPanel } from './components/ModulesPanel'
import { FocusAreasPanel } from './components/FocusAreasPanel'
import LogPanel from './components/LogPanel'
import { LeftNav } from './components/LeftNav'
import { VoicePanel } from './components/VoicePanel'
import { VoiceWidget } from './components/VoiceWidget'
import { TerminalIndicator } from './components/TerminalIndicator'
import { FocusMode } from './components/focus/FocusMode'

// Lazy-load heavy components
const MetricsPage = lazy(() => import('./components/MetricsPage'))
const NotesPanel = lazy(() => import('./components/NotesPanel').then(m => ({ default: m.NotesPanel })))
const ChatPanel = lazy(() => import('./components/ChatPanel')) // the main "DANN" tab: chat + per-focus-area terminal

// ── Tab types ─────────────────────────────────────────────────────────────────
// Terminals live inside the DANN tab now (one per active work stream's focus
// area, see ChatPanel) — there's no standalone terminal-tab system anymore
// since the two panels that used to open one (Projects, Notes) either no
// longer exist or no longer map onto a focus area.

type Tab = 'dann' | 'focus' | 'modules' | 'notes' | 'metrics'

function tabLabel(t: Tab): string {
  if (t === 'dann') return 'DANN'
  if (t === 'focus') return 'Focus Areas'
  if (t === 'modules') return 'Modules'
  if (t === 'notes') return 'Notes'
  return 'Metrics'
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  useDannEvents()
  useFocusModeShortcut()

  const focusMode = useDannStore(s => s.focusMode)
  const setFocusMode = useDannStore(s => s.setFocusMode)

  const tabs: Tab[] = ['dann', 'focus', 'modules', 'notes', 'metrics']
  const [activeTabId, setActiveTabId] = useState<Tab>('dann')

  return (
    <>
    <div className={`h-screen flex flex-col bg-zinc-950 overflow-hidden ${focusMode ? 'invisible pointer-events-none' : ''}`}>
      <StatusBar />

      {/* Tab bar */}
      <nav className="flex items-end border-b border-gray-800 bg-gray-900 px-2 overflow-x-auto shrink-0">
        {tabs.map((tab) => {
          const isActive = tab === activeTabId
          const isDannTab = tab === 'dann'

          return (
            <div
              key={tab}
              className={`group flex items-center gap-1.5 px-3 py-2 text-sm font-medium border-b-2 -mb-px cursor-pointer whitespace-nowrap transition-colors ${
                isActive
                  ? isDannTab ? 'border-teal-400 text-teal-300' : 'border-blue-500 text-white'
                  : isDannTab ? 'border-transparent text-teal-600 hover:text-teal-400' : 'border-transparent text-gray-400 hover:text-gray-200'
              }`}
              onClick={() => setActiveTabId(tab)}
            >
              {isDannTab && (
                <span className="text-[10px] font-mono text-teal-500">{'>'}_</span>
              )}
              {isDannTab && (
                <button
                  onClick={(e) => { e.stopPropagation(); setFocusMode(true) }}
                  title="Enter Focus Mode (Cmd/Ctrl+Shift+F)"
                  className="ml-1 text-[9px] text-teal-700 hover:text-teal-400 transition-colors leading-none"
                >
                  ⛶
                </button>
              )}
              {tabLabel(tab)}
            </div>
          )
        })}
      </nav>

      {/* Body: left nav + tab content + persistent voice panel */}
      <div className="flex flex-1 overflow-hidden">
      <LeftNav />

      {/* Tab content — all panels stacked absolutely so display:none never nukes canvas contexts */}
      <main className="flex-1 overflow-hidden relative">

        {/* DANN — chat with Dann + the live terminal for whichever work stream's focus area is active */}
        <div className={`absolute inset-0 overflow-hidden ${activeTabId === 'dann' ? '' : 'opacity-0 pointer-events-none'}`}>
          <Suspense fallback={<div className="p-8 text-gray-500 text-sm">Loading…</div>}>
            <ChatPanel />
          </Suspense>
        </div>

        <div className={`absolute inset-0 overflow-y-auto ${activeTabId === 'focus' ? '' : 'opacity-0 pointer-events-none'}`}>
          <div className="mx-auto max-w-4xl px-4 py-4">
            <FocusAreasPanel />
          </div>
        </div>

        <div className={`absolute inset-0 overflow-y-auto ${activeTabId === 'modules' ? '' : 'opacity-0 pointer-events-none'}`}>
          <div className="mx-auto max-w-4xl px-4 py-4">
            <ModulesPanel />
          </div>
        </div>

        <div className={`absolute inset-0 overflow-y-auto ${activeTabId === 'notes' ? '' : 'opacity-0 pointer-events-none'}`}>
          <div className="mx-auto max-w-4xl px-4 py-4">
            <Suspense fallback={<div className="p-8 text-gray-500 text-sm">Loading…</div>}>
              <NotesPanel />
            </Suspense>
          </div>
        </div>

        <div className={`absolute inset-0 overflow-y-auto ${activeTabId === 'metrics' ? '' : 'opacity-0 pointer-events-none'}`}>
          <Suspense fallback={<div className="p-8 text-gray-500 text-sm">Loading…</div>}>
            <MetricsPage />
          </Suspense>
        </div>
      </main>

      {/* Permanent voice panel — always visible right sidebar */}
      <VoicePanel />
      </div>

      <VoiceWidget />
      <LogPanel />
      <TerminalIndicator />
    </div>
    <FocusMode active={focusMode} onExit={() => setFocusMode(false)} />
    </>
  )
}
