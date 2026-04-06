import { lazy, Suspense, useState } from 'react'
import { useDannEvents } from './hooks/useDannEvents'
import { StatusBar } from './components/StatusBar'
import { ProjectPanel } from './components/ProjectPanel'
import MetricsBar from './components/MetricsBar'
import LogPanel from './components/LogPanel'
import { api } from './lib/api'

// Lazy-load heavy components
const MetricsPage = lazy(() => import('./components/MetricsPage'))
const TerminalPane = lazy(() => import('./components/TerminalPane'))
const ConversationPanel = lazy(() => import('./components/ConversationPanel'))

// ── Tab types ─────────────────────────────────────────────────────────────────

interface TerminalTab {
  kind: 'terminal'
  id: string          // unique tab id
  sessionId: string   // backend PTY session id
  projectName: string
}

type StaticTab = 'projects' | 'metrics' | 'voice'
type Tab = StaticTab | TerminalTab

function tabId(t: Tab): string {
  if (typeof t === 'string') return t
  return t.id
}

function tabLabel(t: Tab): string {
  if (t === 'projects') return 'Projects'
  if (t === 'metrics') return 'Metrics'
  if (t === 'voice') return 'Voice'
  return t.projectName
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  useDannEvents()

  const [tabs, setTabs] = useState<Tab[]>(['projects', 'voice', 'metrics'])
  const [activeTabId, setActiveTabId] = useState<string>('projects')

  const openTerminal = async (projectName: string) => {
    // If a terminal tab for this project already exists, just switch to it
    const existing = tabs.find(
      (t): t is TerminalTab => typeof t !== 'string' && t.projectName === projectName
    )
    if (existing) {
      setActiveTabId(existing.id)
      return
    }

    try {
      // Ask the backend to create a PTY session
      const session = await api.createTerminal(projectName)
      const newTab: TerminalTab = {
        kind: 'terminal',
        id: `term-${session.session_id}`,
        sessionId: session.session_id,
        projectName,
      }
      setTabs((prev) => [...prev, newTab])
      setActiveTabId(newTab.id)
    } catch (err) {
      console.error('Failed to open terminal for', projectName, err)
    }
  }

  const closeTab = (tab: TerminalTab) => {
    // Tell the backend to clean up the PTY session
    api.closeTerminal(tab.sessionId).catch(() => {})
    setTabs((prev) => prev.filter((t) => tabId(t) !== tab.id))
    if (activeTabId === tab.id) setActiveTabId('projects')
  }

  return (
    <div className="h-screen flex flex-col bg-zinc-950 overflow-hidden">
      <StatusBar />
      <MetricsBar />

      {/* Tab bar */}
      <nav className="flex items-end border-b border-gray-800 bg-gray-900 px-2 overflow-x-auto shrink-0">
        {tabs.map((tab) => {
          const id = tabId(tab)
          const isActive = id === activeTabId
          const isTerminal = typeof tab !== 'string'

          return (
            <div
              key={id}
              className={`group flex items-center gap-1.5 px-3 py-2 text-sm font-medium border-b-2 -mb-px cursor-pointer whitespace-nowrap transition-colors ${
                isActive
                  ? 'border-blue-500 text-white'
                  : 'border-transparent text-gray-400 hover:text-gray-200'
              }`}
              onClick={() => setActiveTabId(id)}
            >
              {isTerminal && (
                <span className="text-[10px] text-emerald-400 font-mono">{'>'}_</span>
              )}
              {tabLabel(tab)}
              {isTerminal && (
                <button
                  onClick={(e) => { e.stopPropagation(); closeTab(tab as TerminalTab) }}
                  className="ml-1 text-gray-600 hover:text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity leading-none"
                  title="Close terminal"
                >
                  ×
                </button>
              )}
            </div>
          )
        })}
      </nav>

      {/* Tab content */}
      <main className="flex-1 overflow-hidden">
        {/* Static panels — keep mounted so state survives tab switching */}
        <div className={`h-full overflow-y-auto ${activeTabId === 'projects' ? '' : 'hidden'}`}>
          <div className="mx-auto max-w-4xl px-4 py-4">
            <ProjectPanel onOpenTerminal={openTerminal} />
          </div>
        </div>

        <div className={`h-full overflow-y-auto ${activeTabId === 'metrics' ? '' : 'hidden'}`}>
          <Suspense fallback={<div className="p-8 text-gray-500 text-sm">Loading…</div>}>
            <MetricsPage />
          </Suspense>
        </div>

        <div className={`h-full overflow-hidden ${activeTabId === 'voice' ? '' : 'hidden'}`}>
          <Suspense fallback={<div className="p-8 text-gray-500 text-sm">Loading…</div>}>
            <ConversationPanel />
          </Suspense>
        </div>

        {/* Terminal tabs */}
        {tabs.filter((t): t is TerminalTab => typeof t !== 'string').map((tab) => (
          <div
            key={tab.id}
            className={`h-full p-2 ${activeTabId === tab.id ? '' : 'hidden'}`}
          >
            <Suspense fallback={<div className="p-8 text-gray-500 text-sm">Opening terminal…</div>}>
              <TerminalPane
                sessionId={tab.sessionId}
                onExit={() => closeTab(tab)}
              />
            </Suspense>
          </div>
        ))}
      </main>

      <LogPanel />
    </div>
  )
}
