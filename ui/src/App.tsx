import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { useDannEvents } from './hooks/useDannEvents'
import { StatusBar } from './components/StatusBar'
import { ProjectPanel } from './components/ProjectPanel'
import LogPanel from './components/LogPanel'
import { api } from './lib/api'
import type { TerminalPaneHandle } from './components/TerminalPane'

// Lazy-load heavy components
const MetricsPage = lazy(() => import('./components/MetricsPage'))
const TerminalPane = lazy(() => import('./components/TerminalPane'))
const ConversationPanel = lazy(() => import('./components/ConversationPanel'))
const RunOutputPane = lazy(() => import('./components/RunOutputPane'))
const NotesPanel = lazy(() => import('./components/NotesPanel').then(m => ({ default: m.NotesPanel })))

// ── Tab types ─────────────────────────────────────────────────────────────────

interface TerminalTab {
  kind: 'terminal'
  id: string
  sessionId: string
  projectName: string
}

interface RunTab {
  kind: 'run'
  id: string
  projectName: string
}

type StaticTab = 'dann' | 'projects' | 'notes' | 'metrics' | 'voice'
type Tab = StaticTab | TerminalTab | RunTab

function tabId(t: Tab): string {
  if (typeof t === 'string') return t
  return t.id
}

function tabLabel(t: Tab): string {
  if (t === 'dann') return 'DANN'
  if (t === 'projects') return 'Projects'
  if (t === 'notes') return 'Notes'
  if (t === 'metrics') return 'Metrics'
  if (t === 'voice') return 'Voice'
  if (t.kind === 'run') return `▶ ${t.projectName}`
  return t.projectName
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  useDannEvents()

  const [tabs, setTabs] = useState<Tab[]>(['dann', 'projects', 'notes', 'voice', 'metrics'])
  const [activeTabId, setActiveTabId] = useState<string>('dann')
  const terminalRefs = useRef<Map<string, React.RefObject<TerminalPaneHandle>>>(new Map())

  // ── DANN persistent terminal ───────────────────────────────────────────────
  const [dannSessionId, setDannSessionId] = useState<string | null>(null)
  const dannTermRef = useRef<TerminalPaneHandle>(null)

  useEffect(() => {
    if (activeTabId === 'dann' && !dannSessionId) {
      api.createDannTerminal().then((s) => setDannSessionId(s.session_id)).catch((err) => {
        console.error('Failed to start DANN terminal', err)
      })
    }
  }, [activeTabId, dannSessionId])

  const openTerminal = async (projectName: string, command?: string) => {
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
      const session = await api.createTerminal(projectName, 40, 120, command)
      const newTab: TerminalTab = {
        kind: 'terminal',
        id: `term-${session.session_id}`,
        sessionId: session.session_id,
        projectName,
      }
      terminalRefs.current.set(newTab.id, { current: null } as React.RefObject<TerminalPaneHandle>)
      setTabs((prev) => [...prev, newTab])
      setActiveTabId(newTab.id)
    } catch (err) {
      console.error('Failed to open terminal for', projectName, err)
    }
  }

  const closeTab = (tab: TerminalTab | RunTab) => {
    if (tab.kind === 'terminal') {
      api.closeTerminal(tab.sessionId).catch(() => {})
      terminalRefs.current.delete(tab.id)
    }
    setTabs((prev) => prev.filter((t) => tabId(t) !== tab.id))
    if (activeTabId === tab.id) setActiveTabId('projects')
  }

  const openRunOutput = (projectName: string) => {
    const existing = tabs.find(
      (t): t is RunTab => typeof t !== 'string' && t.kind === 'run' && t.projectName === projectName
    )
    if (existing) {
      setActiveTabId(existing.id)
      return
    }
    const newTab: RunTab = { kind: 'run', id: `run-${projectName}`, projectName }
    setTabs((prev) => [...prev, newTab])
    setActiveTabId(newTab.id)
  }

  return (
    <div className="h-screen flex flex-col bg-zinc-950 overflow-hidden">
      <StatusBar />

      {/* Tab bar */}
      <nav className="flex items-end border-b border-gray-800 bg-gray-900 px-2 overflow-x-auto shrink-0">
        {tabs.map((tab) => {
          const id = tabId(tab)
          const isActive = id === activeTabId

          const isCloseable = typeof tab !== 'string'
          const isRunTab = typeof tab !== 'string' && tab.kind === 'run'
          const isDannTab = tab === 'dann'

          return (
            <div
              key={id}
              className={`group flex items-center gap-1.5 px-3 py-2 text-sm font-medium border-b-2 -mb-px cursor-pointer whitespace-nowrap transition-colors ${
                isActive
                  ? isDannTab ? 'border-teal-400 text-teal-300' : 'border-blue-500 text-white'
                  : isDannTab ? 'border-transparent text-teal-600 hover:text-teal-400' : 'border-transparent text-gray-400 hover:text-gray-200'
              }`}
              onClick={() => setActiveTabId(id)}
            >
              {isDannTab && (
                <span className="text-[10px] font-mono text-teal-500">{'>'}_</span>
              )}
              {!isDannTab && !isRunTab && typeof tab !== 'string' && (
                <span className="text-[10px] text-emerald-400 font-mono">{'>'}_</span>
              )}
              {isRunTab && (
                <span className="text-[10px] text-green-400">▶</span>
              )}
              {tabLabel(tab)}
              {isCloseable && (
                <button
                  onClick={(e) => { e.stopPropagation(); closeTab(tab as TerminalTab | RunTab) }}
                  className="ml-1 text-gray-600 hover:text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity leading-none"
                  title="Close"
                >
                  ×
                </button>
              )}
            </div>
          )
        })}
      </nav>

      {/* Tab content — all panels stacked absolutely so display:none never nukes canvas contexts */}
      <main className="flex-1 overflow-hidden relative">

        {/* DANN persistent terminal */}
        <div className={`absolute inset-0 p-2 ${activeTabId === 'dann' ? 'z-10' : 'opacity-0 pointer-events-none'}`}>
          {dannSessionId ? (
            <Suspense fallback={<div className="p-8 text-gray-500 text-sm">Opening terminal…</div>}>
              <TerminalPane
                ref={dannTermRef}
                sessionId={dannSessionId}
                isActive={activeTabId === 'dann'}
                onExit={() => setDannSessionId(null)}
              />
            </Suspense>
          ) : (
            <div className="p-8 text-teal-700 text-sm font-mono animate-pulse">Starting DANN terminal…</div>
          )}
        </div>

        <div className={`absolute inset-0 overflow-y-auto ${activeTabId === 'projects' ? '' : 'opacity-0 pointer-events-none'}`}>
          <div className="mx-auto max-w-4xl px-4 py-4">
            <ProjectPanel onOpenTerminal={openTerminal} onRunOpened={openRunOutput} />
          </div>
        </div>

        <div className={`absolute inset-0 overflow-y-auto ${activeTabId === 'notes' ? '' : 'opacity-0 pointer-events-none'}`}>
          <div className="mx-auto max-w-4xl px-4 py-4">
            <Suspense fallback={<div className="p-8 text-gray-500 text-sm">Loading…</div>}>
              <NotesPanel onOpenTerminal={openTerminal} />
            </Suspense>
          </div>
        </div>

        <div className={`absolute inset-0 overflow-y-auto ${activeTabId === 'metrics' ? '' : 'opacity-0 pointer-events-none'}`}>
          <Suspense fallback={<div className="p-8 text-gray-500 text-sm">Loading…</div>}>
            <MetricsPage />
          </Suspense>
        </div>

        <div className={`absolute inset-0 overflow-hidden ${activeTabId === 'voice' ? '' : 'opacity-0 pointer-events-none'}`}>
          <Suspense fallback={<div className="p-8 text-gray-500 text-sm">Loading…</div>}>
            <ConversationPanel />
          </Suspense>
        </div>

        {/* Terminal tabs */}
        {tabs.filter((t): t is TerminalTab => typeof t !== 'string' && t.kind === 'terminal').map((tab) => (
          <div
            key={tab.id}
            className={`absolute inset-0 p-2 ${activeTabId === tab.id ? 'z-10' : 'opacity-0 pointer-events-none'}`}
          >
            <Suspense fallback={<div className="p-8 text-gray-500 text-sm">Opening terminal…</div>}>
              <TerminalPane
                ref={terminalRefs.current.get(tab.id) ?? null}
                sessionId={tab.sessionId}
                isActive={activeTabId === tab.id}
                onExit={() => closeTab(tab)}
              />
            </Suspense>
          </div>
        ))}

        {/* Run output tabs */}
        {tabs.filter((t): t is RunTab => typeof t !== 'string' && t.kind === 'run').map((tab) => (
          <div
            key={tab.id}
            className={`absolute inset-0 p-2 ${activeTabId === tab.id ? 'z-10' : 'invisible pointer-events-none'}`}
          >
            <Suspense fallback={<div className="p-8 text-gray-500 text-sm">Connecting…</div>}>
              <RunOutputPane
                projectName={tab.projectName}
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
