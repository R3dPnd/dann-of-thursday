import { useState } from 'react'
import type { Module } from '../types'
import { useDannStore } from '../hooks/useDannState'
import { api } from '../lib/api'

function statusBadge(mod: Module): { label: string; className: string } {
  if (mod.always_on) return { label: 'Always on', className: 'bg-zinc-800 text-zinc-400' }
  if (mod.enabled) return { label: 'Active', className: 'bg-blue-950/40 text-blue-300' }
  return { label: 'Available', className: 'bg-zinc-900 text-zinc-500' }
}

function ModuleCard({ mod }: { mod: Module }) {
  const setModuleEnabled = useDannStore((s) => s.setModuleEnabled)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const badge = statusBadge(mod)

  const toggle = async () => {
    setPending(true)
    setError(null)
    try {
      if (mod.enabled) {
        await api.disableModule(mod.name)
        setModuleEnabled(mod.name, false)
      } else {
        await api.enableModule(mod.name)
        setModuleEnabled(mod.name, true)
      }
    } catch {
      setError(`Could not ${mod.enabled ? 'stop' : 'start'} '${mod.name}'.`)
    } finally {
      setPending(false)
    }
  }

  return (
    <div
      className={`rounded-lg border transition-colors ${
        mod.enabled ? 'border-blue-900/60 bg-blue-950/10' : 'border-zinc-800 bg-zinc-900 hover:border-zinc-700'
      }`}
    >
      <div className="flex items-center gap-3 px-4 py-3">
        <span
          className={`inline-block h-2 w-2 flex-shrink-0 rounded-full ${
            mod.enabled ? 'bg-blue-400 pulse-dot' : 'bg-zinc-700'
          }`}
        />

        <div className="min-w-0 flex-1">
          <p className="truncate font-medium text-zinc-100">{mod.name}</p>
          <p className="truncate text-xs text-zinc-500" title={mod.description}>
            {mod.description}
          </p>
        </div>

        <span className={`flex-shrink-0 rounded-full px-2 py-0.5 text-xs ${badge.className}`}>
          {badge.label}
        </span>

        {!mod.always_on && (
          <button
            onClick={toggle}
            disabled={pending}
            className={`flex-shrink-0 rounded px-2 py-1 text-xs transition-colors disabled:opacity-40 ${
              mod.enabled
                ? 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
                : 'bg-blue-900/40 text-blue-300 hover:bg-blue-900/60'
            }`}
          >
            {pending ? '…' : mod.enabled ? 'Stop' : 'Start'}
          </button>
        )}
      </div>

      {error && <p className="px-4 pb-2 text-[11px] text-neon-red">{error}</p>}
    </div>
  )
}

export function ModulesPanel() {
  const modules = useDannStore((s) => s.modules)

  if (modules.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-zinc-600">
        <p className="text-sm">No modules configured.</p>
        <p className="mt-1 text-xs">Modules are listed under mcp.servers in config.yaml.</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs text-zinc-500">
        Optional modules start on demand — Dann enables one itself the first time a turn needs it,
        or start one here yourself.
      </p>
      {modules.map((mod) => (
        <ModuleCard key={mod.name} mod={mod} />
      ))}
    </div>
  )
}
