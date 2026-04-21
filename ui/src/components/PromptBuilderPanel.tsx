import { useState } from 'react'
import { api } from '../lib/api'

interface Section {
  id: string
  title: string
  content: string
}

const DEFAULT_SECTIONS: Section[] = [
  { id: 'thoughts', title: 'Thoughts', content: '' },
  { id: 'goals',    title: 'Goals',    content: '' },
  { id: 'notes',    title: 'Notes',    content: '' },
]

interface BuildResult {
  recommended_model: string
  prompt: string
  reasoning: string
}

const MODEL_LABELS: Record<string, string> = {
  'claude-opus-4-6':           'Opus 4.6',
  'claude-sonnet-4-6':         'Sonnet 4.6',
  'claude-haiku-4-5-20251001': 'Haiku 4.5',
}

const MODEL_COLORS: Record<string, string> = {
  'claude-opus-4-6':           'text-neon-violet',
  'claude-sonnet-4-6':         'text-neon-cyan',
  'claude-haiku-4-5-20251001': 'text-neon-green',
}

export function PromptBuilderPanel() {
  const [sections, setSections] = useState<Section[]>(DEFAULT_SECTIONS)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<BuildResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const updateSection = (id: string, content: string) => {
    setSections(prev => prev.map(s => s.id === id ? { ...s, content } : s))
  }

  const addSection = () => {
    setSections(prev => [...prev, {
      id: crypto.randomUUID(),
      title: 'Section',
      content: '',
    }])
  }

  const removeSection = (id: string) => {
    setSections(prev => prev.filter(s => s.id !== id))
  }

  const updateTitle = (id: string, title: string) => {
    setSections(prev => prev.map(s => s.id === id ? { ...s, title } : s))
  }

  const hasContent = sections.some(s => s.content.trim().length > 0)

  const build = async () => {
    if (!hasContent) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const thoughts = sections.find(s => s.id === 'thoughts')?.content ?? ''
      const goals    = sections.find(s => s.id === 'goals')?.content ?? ''
      const notes    = sections.find(s => s.id === 'notes')?.content ?? ''
      const custom   = sections
        .filter(s => !['thoughts', 'goals', 'notes'].includes(s.id))
        .map(s => ({ title: s.title, content: s.content }))

      const res = await api.buildPrompt({ thoughts, goals, notes, custom_sections: custom })
      setResult(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  const copyPrompt = () => {
    if (!result) return
    navigator.clipboard.writeText(result.prompt).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  const reset = () => {
    setSections(DEFAULT_SECTIONS.map(s => ({ ...s, content: '' })))
    setResult(null)
    setError(null)
  }

  return (
    <div className="flex flex-col h-full text-xs">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800 shrink-0">
        <span className="text-neon-violet font-bold tracking-widest uppercase text-[10px]">Prompt Builder</span>
        {(result || error) && (
          <button onClick={reset} className="text-zinc-600 hover:text-zinc-400 transition-colors text-[10px]">
            reset
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Input sections */}
        {!result && (
          <div className="px-3 py-2 space-y-3">
            {sections.map(section => (
              <div key={section.id} className="space-y-1">
                <div className="flex items-center justify-between">
                  {['thoughts', 'goals', 'notes'].includes(section.id) ? (
                    <span className="text-zinc-500 uppercase tracking-widest text-[10px]">{section.title}</span>
                  ) : (
                    <input
                      value={section.title}
                      onChange={e => updateTitle(section.id, e.target.value)}
                      className="text-zinc-500 uppercase tracking-widest text-[10px] bg-transparent border-b border-zinc-800 focus:border-violet-700 outline-none w-24"
                    />
                  )}
                  {!['thoughts', 'goals', 'notes'].includes(section.id) && (
                    <button
                      onClick={() => removeSection(section.id)}
                      className="text-zinc-700 hover:text-red-500 transition-colors"
                    >
                      <svg viewBox="0 0 12 12" className="w-3 h-3" fill="none">
                        <path d="M2 2L10 10M10 2L2 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                      </svg>
                    </button>
                  )}
                </div>
                <textarea
                  value={section.content}
                  onChange={e => updateSection(section.id, e.target.value)}
                  rows={3}
                  placeholder={`${section.title.toLowerCase()}…`}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-zinc-200 placeholder-zinc-700 text-[11px] resize-none focus:border-violet-700 leading-relaxed"
                />
              </div>
            ))}

            <button
              onClick={addSection}
              className="text-zinc-700 hover:text-zinc-500 transition-colors text-[10px] flex items-center gap-1"
            >
              <span>+</span> add section
            </button>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="px-3 py-2 space-y-3">
            {/* Model recommendation */}
            <div className="bg-zinc-900 border border-zinc-800 rounded p-2.5 space-y-1.5">
              <div className="text-zinc-600 text-[10px] uppercase tracking-widest">Recommended Model</div>
              <div className={`font-bold text-sm ${MODEL_COLORS[result.recommended_model] ?? 'text-zinc-200'}`}>
                {MODEL_LABELS[result.recommended_model] ?? result.recommended_model}
              </div>
              <div className="text-zinc-500 text-[10px] leading-relaxed">{result.reasoning}</div>
            </div>

            {/* Generated prompt */}
            <div className="bg-zinc-900 border border-zinc-800 rounded p-2.5 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-zinc-600 text-[10px] uppercase tracking-widest">Prompt</span>
                <button
                  onClick={copyPrompt}
                  className="text-zinc-600 hover:text-cyan-400 transition-colors text-[10px] flex items-center gap-1"
                >
                  {copied ? (
                    <span className="text-neon-green">copied!</span>
                  ) : (
                    <>
                      <svg viewBox="0 0 12 12" className="w-3 h-3" fill="none">
                        <rect x="4" y="1" width="7" height="8" rx="1" stroke="currentColor" strokeWidth="1"/>
                        <path d="M1 4h3v7H8" stroke="currentColor" strokeWidth="1" strokeLinecap="round"/>
                      </svg>
                      copy
                    </>
                  )}
                </button>
              </div>
              <pre className="text-zinc-300 text-[11px] whitespace-pre-wrap leading-relaxed font-mono">
                {result.prompt}
              </pre>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="px-3 py-2">
            <div className="bg-zinc-900 border border-red-900 rounded p-2.5 text-red-400 text-[11px]">
              {error}
            </div>
          </div>
        )}
      </div>

      {/* Build button */}
      {!result && (
        <div className="px-3 py-2 border-t border-zinc-800 shrink-0">
          <button
            onClick={build}
            disabled={!hasContent || loading}
            className="w-full py-1.5 rounded border transition-colors text-[11px] font-bold tracking-widest uppercase disabled:opacity-30 disabled:cursor-not-allowed bg-zinc-900 border-violet-800 text-violet-300 hover:bg-violet-900/30 hover:border-violet-600 hover:text-neon-violet"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="w-3 h-3 animate-spin" viewBox="0 0 12 12" fill="none">
                  <circle cx="6" cy="6" r="5" stroke="currentColor" strokeWidth="1.5" strokeDasharray="10 20" strokeLinecap="round"/>
                </svg>
                Building…
              </span>
            ) : 'Build Prompt'}
          </button>
        </div>
      )}
    </div>
  )
}
