import { useEffect, useRef } from 'react'
import { Terminal } from '@xterm/xterm'
import { WebLinksAddon } from '@xterm/addon-web-links'
import '@xterm/xterm/css/xterm.css'

const FIXED_COLS = 220
const FIXED_ROWS = 50

interface Props {
  sessionId: string
  isActive?: boolean
  onExit?: () => void
}

export default function TerminalPane({ sessionId, isActive, onExit }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const termRef = useRef<Terminal | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  // Keep onExit in a ref so it never triggers a re-mount
  const onExitRef = useRef(onExit)
  onExitRef.current = onExit

  // Focus when tab becomes active
  useEffect(() => {
    if (isActive) termRef.current?.focus()
  }, [isActive])

  // Terminal + WebSocket — only created/destroyed when sessionId changes
  useEffect(() => {
    if (!containerRef.current) return

    const term = new Terminal({
      cols: FIXED_COLS,
      rows: FIXED_ROWS,
      theme: {
        background: '#09090b',
        foreground: '#e4e4e7',
        cursor: '#a1a1aa',
        selectionBackground: '#3f3f46',
        black: '#18181b',        brightBlack: '#3f3f46',
        red: '#ef4444',          brightRed: '#f87171',
        green: '#22c55e',        brightGreen: '#4ade80',
        yellow: '#eab308',       brightYellow: '#facc15',
        blue: '#3b82f6',         brightBlue: '#60a5fa',
        magenta: '#a855f7',      brightMagenta: '#c084fc',
        cyan: '#06b6d4',         brightCyan: '#22d3ee',
        white: '#d4d4d8',        brightWhite: '#f4f4f5',
      },
      fontFamily: '"JetBrains Mono", "Fira Code", "Cascadia Code", Menlo, monospace',
      fontSize: 13,
      lineHeight: 1.4,
      cursorBlink: true,
      scrollback: 5000,
      allowProposedApi: true,
    })

    term.loadAddon(new WebLinksAddon())
    term.open(containerRef.current)
    termRef.current = term

    const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/v1/terminals/${sessionId}/ws`
    const ws = new WebSocket(wsUrl)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'resize', rows: FIXED_ROWS, cols: FIXED_COLS }))
      term.focus()
    }

    ws.onmessage = (evt) => {
      if (evt.data instanceof ArrayBuffer) {
        term.write(new Uint8Array(evt.data))
      } else if (typeof evt.data === 'string') {
        try {
          const msg = JSON.parse(evt.data)
          if (msg.type === 'exit') {
            term.write('\r\n\x1b[90m[process exited]\x1b[0m\r\n')
            onExitRef.current?.()
          }
        } catch {
          term.write(evt.data)
        }
      }
    }

    ws.onclose = () => term.write('\r\n\x1b[90m[disconnected]\x1b[0m\r\n')
    ws.onerror = () => term.write('\r\n\x1b[31m[connection error]\x1b[0m\r\n')

    term.onData((data) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(new TextEncoder().encode(data))
      }
    })

    return () => {
      ws.close()
      term.dispose()
      termRef.current = null
      wsRef.current = null
    }
  }, [sessionId]) // ← onExit intentionally omitted — read via ref above

  return (
    <div className="h-full w-full overflow-auto bg-zinc-950 p-2">
      <div ref={containerRef} />
    </div>
  )
}
