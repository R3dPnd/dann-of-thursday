import { useEffect, useRef, useCallback } from 'react'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import { WebLinksAddon } from '@xterm/addon-web-links'
import '@xterm/xterm/css/xterm.css'

interface Props {
  sessionId: string
  onExit?: () => void
}

export default function TerminalPane({ sessionId, onExit }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const termRef = useRef<Terminal | null>(null)
  const fitRef = useRef<FitAddon | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const resizeObserverRef = useRef<ResizeObserver | null>(null)

  const sendResize = useCallback((rows: number, cols: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'resize', rows, cols }))
    }
  }, [])

  useEffect(() => {
    if (!containerRef.current) return

    // ── xterm.js setup ────────────────────────────────────────────────────
    const term = new Terminal({
      theme: {
        background: '#09090b',   // zinc-950
        foreground: '#e4e4e7',   // zinc-200
        cursor: '#a1a1aa',
        selectionBackground: '#3f3f46',
        black: '#18181b',
        brightBlack: '#3f3f46',
        red: '#ef4444',
        brightRed: '#f87171',
        green: '#22c55e',
        brightGreen: '#4ade80',
        yellow: '#eab308',
        brightYellow: '#facc15',
        blue: '#3b82f6',
        brightBlue: '#60a5fa',
        magenta: '#a855f7',
        brightMagenta: '#c084fc',
        cyan: '#06b6d4',
        brightCyan: '#22d3ee',
        white: '#d4d4d8',
        brightWhite: '#f4f4f5',
      },
      fontFamily: '"JetBrains Mono", "Fira Code", "Cascadia Code", Menlo, monospace',
      fontSize: 13,
      lineHeight: 1.4,
      cursorBlink: true,
      scrollback: 5000,
      allowProposedApi: true,
    })

    const fitAddon = new FitAddon()
    const webLinksAddon = new WebLinksAddon()
    term.loadAddon(fitAddon)
    term.loadAddon(webLinksAddon)
    term.open(containerRef.current)
    fitAddon.fit()

    termRef.current = term
    fitRef.current = fitAddon

    // ── WebSocket ─────────────────────────────────────────────────────────
    const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/v1/terminals/${sessionId}/ws`
    const ws = new WebSocket(wsUrl)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onopen = () => {
      // Send initial size
      sendResize(term.rows, term.cols)
    }

    ws.onmessage = (evt) => {
      if (evt.data instanceof ArrayBuffer) {
        term.write(new Uint8Array(evt.data))
      } else if (typeof evt.data === 'string') {
        try {
          const msg = JSON.parse(evt.data)
          if (msg.type === 'exit') {
            term.write('\r\n\x1b[90m[process exited]\x1b[0m\r\n')
            onExit?.()
          }
        } catch {
          term.write(evt.data)
        }
      }
    }

    ws.onclose = () => {
      term.write('\r\n\x1b[90m[disconnected]\x1b[0m\r\n')
    }

    ws.onerror = () => {
      term.write('\r\n\x1b[31m[connection error]\x1b[0m\r\n')
    }

    // Forward keystrokes to PTY
    term.onData((data) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(new TextEncoder().encode(data))
      }
    })

    // ── Resize observer ───────────────────────────────────────────────────
    const ro = new ResizeObserver(() => {
      fitAddon.fit()
      sendResize(term.rows, term.cols)
    })
    if (containerRef.current) ro.observe(containerRef.current)
    resizeObserverRef.current = ro

    return () => {
      ro.disconnect()
      ws.close()
      term.dispose()
      termRef.current = null
      fitRef.current = null
      wsRef.current = null
    }
  }, [sessionId, sendResize, onExit])

  return (
    <div
      ref={containerRef}
      className="h-full w-full overflow-hidden bg-zinc-950"
      // xterm.js needs a concrete height to render; the parent tab panel provides it
    />
  )
}
