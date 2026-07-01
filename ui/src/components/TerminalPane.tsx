import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import { WebLinksAddon } from '@xterm/addon-web-links'
import '@xterm/xterm/css/xterm.css'

export interface TerminalPaneHandle {
  focus(): void
  clear(): void
  reconnect(): void
  write(data: string): void
  readonly connected: boolean
}

interface Props {
  sessionId: string
  isActive?: boolean
  onExit?: () => void
}

const TerminalPane = forwardRef<TerminalPaneHandle, Props>(
  ({ sessionId, isActive, onExit }, ref) => {
    const containerRef = useRef<HTMLDivElement>(null)
    const termRef = useRef<Terminal | null>(null)
    const fitAddonRef = useRef<FitAddon | null>(null)
    const wsRef = useRef<WebSocket | null>(null)
    const onExitRef = useRef(onExit)
    onExitRef.current = onExit

    useImperativeHandle(ref, () => ({
      focus() {
        termRef.current?.focus()
      },
      clear() {
        termRef.current?.clear()
      },
      write(data: string) {
        termRef.current?.write(data)
      },
      reconnect() {
        wsRef.current?.close()
        connect()
      },
      get connected() {
        return wsRef.current?.readyState === WebSocket.OPEN
      },
    }))

    function sendResize() {
      const term = termRef.current
      const ws = wsRef.current
      if (!term || ws?.readyState !== WebSocket.OPEN) return
      ws.send(JSON.stringify({ type: 'resize', rows: term.rows, cols: term.cols }))
    }

    function connect() {
      const term = termRef.current
      if (!term) return
      const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/v1/terminals/${sessionId}/ws`
      const ws = new WebSocket(wsUrl)
      ws.binaryType = 'arraybuffer'
      wsRef.current = ws

      ws.onopen = () => {
        sendResize()
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
    }

    useEffect(() => {
      if (isActive) termRef.current?.focus()
    }, [isActive])

    useEffect(() => {
      if (!containerRef.current) return

      const term = new Terminal({
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

      const fitAddon = new FitAddon()
      fitAddonRef.current = fitAddon
      term.loadAddon(fitAddon)
      term.loadAddon(new WebLinksAddon())
      term.open(containerRef.current)
      fitAddon.fit()
      termRef.current = term

      term.onData((data) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(new TextEncoder().encode(data))
        }
      })

      connect()

      const ro = new ResizeObserver(() => {
        fitAddon.fit()
        sendResize()
      })
      ro.observe(containerRef.current)

      return () => {
        ro.disconnect()
        wsRef.current?.close()
        term.dispose()
        termRef.current = null
        fitAddonRef.current = null
        wsRef.current = null
      }
    }, [sessionId]) // eslint-disable-line react-hooks/exhaustive-deps

    return (
      <div className="h-full w-full overflow-hidden bg-zinc-950 p-2">
        <div ref={containerRef} className="h-full w-full" />
      </div>
    )
  }
)

TerminalPane.displayName = 'TerminalPane'

export default TerminalPane
