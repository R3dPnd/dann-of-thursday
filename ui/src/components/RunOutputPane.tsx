import { useEffect, useRef } from 'react'

interface Props {
  projectName: string
  onExit?: () => void
}

export default function RunOutputPane({ projectName, onExit }: Props) {
  const outputRef = useRef<HTMLPreElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/v1/runs/${encodeURIComponent(projectName)}/ws`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onmessage = (evt) => {
      const el = outputRef.current
      if (!el) return
      if (evt.data === '') return  // keepalive
      const span = document.createElement('span')
      span.textContent = evt.data
      if (evt.data.includes('[process exited]')) {
        span.className = 'text-zinc-500'
        onExit?.()
      }
      el.appendChild(span)
      el.scrollTop = el.scrollHeight
    }

    ws.onclose = () => {
      const el = outputRef.current
      if (!el) return
      const span = document.createElement('span')
      span.className = 'text-zinc-600'
      span.textContent = '\n[disconnected]\n'
      el.appendChild(span)
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [projectName, onExit])

  return (
    <div className="h-full flex flex-col bg-zinc-950 p-2 overflow-hidden">
      <pre
        ref={outputRef}
        className="flex-1 overflow-auto font-mono text-xs text-zinc-300 leading-5 whitespace-pre-wrap break-all"
      />
    </div>
  )
}
