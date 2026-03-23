import { useEffect, useRef } from 'react'
import type { DannEvent } from '../types'
import { useDannStore } from './useDannState'
import { api } from '../lib/api'

const WS_PATH = '/api/v1/events'
const RECONNECT_BASE_MS = 1_000
const RECONNECT_MAX_MS = 30_000

/**
 * Connects to the orchestrator WebSocket event stream and feeds events into
 * the Zustand store. Reconnects automatically with exponential backoff.
 * Also seeds the store from GET /api/v1/state on first mount.
 */
export function useDannEvents(): void {
  const applyEvent = useDannStore((s) => s.applyEvent)
  const applySnapshot = useDannStore((s) => s.applySnapshot)
  const setWsConnected = useDannStore((s) => s.setWsConnected)
  const setProjects = useDannStore((s) => s.setProjects)

  const retryDelay = useRef(RECONNECT_BASE_MS)
  const retryTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const ws = useRef<WebSocket | null>(null)
  const unmounted = useRef(false)

  // Seed state + projects on mount
  useEffect(() => {
    api.getState().then(applySnapshot).catch(() => {/* API not yet up */})
    api.getProjects().then(setProjects).catch(() => {})

    // Refresh project list every 60 s
    const interval = setInterval(() => {
      api.getProjects().then(setProjects).catch(() => {})
    }, 60_000)

    return () => clearInterval(interval)
  }, [applySnapshot, setProjects])

  // WebSocket connection with reconnect
  useEffect(() => {
    unmounted.current = false

    function connect() {
      if (unmounted.current) return

      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const socket = new WebSocket(`${protocol}://${window.location.host}${WS_PATH}`)
      ws.current = socket

      socket.onopen = () => {
        if (unmounted.current) { socket.close(); return }
        setWsConnected(true)
        retryDelay.current = RECONNECT_BASE_MS
      }

      socket.onmessage = (evt) => {
        try {
          const event = JSON.parse(evt.data as string) as DannEvent
          applyEvent(event.type, event.payload)
        } catch {
          // malformed frame — ignore
        }
      }

      socket.onclose = () => {
        setWsConnected(false)
        if (unmounted.current) return
        retryTimer.current = setTimeout(() => {
          retryDelay.current = Math.min(retryDelay.current * 2, RECONNECT_MAX_MS)
          connect()
        }, retryDelay.current)
      }

      socket.onerror = () => {
        socket.close()
      }
    }

    connect()

    return () => {
      unmounted.current = true
      if (retryTimer.current) clearTimeout(retryTimer.current)
      ws.current?.close()
    }
  }, [applyEvent, setWsConnected])
}
