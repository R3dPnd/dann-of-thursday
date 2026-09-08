import { useEffect } from 'react'
import { useDannStore } from './useDannState'

// Cmd/Ctrl+Shift+F toggles Focus Mode — the first app-level keyboard
// shortcut in this codebase (existing onKeyDown handlers in ChatPanel/
// TasksPanel are scoped to their own inputs). Shift avoids colliding with
// browser find (Cmd/Ctrl+F); the typing guard below avoids swallowing a
// bare "f" keystroke while the user is in a text field.
export function useFocusModeShortcut(): void {
  const toggleFocusMode = useDannStore(s => s.toggleFocusMode)

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null
      const typing = !!target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
      if (typing) return
      const combo = (e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === 'f'
      if (combo) {
        e.preventDefault()
        toggleFocusMode()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [toggleFocusMode])
}
