// ============================================================
// useSettings hook — load/save AppSettings via IndexedDB
// ============================================================

import { useCallback, useEffect, useState } from 'react'
import type { AppSettings } from '../types'
import { DEFAULT_SETTINGS } from '../types'
import { loadSettings, saveSettings } from '../lib/db'

export function useSettings() {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    loadSettings().then((s) => {
      setSettings(s)
      setLoaded(true)
    })
  }, [])

  const updateSettings = useCallback((patch: Partial<AppSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch }
      saveSettings(next).catch(console.error)
      return next
    })
  }, [])

  return { settings, updateSettings, loaded }
}

// ── Online status hook ────────────────────────────────────────
export function useOnlineStatus(): boolean {
  const [online, setOnline] = useState(navigator.onLine)
  useEffect(() => {
    const on = () => setOnline(true)
    const off = () => setOnline(false)
    window.addEventListener('online', on)
    window.addEventListener('offline', off)
    return () => {
      window.removeEventListener('online', on)
      window.removeEventListener('offline', off)
    }
  }, [])
  return online
}
