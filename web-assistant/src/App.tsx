// ============================================================
// App.tsx — Root component
// Wires together: sidebar, chat view, settings, hooks
// ============================================================

import { useEffect, useState } from 'react'
import { Sidebar } from './components/Sidebar'
import { ChatView } from './components/ChatView'
import { SettingsPanel } from './components/SettingsPanel'
import { useModel } from './hooks/useModel'
import { useConversations } from './hooks/useConversations'
import { useSettings, useOnlineStatus } from './hooks/useSettings'

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)

  const modelHook = useModel()
  const convHook = useConversations()
  const { settings, updateSettings } = useSettings()
  const isOnline = useOnlineStatus()

  const {
    conversations,
    activeConversation,
    newConversation,
    selectConversation,
    removeConversation,
    clearHistory,
  } = convHook

  // ── Auto-load the model on refresh if it was loaded before ──
  // First-ever visit: user must click "Load Model" manually.
  // Every refresh after: loads silently from cache in 2–5s.
  useEffect(() => {
    const hasLoadedBefore =
      localStorage.getItem('model-loaded-once') === 'true'

    if (hasLoadedBefore) {
      console.log('[App] Auto-loading model from cache (previously downloaded)')
      modelHook.loadModel()
    }
    // Run only when the app mounts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Remember a successful load so future refreshes auto-load ──
  useEffect(() => {
    if (modelHook.modelProgress.status === 'ready') {
      localStorage.setItem('model-loaded-once', 'true')
    }
  }, [modelHook.modelProgress.status])

  const handleClearConversation = () => {
    if (activeConversation) {
      // Remove messages from current conversation by re-creating it
      removeConversation(activeConversation.id)
    }
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-surface-900 text-slate-100">
      {/* Sidebar */}
      <Sidebar
        conversations={conversations}
        activeId={activeConversation?.id ?? null}
        onNew={newConversation}
        onSelect={selectConversation}
        onDelete={removeConversation}
        onClearAll={clearHistory}
        onOpenSettings={() => setSettingsOpen(true)}
        onClose={() => setSidebarOpen(false)}
        isVisible={sidebarOpen}
      />

      {/* Main content */}
      <main className="flex-1 min-w-0 flex flex-col">
        <ChatView
          conversation={activeConversation}
          modelHook={modelHook}
          convHook={convHook}
          settings={settings}
          isOnline={isOnline}
          onOpenSidebar={() => setSidebarOpen(true)}
          onClearConversation={handleClearConversation}
        />
      </main>

      {/* Settings panel */}
      {settingsOpen && (
        <SettingsPanel
          settings={settings}
          onUpdate={updateSettings}
          onClose={() => setSettingsOpen(false)}
        />
      )}
    </div>
  )
}