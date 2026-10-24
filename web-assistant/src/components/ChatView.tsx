// ============================================================
// ChatView — the main conversation panel
// ============================================================

import { useCallback, useEffect, useRef, useState } from 'react'
import type { Conversation, AppSettings } from '../types'
import { MessageBubble } from './MessageBubble'
import { InputBar } from './InputBar'
import { ModelStatusBar } from './ModelStatusBar'
import type { UseModelReturn } from '../hooks/useModel'
import type { UseConversationsReturn } from '../hooks/useConversations'
import { Bot, Menu, Wifi, WifiOff, Trash2 } from 'lucide-react'

interface Props {
  conversation: Conversation | null
  modelHook: UseModelReturn
  convHook: UseConversationsReturn
  settings: AppSettings
  isOnline: boolean
  onOpenSidebar: () => void
  onClearConversation: () => void
}

export function ChatView({
  conversation,
  modelHook,
  convHook,
  settings,
  isOnline,
  onOpenSidebar,
  onClearConversation,
}: Props) {
  const [inputValue, setInputValue] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const activeAssistantIdRef = useRef<string | null>(null)

  const { modelProgress, isGenerating, loadModel, generate, abort } = modelHook
  const {
    addUserMessage,
    addAssistantMessage,
    appendToken,
    finaliseAssistant,
    markError,
  } = convHook

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [
    conversation?.messages.length,
    conversation?.messages[conversation.messages.length - 1]?.content,
  ])

  const handleSend = useCallback(() => {
    const text = inputValue.trim()
    if (!text || isGenerating || modelProgress.status !== 'ready') return

    setInputValue('')

    // Build chat history for the model BEFORE adding the user message to
    // state, so we can include the new user message exactly once.
    const history: Array<{ role: string; content: string }> = []

    if (settings.systemPrompt.trim()) {
      history.push({ role: 'system', content: settings.systemPrompt })
    }

    // Existing conversation messages
    const existingMessages = conversation?.messages ?? []
    for (const msg of existingMessages) {
      if (msg.role === 'user' || msg.role === 'assistant') {
        history.push({ role: msg.role, content: msg.content })
      }
    }

    // The new user message
    history.push({ role: 'user', content: text })

    // Now update the UI state (order doesn't matter for the model —
    // history is already built above)
    addUserMessage(text)
    const assistantId = addAssistantMessage()
    activeAssistantIdRef.current = assistantId

    generate(
      history,
      { maxNewTokens: settings.maxNewTokens, temperature: settings.temperature },
      (token) => {
        appendToken(assistantId, token)
      },
      (fullText, backend) => {
        finaliseAssistant(assistantId, fullText, backend)
        activeAssistantIdRef.current = null
      },
      (error) => {
        markError(assistantId, error)
        activeAssistantIdRef.current = null
      }
    )
  }, [
    inputValue,
    isGenerating,
    modelProgress.status,
    settings,
    conversation,
    addUserMessage,
    addAssistantMessage,
    appendToken,
    finaliseAssistant,
    markError,
    generate,
  ])

  const messages = conversation?.messages ?? []

  return (
    <div className="flex flex-col h-full bg-surface-900">
      {/* Top navbar */}
      <header className="flex items-center gap-3 px-4 py-3 border-b border-surface-700 bg-surface-900">
        <button
          onClick={onOpenSidebar}
          className="md:hidden p-2 rounded-xl text-slate-400 hover:text-slate-200 hover:bg-surface-800 transition-all"
        >
          <Menu size={20} />
        </button>

        <div className="flex-1 min-w-0">
          <h1 className="font-semibold text-slate-100 truncate text-sm md:text-base">
            {conversation?.title ?? 'Personal AI Assistant'}
          </h1>
          {conversation && (
            <p className="text-xs text-slate-500">
              {messages.length} {messages.length === 1 ? 'message' : 'messages'}
            </p>
          )}
        </div>

        <div
          className={`flex items-center gap-1.5 text-xs font-medium ${
            isOnline ? 'text-success' : 'text-slate-500'
          }`}
          title={isOnline ? 'Connected to internet' : 'Offline — inference still works if model is loaded'}
        >
          {isOnline ? <Wifi size={14} /> : <WifiOff size={14} />}
          <span className="hidden sm:inline">{isOnline ? 'Online' : 'Offline'}</span>
        </div>

        {messages.length > 0 && (
          <button
            onClick={onClearConversation}
            title="Clear this conversation"
            className="p-2 rounded-xl text-slate-500 hover:text-danger hover:bg-danger/10 transition-all"
          >
            <Trash2 size={16} />
          </button>
        )}
      </header>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState
            modelReady={modelProgress.status === 'ready'}
            onLoadModel={() => loadModel()}
            modelStatus={modelProgress.status}
            isOnline={isOnline}
          />
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-4 space-y-4">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <ModelStatusBar
        progress={modelProgress}
        isOnline={isOnline}
        onLoad={() => loadModel()}
      />

      <InputBar
        value={inputValue}
        onChange={setInputValue}
        onSend={handleSend}
        onStop={abort}
        isGenerating={isGenerating}
        modelReady={modelProgress.status === 'ready'}
      />
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────
function EmptyState({
  modelReady,
  onLoadModel,
  modelStatus,
  isOnline,
}: {
  modelReady: boolean
  onLoadModel: () => void
  modelStatus: string
  isOnline: boolean
}) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-6 text-center select-none">
      <div className="w-16 h-16 rounded-2xl bg-surface-800 border border-surface-700 flex items-center justify-center mb-4 shadow-lg">
        <Bot size={32} className="text-accent-400" />
      </div>
      <h2 className="text-xl font-semibold text-slate-100 mb-1">Personal AI Assistant</h2>
      <p className="text-slate-400 text-sm max-w-xs leading-relaxed mb-6">
        Fully private on-device AI. Your conversations stay on your device — no data is ever sent to a server.
      </p>

      {!modelReady && modelStatus === 'idle' && (
        <div className="space-y-3 w-full max-w-xs">
          <p className="text-xs text-slate-500">
            {isOnline
              ? 'Load the SmolLM2 model (~270 MB) to start chatting.'
              : 'Internet connection required to download the model for the first time.'}
          </p>
          <button
            onClick={onLoadModel}
            disabled={!isOnline}
            className={`w-full rounded-2xl px-6 py-3 text-sm font-medium transition-all ${
              isOnline
                ? 'bg-accent-600 hover:bg-accent-500 text-white shadow-glow-accent active:scale-95'
                : 'bg-surface-800 text-slate-500 cursor-not-allowed border border-surface-700'
            }`}
          >
            {isOnline ? '⬇ Download & Load Model' : '📶 Go online to download model'}
          </button>
          <p className="text-xs text-slate-600">
            First download ~270 MB (Wi-Fi recommended). Stored offline afterwards.
          </p>
        </div>
      )}

      {(modelStatus === 'downloading' || modelStatus === 'loading') && (
        <p className="text-sm text-accent-400 animate-pulse">Loading model…</p>
      )}

      {modelReady && (
        <p className="text-sm text-success">✓ Model ready — start typing below!</p>
      )}

      {modelReady && (
        <div className="mt-6 flex flex-wrap gap-2 justify-center max-w-xs">
          {[
            'Explain how transformers work',
            'Write a Python function to sort a list',
            'Summarize the key ideas of machine learning',
            'What are the best practices for REST APIs?',
          ].map((s) => (
            <button
              key={s}
              className="rounded-xl border border-surface-600 bg-surface-800 px-3 py-1.5 text-xs text-slate-300 hover:border-accent-500 hover:text-accent-400 transition-all"
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}