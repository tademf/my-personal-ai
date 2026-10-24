// ============================================================
// useConversations hook — CRUD for all conversations via IndexedDB
// ============================================================

import { useCallback, useEffect, useRef, useState } from 'react'
import type { Conversation, Message } from '../types'
import {
  getAllConversations,
  getConversation,
  saveConversation,
  deleteConversation,
  clearAllConversations,
} from '../lib/db'

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

function generateTitle(firstUserMessage: string): string {
  const trimmed = firstUserMessage.trim().slice(0, 60)
  return trimmed.length < firstUserMessage.trim().length ? `${trimmed}…` : trimmed
}

export interface UseConversationsReturn {
  conversations: Conversation[]
  activeConversation: Conversation | null
  isLoading: boolean
  newConversation: () => void
  selectConversation: (id: string) => void
  addUserMessage: (content: string) => string
  addAssistantMessage: () => string
  appendToken: (messageId: string, token: string) => void
  finaliseAssistant: (messageId: string, fullText: string, backend?: 'webgpu' | 'wasm') => void
  markError: (messageId: string, error: string) => void
  removeConversation: (id: string) => void
  clearHistory: () => void
}

export function useConversations(): UseConversationsReturn {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const activeRef = useRef<Conversation | null>(null)
  activeRef.current = activeConversation

  useEffect(() => {
    getAllConversations()
      .then((all) => {
        setConversations(all)
        if (all.length > 0) {
          setActiveConversation(all[0])
        }
      })
      .catch(console.error)
      .finally(() => setIsLoading(false))
  }, [])

  /**
   * Persist + update state + update ref SYNCHRONOUSLY.
   *
   * The synchronous ref update is critical: when addUserMessage() is
   * followed by addAssistantMessage() in the same tick, the second
   * call must see the first call's result — otherwise the user message
   * is silently overwritten and never rendered.
   */
  const persist = useCallback((updated: Conversation) => {
    saveConversation(updated).catch(console.error)
    activeRef.current = updated          // ← THE FIX
    setActiveConversation(updated)
    setConversations((prev) => {
      const without = prev.filter((c) => c.id !== updated.id)
      return [updated, ...without]
    })
  }, [])

  const newConversation = useCallback(() => {
    const conv: Conversation = {
      id: generateId(),
      title: 'New conversation',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    }
    persist(conv)
  }, [persist])

  const selectConversation = useCallback(async (id: string) => {
    const conv = await getConversation(id)
    if (conv) {
      activeRef.current = conv          // ← keep ref in sync
      setActiveConversation(conv)
    }
  }, [])

  const addUserMessage = useCallback(
    (content: string): string => {
      const msgId = generateId()
      const msg: Message = {
        id: msgId,
        role: 'user',
        content,
        timestamp: Date.now(),
      }
      const current = activeRef.current
      let updated: Conversation

      if (current) {
        updated = {
          ...current,
          updatedAt: Date.now(),
          title:
            current.messages.length === 0 ? generateTitle(content) : current.title,
          messages: [...current.messages, msg],
        }
      } else {
        updated = {
          id: generateId(),
          title: generateTitle(content),
          createdAt: Date.now(),
          updatedAt: Date.now(),
          messages: [msg],
        }
      }
      persist(updated)
      return msgId
    },
    [persist]
  )

  const addAssistantMessage = useCallback((): string => {
    const msgId = generateId()
    const msg: Message = {
      id: msgId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      streaming: true,
    }
    const current = activeRef.current
    if (!current) return msgId
    const updated: Conversation = {
      ...current,
      updatedAt: Date.now(),
      messages: [...current.messages, msg],
    }
    persist(updated)
    return msgId
  }, [persist])

  const appendToken = useCallback((messageId: string, token: string) => {
    setActiveConversation((prev) => {
      if (!prev) return prev
      const messages = prev.messages.map((m) =>
        m.id === messageId ? { ...m, content: m.content + token } : m
      )
      const updated = { ...prev, messages, updatedAt: Date.now() }
      activeRef.current = updated
      return updated
    })
  }, [])

  const finaliseAssistant = useCallback(
    (messageId: string, fullText: string, backend?: 'webgpu' | 'wasm') => {
      const current = activeRef.current
      if (!current) return
      const messages = current.messages.map((m) =>
        m.id === messageId
          ? { ...m, content: fullText, streaming: false, backend }
          : m
      )
      const updated = { ...current, messages, updatedAt: Date.now() }
      persist(updated)
    },
    [persist]
  )

  const markError = useCallback(
    (messageId: string, error: string) => {
      const current = activeRef.current
      if (!current) return
      const messages = current.messages.map((m) =>
        m.id === messageId
          ? { ...m, content: '', streaming: false, error }
          : m
      )
      const updated = { ...current, messages, updatedAt: Date.now() }
      persist(updated)
    },
    [persist]
  )

  const removeConversation = useCallback((id: string) => {
    deleteConversation(id).catch(console.error)
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id)
      if (activeRef.current?.id === id) {
        const newActive = next[0] ?? null
        activeRef.current = newActive   // ← keep ref in sync
        setActiveConversation(newActive)
      }
      return next
    })
  }, [])

  const clearHistory = useCallback(() => {
    clearAllConversations().catch(console.error)
    setConversations([])
    activeRef.current = null            // ← keep ref in sync
    setActiveConversation(null)
  }, [])

  return {
    conversations,
    activeConversation,
    isLoading,
    newConversation,
    selectConversation,
    addUserMessage,
    addAssistantMessage,
    appendToken,
    finaliseAssistant,
    markError,
    removeConversation,
    clearHistory,
  }
}