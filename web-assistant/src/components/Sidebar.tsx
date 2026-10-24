// ============================================================
// Sidebar — conversation list + new chat + controls
// ============================================================

import { useState } from 'react'
import type { Conversation } from '../types'
import {
  MessageSquarePlus,
  Trash2,
  MessageSquare,
  Settings,
  X,
  AlertTriangle,
} from 'lucide-react'
import { clsx } from '../lib/utils'

interface Props {
  conversations: Conversation[]
  activeId: string | null
  onNew: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
  onClearAll: () => void
  onOpenSettings: () => void
  onClose?: () => void
  isVisible: boolean
}

function formatRelativeTime(ts: number): string {
  const diff = Date.now() - ts
  const min = Math.floor(diff / 60000)
  const hr = Math.floor(diff / 3600000)
  const day = Math.floor(diff / 86400000)
  if (min < 1) return 'Just now'
  if (min < 60) return `${min}m ago`
  if (hr < 24) return `${hr}h ago`
  return `${day}d ago`
}

export function Sidebar({
  conversations,
  activeId,
  onNew,
  onSelect,
  onDelete,
  onClearAll,
  onOpenSettings,
  onClose,
  isVisible,
}: Props) {
  const [confirmClear, setConfirmClear] = useState(false)
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const handleClearAll = () => {
    if (confirmClear) {
      onClearAll()
      setConfirmClear(false)
    } else {
      setConfirmClear(true)
      setTimeout(() => setConfirmClear(false), 3000)
    }
  }

  return (
    <>
      {/* Mobile overlay backdrop */}
      {isVisible && (
        <div
          className="fixed inset-0 bg-black/60 z-40 md:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar panel */}
      <aside
        className={clsx(
          'fixed top-0 left-0 bottom-0 z-50 w-72 flex flex-col bg-surface-900 border-r border-surface-700',
          'transition-transform duration-300 ease-in-out',
          // Mobile: slide in/out
          isVisible ? 'translate-x-0' : '-translate-x-full',
          // Desktop: always visible
          'md:relative md:translate-x-0 md:z-auto'
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-4 border-b border-surface-700">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-accent-600 flex items-center justify-center">
              <MessageSquare size={14} className="text-white" />
            </div>
            <span className="font-semibold text-slate-100">AI Assistant</span>
          </div>
          {/* Mobile close button */}
          {onClose && (
            <button
              onClick={onClose}
              className="md:hidden text-slate-500 hover:text-slate-300 transition-colors"
            >
              <X size={18} />
            </button>
          )}
        </div>

        {/* New chat button */}
        <div className="px-3 pt-3 pb-2">
          <button
            onClick={() => { onNew(); onClose?.() }}
            className="w-full flex items-center gap-2 rounded-xl bg-accent-600 hover:bg-accent-500 px-4 py-2.5 text-sm font-medium text-white transition-all active:scale-95"
          >
            <MessageSquarePlus size={16} />
            New Conversation
          </button>
        </div>

        {/* Conversation list */}
        <div className="flex-1 overflow-y-auto px-2 py-1 space-y-0.5">
          {conversations.length === 0 ? (
            <div className="px-4 py-8 text-center text-sm text-slate-500">
              <MessageSquare size={32} className="mx-auto mb-2 opacity-30" />
              <p>No conversations yet.</p>
              <p className="text-xs mt-1">Start a new chat above.</p>
            </div>
          ) : (
            conversations.map((conv) => (
              <div
                key={conv.id}
                className={clsx(
                  'group relative flex items-center rounded-xl px-3 py-2.5 cursor-pointer transition-all',
                  conv.id === activeId
                    ? 'bg-surface-750 border border-surface-600'
                    : 'hover:bg-surface-800'
                )}
                onClick={() => { onSelect(conv.id); onClose?.() }}
                onMouseEnter={() => setHoveredId(conv.id)}
                onMouseLeave={() => setHoveredId(null)}
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-200 truncate">{conv.title}</p>
                  <p className="text-xs text-slate-500 mt-0.5">
                    {conv.messages.length} {conv.messages.length === 1 ? 'message' : 'messages'} ·{' '}
                    {formatRelativeTime(conv.updatedAt)}
                  </p>
                </div>
                {/* Delete button */}
                {(hoveredId === conv.id || conv.id === activeId) && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onDelete(conv.id) }}
                    className="ml-2 p-1 rounded-lg text-slate-500 hover:text-danger hover:bg-danger/10 transition-all opacity-0 group-hover:opacity-100"
                    title="Delete conversation"
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>
            ))
          )}
        </div>

        {/* Bottom controls */}
        <div className="border-t border-surface-700 p-3 space-y-1">
          {/* Clear all */}
          <button
            onClick={handleClearAll}
            className={clsx(
              'w-full flex items-center gap-2 rounded-xl px-3 py-2 text-sm transition-all',
              confirmClear
                ? 'bg-danger/20 text-danger border border-danger/30'
                : 'text-slate-400 hover:bg-surface-800 hover:text-slate-200'
            )}
          >
            {confirmClear ? <AlertTriangle size={15} /> : <Trash2 size={15} />}
            {confirmClear ? 'Tap again to confirm delete all' : 'Clear all history'}
          </button>

          {/* Settings */}
          <button
            onClick={() => { onOpenSettings(); onClose?.() }}
            className="w-full flex items-center gap-2 rounded-xl px-3 py-2 text-sm text-slate-400 hover:bg-surface-800 hover:text-slate-200 transition-all"
          >
            <Settings size={15} />
            Settings
          </button>
        </div>
      </aside>
    </>
  )
}
