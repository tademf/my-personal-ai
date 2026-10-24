// ============================================================
// MessageBubble — renders a single chat message
// Handles: user, assistant (streaming), assistant (error)
// ============================================================

import React, { memo } from 'react'
import type { Message } from '../types'
import { Bot, User, AlertCircle, Cpu } from 'lucide-react'
import { clsx } from '../lib/utils'

interface Props {
  message: Message
}

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/** Simple inline code and code block renderer (no external deps) */
function renderContent(content: string): React.ReactNode {
  if (!content) return null

  // Split on triple-backtick code blocks
  const parts = content.split(/(```[\s\S]*?```)/g)
  return parts.map((part, i) => {
    if (part.startsWith('```') && part.endsWith('```')) {
      const inner = part.slice(3, -3)
      const newline = inner.indexOf('\n')
      const lang = newline > 0 ? inner.slice(0, newline).trim() : ''
      const code = newline > 0 ? inner.slice(newline + 1) : inner
      return (
        <pre
          key={i}
          className="my-3 overflow-x-auto rounded-xl bg-surface-950 p-4 text-sm font-mono text-slate-200 border border-surface-600"
        >
          {lang && (
            <div className="mb-2 text-xs text-slate-500 uppercase tracking-wide">{lang}</div>
          )}
          <code>{code}</code>
        </pre>
      )
    }

    // Inline code: `code`
    const inlineParts = part.split(/(`[^`]+`)/g)
    return (
      <React.Fragment key={i}>
        {inlineParts.map((ip, j) => {
          if (ip.startsWith('`') && ip.endsWith('`')) {
            return (
              <code key={j} className="rounded bg-surface-750 px-1.5 py-0.5 text-sm font-mono text-accent-400">
                {ip.slice(1, -1)}
              </code>
            )
          }
          // Bold: **text**
          const boldParts = ip.split(/(\*\*[^*]+\*\*)/g)
          return (
            <React.Fragment key={j}>
              {boldParts.map((bp, k) => {
                if (bp.startsWith('**') && bp.endsWith('**')) {
                  return <strong key={k} className="font-semibold">{bp.slice(2, -2)}</strong>
                }
                // Preserve newlines
                return bp.split('\n').map((line, l, arr) => (
                  <React.Fragment key={`${k}-${l}`}>
                    {line}
                    {l < arr.length - 1 && <br />}
                  </React.Fragment>
                ))
              })}
            </React.Fragment>
          )
        })}
      </React.Fragment>
    )
  })
}

export const MessageBubble = memo(function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'
  const hasError = Boolean(message.error)

  return (
    <div
      className={clsx(
        'flex w-full gap-3 animate-fade-in',
        isUser ? 'flex-row-reverse' : 'flex-row'
      )}
    >
      {/* Avatar */}
      <div
        className={clsx(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white text-sm mt-1',
          isUser
            ? 'bg-accent-500'
            : hasError
            ? 'bg-danger/20 text-danger'
            : 'bg-surface-700 border border-surface-600'
        )}
      >
        {isUser ? <User size={15} /> : hasError ? <AlertCircle size={15} /> : <Bot size={15} />}
      </div>

      {/* Bubble */}
      <div className={clsx('flex flex-col max-w-[85%] gap-1', isUser ? 'items-end' : 'items-start')}>
        <div
          className={clsx(
            'rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm',
            isUser
              ? 'bg-accent-600 text-white rounded-tr-sm'
              : hasError
              ? 'bg-danger/10 border border-danger/30 text-danger rounded-tl-sm'
              : 'bg-surface-800 text-slate-100 border border-surface-700 rounded-tl-sm'
          )}
        >
          {hasError ? (
            <div className="flex items-start gap-2">
              <AlertCircle size={14} className="mt-0.5 flex-shrink-0" />
              <span>{message.error}</span>
            </div>
          ) : message.streaming && message.content === '' ? (
            /* Typing indicator while waiting for first token */
            <span className="flex gap-1 items-center h-4">
              {[0, 150, 300].map((delay) => (
                <span
                  key={delay}
                  className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce"
                  style={{ animationDelay: `${delay}ms` }}
                />
              ))}
            </span>
          ) : (
            <div className="whitespace-pre-wrap break-words">
              {renderContent(message.content)}
              {message.streaming && (
                <span className="inline-block w-0.5 h-4 bg-slate-300 animate-pulse ml-0.5 align-middle" />
              )}
            </div>
          )}
        </div>

        {/* Timestamp + backend badge */}
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>{formatTime(message.timestamp)}</span>
          {message.backend && !isUser && (
            <span className="flex items-center gap-1">
              <Cpu size={10} />
              {message.backend.toUpperCase()}
            </span>
          )}
        </div>
      </div>
    </div>
  )
})