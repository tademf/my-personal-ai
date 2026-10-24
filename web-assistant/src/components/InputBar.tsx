// ============================================================
// InputBar — multiline message input with send/stop controls
// ============================================================

import { useCallback, useEffect, useRef } from 'react'
import { Send, Square } from 'lucide-react'
import { clsx } from '../lib/utils'

interface Props {
  value: string
  onChange: (v: string) => void
  onSend: () => void
  onStop: () => void
  isGenerating: boolean
  modelReady: boolean
  disabled?: boolean
}

export function InputBar({ value, onChange, onSend, onStop, isGenerating, modelReady, disabled }: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [value])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      // Shift+Enter = new line; Enter alone = send
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        if (!isGenerating && value.trim() && modelReady) onSend()
      }
    },
    [isGenerating, value, modelReady, onSend]
  )

  const canSend = modelReady && value.trim().length > 0 && !isGenerating && !disabled
  const placeholder = !modelReady
    ? 'Load the model first to start chatting…'
    : isGenerating
    ? 'Generating…'
    : 'Message (Enter to send, Shift+Enter for new line)'

  return (
    <div className="border-t border-surface-700 bg-surface-900 px-3 py-3">
      <div className="flex items-end gap-2 rounded-2xl border border-surface-600 bg-surface-800 px-3 py-2 focus-within:border-accent-500 transition-colors shadow-inner-lg">
        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled || (!modelReady && !isGenerating)}
          rows={1}
          className={clsx(
            'flex-1 resize-none bg-transparent text-sm text-slate-100 placeholder-slate-500',
            'focus:outline-none leading-relaxed py-1',
            'disabled:cursor-not-allowed disabled:opacity-50'
          )}
          style={{ maxHeight: '160px', minHeight: '24px' }}
        />

        {/* Send / Stop button */}
        {isGenerating ? (
          <button
            onClick={onStop}
            title="Stop generation"
            className="flex-shrink-0 mb-0.5 w-8 h-8 rounded-xl flex items-center justify-center bg-danger/20 text-danger hover:bg-danger/30 transition-all active:scale-90"
          >
            <Square size={14} fill="currentColor" />
          </button>
        ) : (
          <button
            onClick={onSend}
            disabled={!canSend}
            title={canSend ? 'Send message' : 'Type a message'}
            className={clsx(
              'flex-shrink-0 mb-0.5 w-8 h-8 rounded-xl flex items-center justify-center transition-all',
              canSend
                ? 'bg-accent-600 hover:bg-accent-500 text-white active:scale-90'
                : 'bg-surface-700 text-slate-600 cursor-not-allowed'
            )}
          >
            <Send size={14} />
          </button>
        )}
      </div>

      {/* Hint text */}
      <p className="mt-1.5 px-1 text-xs text-slate-600 text-center">
        {modelReady
          ? 'All inference runs on your device — no data is sent anywhere.'
          : 'Load the model to enable on-device AI inference.'}
      </p>
    </div>
  )
}
