// ============================================================
// ModelStatus bar — shows download/load progress and model info
// ============================================================

import { useState } from 'react'
import type { ModelProgress } from '../types'
import { MODEL_INFO } from '../types'
import { Download, CheckCircle, AlertCircle, Loader2, Zap, Cpu, ChevronDown, ChevronUp } from 'lucide-react'
import { clsx } from '../lib/utils'

interface Props {
  progress: ModelProgress
  isOnline: boolean
  onLoad: () => void
}

export function ModelStatusBar({ progress, isOnline, onLoad }: Props) {
  const [showInfo, setShowInfo] = useState(false)
  const { status } = progress

  const statusIcon = {
    idle: <Download size={16} />,
    downloading: <Loader2 size={16} className="animate-spin" />,
    loading: <Loader2 size={16} className="animate-spin" />,
    ready: <CheckCircle size={16} className="text-success" />,
    error: <AlertCircle size={16} className="text-danger" />,
    unsupported: <AlertCircle size={16} className="text-warning" />,
  }[status]

  const statusColor = {
    idle: 'text-slate-400',
    downloading: 'text-accent-400',
    loading: 'text-accent-400',
    ready: 'text-success',
    error: 'text-danger',
    unsupported: 'text-warning',
  }[status]

  const backendIcon =
    progress.backend === 'webgpu' ? (
      <span className="flex items-center gap-1 text-xs text-yellow-400"><Zap size={11} />WebGPU</span>
    ) : progress.backend === 'wasm' ? (
      <span className="flex items-center gap-1 text-xs text-blue-400"><Cpu size={11} />WASM/CPU</span>
    ) : null

  return (
    <div className="border-t border-surface-700 bg-surface-850">
      {/* Main status row */}
      <div className="flex items-center gap-3 px-4 py-2.5">
        <span className={statusColor}>{statusIcon}</span>

        <div className="flex-1 min-w-0">
          <div className={clsx('text-sm font-medium truncate', statusColor)}>
            {status === 'idle' && 'Model not loaded'}
            {status === 'downloading' && `${progress.label} (${progress.progress}%)`}
            {status === 'loading' && progress.label}
            {status === 'ready' && `${MODEL_INFO.name} — ready`}
            {status === 'error' && `Error: ${progress.error ?? progress.label}`}
            {status === 'unsupported' && 'Browser unsupported'}
          </div>
          {backendIcon && status === 'ready' && (
            <div className="mt-0.5">{backendIcon}</div>
          )}
        </div>

        {/* Download progress bar */}
        {(status === 'downloading' || status === 'loading') && (
          <div className="w-24 h-1.5 rounded-full bg-surface-600 overflow-hidden">
            <div
              className="h-full rounded-full bg-accent-500 transition-all duration-300"
              style={{ width: `${progress.progress}%` }}
            />
          </div>
        )}

        {/* Load button */}
        {status === 'idle' && (
          <button
            onClick={onLoad}
            disabled={!isOnline}
            title={!isOnline ? 'Internet connection required for first download' : 'Load model'}
            className={clsx(
              'flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all',
              isOnline
                ? 'bg-accent-600 hover:bg-accent-500 text-white'
                : 'bg-surface-700 text-slate-500 cursor-not-allowed'
            )}
          >
            <Download size={12} />
            Load Model
          </button>
        )}

        {/* Retry button on error */}
        {status === 'error' && (
          <button
            onClick={onLoad}
            disabled={!isOnline}
            className="rounded-lg bg-danger/20 px-3 py-1.5 text-xs font-medium text-danger hover:bg-danger/30 transition-colors"
          >
            Retry
          </button>
        )}

        {/* Info toggle */}
        <button
          onClick={() => setShowInfo((v) => !v)}
          className="text-slate-500 hover:text-slate-300 transition-colors"
          aria-label="Model information"
        >
          {showInfo ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
      </div>

      {/* Expandable model info panel */}
      {showInfo && (
        <div className="px-4 pb-3 border-t border-surface-700 pt-3 space-y-2 text-xs animate-fade-in">
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-slate-400">
            <span className="font-medium text-slate-300">Model</span>
            <span>{MODEL_INFO.name}</span>
            <span className="font-medium text-slate-300">Parameters</span>
            <span>{MODEL_INFO.parameters}</span>
            <span className="font-medium text-slate-300">Quantization</span>
            <span>{MODEL_INFO.quantization}</span>
            <span className="font-medium text-slate-300">Download size</span>
            <span>{MODEL_INFO.downloadSizeMB}</span>
            <span className="font-medium text-slate-300">License</span>
            <span>{MODEL_INFO.license}</span>
          </div>

          <p className="text-slate-400 leading-relaxed">{MODEL_INFO.description}</p>

          <div>
            <p className="font-medium text-slate-300 mb-1">Known limitations:</p>
            <ul className="list-disc list-inside space-y-0.5 text-slate-500">
              {MODEL_INFO.limitations.map((l, i) => (
                <li key={i}>{l}</li>
              ))}
            </ul>
          </div>

          <div className="rounded-lg bg-amber-500/10 border border-amber-500/20 p-2 text-amber-400">
            <strong>Privacy:</strong> All inference runs on your device. No text is sent to any server.
          </div>

          {!isOnline && (
            <div className="rounded-lg bg-surface-700 border border-surface-600 p-2 text-slate-400">
              <strong>Offline:</strong> Model download requires internet. Once downloaded, inference works offline.
            </div>
          )}
        </div>
      )}
    </div>
  )
}
