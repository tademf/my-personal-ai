// ============================================================
// Settings panel — slide-in modal for user preferences
// ============================================================

import type { AppSettings } from '../types'
import { MODEL_INFO, DEFAULT_SETTINGS } from '../types'
import { X, RotateCcw, Info } from 'lucide-react'

interface Props {
  settings: AppSettings
  onUpdate: (patch: Partial<AppSettings>) => void
  onClose: () => void
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  format: (v: number) => string
  onChange: (v: number) => void
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center text-sm">
        <label className="text-slate-300 font-medium">{label}</label>
        <span className="text-accent-400 font-mono text-xs bg-surface-800 px-2 py-0.5 rounded">
          {format(value)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-accent-500 cursor-pointer h-1.5 rounded-full appearance-none bg-surface-600"
      />
      <div className="flex justify-between text-xs text-slate-600">
        <span>{format(min)}</span>
        <span>{format(max)}</span>
      </div>
    </div>
  )
}

export function SettingsPanel({ settings, onUpdate, onClose }: Props) {
  const resetSettings = () => {
    onUpdate({ ...DEFAULT_SETTINGS })
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/70 z-50 animate-fade-in"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="fixed inset-y-0 right-0 z-50 w-full max-w-sm bg-surface-900 border-l border-surface-700 flex flex-col animate-slide-up overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-surface-700">
          <h2 className="text-base font-semibold text-slate-100">Settings</h2>
          <div className="flex items-center gap-2">
            <button
              onClick={resetSettings}
              title="Reset to defaults"
              className="p-1.5 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-surface-800 transition-all"
            >
              <RotateCcw size={15} />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-surface-800 transition-all"
            >
              <X size={17} />
            </button>
          </div>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-6">

          {/* Generation settings */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-4">
              Generation
            </h3>
            <div className="space-y-5">
              <Slider
                label="Max response length"
                value={settings.maxNewTokens}
                min={64}
                max={512}
                step={32}
                format={(v) => `${v} tokens`}
                onChange={(v) => onUpdate({ maxNewTokens: v })}
              />
              <Slider
                label="Temperature"
                value={settings.temperature}
                min={0}
                max={1.5}
                step={0.05}
                format={(v) => v.toFixed(2)}
                onChange={(v) => onUpdate({ temperature: v })}
              />

              {/* Temperature explanation */}
              <div className="rounded-xl bg-surface-800 border border-surface-700 p-3 text-xs text-slate-400 flex gap-2">
                <Info size={13} className="flex-shrink-0 mt-0.5 text-accent-400" />
                <span>
                  <strong className="text-slate-300">Temperature 0</strong> = deterministic, focused.{' '}
                  <strong className="text-slate-300">1.0+</strong> = more creative but less coherent.
                  Start at 0.7.
                </span>
              </div>
            </div>
          </section>

          {/* Streaming toggle */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-4">
              Interface
            </h3>
            <div className="flex items-center justify-between rounded-xl bg-surface-800 border border-surface-700 px-4 py-3">
              <div>
                <p className="text-sm font-medium text-slate-200">Token streaming</p>
                <p className="text-xs text-slate-500 mt-0.5">Show words as they are generated</p>
              </div>
              <button
                onClick={() => onUpdate({ streamingEnabled: !settings.streamingEnabled })}
                className={`relative w-11 h-6 rounded-full transition-colors ${
                  settings.streamingEnabled ? 'bg-accent-600' : 'bg-surface-600'
                }`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    settings.streamingEnabled ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>
          </section>

          {/* System prompt */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-4">
              System Prompt
            </h3>
            <textarea
              value={settings.systemPrompt}
              onChange={(e) => onUpdate({ systemPrompt: e.target.value })}
              rows={5}
              placeholder="Describe how the AI should behave…"
              className="w-full rounded-xl bg-surface-800 border border-surface-700 px-3 py-2.5 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-accent-500 transition-colors resize-none"
            />
          </section>

          {/* Model information */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-4">
              Active Model
            </h3>
            <div className="rounded-xl bg-surface-800 border border-surface-700 p-4 space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-500">Model</span>
                <span className="text-slate-200 font-medium">{MODEL_INFO.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Parameters</span>
                <span className="text-slate-200">{MODEL_INFO.parameters}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Quantization</span>
                <span className="text-slate-200">{MODEL_INFO.quantization}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Size</span>
                <span className="text-slate-200">{MODEL_INFO.downloadSizeMB}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">License</span>
                <span className="text-success">{MODEL_INFO.license}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Publisher</span>
                <span className="text-slate-200">{MODEL_INFO.publisher}</span>
              </div>

              <div className="pt-2 border-t border-surface-700 text-xs text-slate-500 leading-relaxed">
                {MODEL_INFO.description}
              </div>
            </div>
          </section>

          {/* Privacy notice */}
          <section>
            <div className="rounded-xl bg-emerald-500/10 border border-emerald-500/20 p-4 text-sm">
              <p className="font-semibold text-emerald-400 mb-1">🔒 100% Private</p>
              <p className="text-slate-400 text-xs leading-relaxed">
                This app never sends your messages to any server. Model inference happens
                entirely inside your browser using WebGPU or WASM. Your conversations
                are stored locally in your browser's IndexedDB.
              </p>
            </div>
          </section>
        </div>
      </div>
    </>
  )
}
