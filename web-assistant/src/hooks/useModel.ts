// ============================================================
// useModel hook — manages the inference Web Worker lifecycle
//
// Responsibilities:
//   - Create and own the worker instance
//   - Handle load / generate / abort commands
//   - Expose ModelProgress state to React components
//   - Stream tokens into the active message buffer
// ============================================================

import { useCallback, useEffect, useRef, useState } from 'react'
import type { ModelProgress, WorkerOutMessage, AppSettings } from '../types'
import { DEFAULT_MODEL_ID } from '../types'

// Vite worker import: ?worker suffix tells Vite to bundle this as a Worker
import InferenceWorker from '../workers/inference.worker?worker'

export interface UseModelReturn {
  /** Current model state */
  modelProgress: ModelProgress
  /** Whether a response is currently streaming */
  isGenerating: boolean
  /** Load (or reload) the model */
  loadModel: (modelId?: string) => void
  /** Send a chat completion request; returns streamed tokens via onToken */
  generate: (
    history: Array<{ role: string; content: string }>,
    settings: Pick<AppSettings, 'maxNewTokens' | 'temperature'>,
    onToken: (token: string) => void,
    onComplete: (fullText: string, backend: 'webgpu' | 'wasm') => void,
    onError: (error: string) => void
  ) => void
  /** Abort the current generation */
  abort: () => void
}

const INITIAL_PROGRESS: ModelProgress = {
  status: 'idle',
  progress: 0,
  label: 'Model not loaded',
  backend: 'unknown',
}

export function useModel(): UseModelReturn {
  const [modelProgress, setModelProgress] = useState<ModelProgress>(INITIAL_PROGRESS)
  const [isGenerating, setIsGenerating] = useState(false)

  // Worker ref persists across renders without triggering re-renders
  const workerRef = useRef<Worker | null>(null)

  // Callbacks from the current generate() call
  const onTokenRef = useRef<((token: string) => void) | null>(null)
  const onCompleteRef = useRef<((fullText: string, backend: 'webgpu' | 'wasm') => void) | null>(null)
  const onErrorRef = useRef<((error: string) => void) | null>(null)

  // Initialise worker once on mount
  useEffect(() => {
    const worker = new InferenceWorker()

    worker.addEventListener('message', (event: MessageEvent<WorkerOutMessage>) => {
      const msg = event.data
      switch (msg.type) {
        case 'progress':
          setModelProgress(msg.progress)
          break
        case 'token':
          onTokenRef.current?.(msg.token)
          break
        case 'complete':
          setIsGenerating(false)
          onCompleteRef.current?.(msg.fullText, msg.backend)
          onTokenRef.current = null
          onCompleteRef.current = null
          onErrorRef.current = null
          break
        case 'error':
          setIsGenerating(false)
          onErrorRef.current?.(msg.error)
          onTokenRef.current = null
          onCompleteRef.current = null
          onErrorRef.current = null
          break
      }
    })

    worker.addEventListener('error', (err) => {
      console.error('[useModel] Worker error:', err)
      setIsGenerating(false)
      setModelProgress({
        status: 'error',
        progress: 0,
        label: 'Worker crashed',
        backend: 'unknown',
        error: err.message,
      })
    })

    workerRef.current = worker
    return () => {
      worker.terminate()
      workerRef.current = null
    }
  }, [])

  const loadModel = useCallback((modelId: string = DEFAULT_MODEL_ID) => {
    workerRef.current?.postMessage({ type: 'load', modelId })
  }, [])

  const generate = useCallback(
    (
      history: Array<{ role: string; content: string }>,
      settings: Pick<AppSettings, 'maxNewTokens' | 'temperature'>,
      onToken: (token: string) => void,
      onComplete: (fullText: string, backend: 'webgpu' | 'wasm') => void,
      onError: (error: string) => void
    ) => {
      if (!workerRef.current) {
        onError('Worker not available')
        return
      }
      onTokenRef.current = onToken
      onCompleteRef.current = onComplete
      onErrorRef.current = onError
      setIsGenerating(true)
      workerRef.current.postMessage({
        type: 'generate',
        conversationHistory: history,
        settings,
      })
    },
    []
  )

  const abort = useCallback(() => {
    workerRef.current?.postMessage({ type: 'abort' })
    setIsGenerating(false)
  }, [])

  return { modelProgress, isGenerating, loadModel, generate, abort }
}
