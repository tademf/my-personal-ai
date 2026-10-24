// Hook that calls the local Python server for inference
// (replaces the browser worker / SmolLM2 approach)

import { useCallback, useState } from 'react'

const SERVER_URL = 'http://localhost:5000'

export interface UsePythonModelReturn {
  isReady: boolean
  isGenerating: boolean
  modelProgress: {
    status: string
    label: string
    backend: string
    progress: number
    error?: string
    modelId?: string
  }
  loadModel: () => Promise<void>
  generate: (
    conversationHistory: Array<{ role: string; content: string }>,
    settings: { maxNewTokens: number; temperature: number },
    onToken: (token: string) => void,
    onComplete: (fullText: string, backend: 'webgpu' | 'wasm' | 'python') => void,
    onError: (error: string) => void,
  ) => Promise<void>
  abort: () => void
}

export function usePythonModel(): UsePythonModelReturn {
  const [isReady, setIsReady] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [modelProgress, setModelProgress] = useState({
    status: 'idle',
    label: 'Not loaded',
    backend: 'unknown',
    progress: 0,
  })

  const loadModel = useCallback(async () => {
    setModelProgress({
      status: 'loading',
      label: 'Checking server…',
      backend: 'python',
      progress: 0,
    })
    try {
      const res = await fetch(`${SERVER_URL}/health`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setModelProgress({
        status: 'ready',
        label: 'Python model ready',
        backend: 'python',
        progress: 100,
        modelId: 'my-personal-ai-817K',
      })
      setIsReady(true)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setModelProgress({
        status: 'error',
        label: `Cannot reach server: ${msg}`,
        backend: 'unknown',
        progress: 0,
        error: msg,
      })
      setIsReady(false)
    }
  }, [])

  const generate = useCallback(
    async (
      conversationHistory: Array<{ role: string; content: string }>,
      _settings: { maxNewTokens: number; temperature: number },
      onToken: (token: string) => void,
      onComplete: (fullText: string, backend: 'webgpu' | 'wasm' | 'python') => void,
      onError: (error: string) => void,
    ) => {
      // Use the last user message as the prompt (model has no multi-turn memory)
      const lastUser = [...conversationHistory].reverse().find((m) => m.role === 'user')
      if (!lastUser) {
        onError('No user message')
        return
      }

      setIsGenerating(true)
      try {
        const res = await fetch(`${SERVER_URL}/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: lastUser.content }),
        })

        if (!res.ok) {
          const text = await res.text()
          throw new Error(`Server error ${res.status}: ${text}`)
        }

        const data = await res.json()
        if (data.error) {
          throw new Error(data.error)
        }

        const reply: string = data.reply ?? ''
        // Simulate streaming: emit the whole reply as one token
        onToken(reply)
        onComplete(reply, 'python')
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        onError(msg)
      } finally {
        setIsGenerating(false)
      }
    },
    [],
  )

  const abort = useCallback(() => {
    // The Python server doesn't support abort — no-op
  }, [])

  return {
    isReady,
    isGenerating,
    modelProgress,
    loadModel,
    generate,
    abort,
  }
}