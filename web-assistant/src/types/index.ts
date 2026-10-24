// ============================================================
// Core type definitions for the Personal AI Assistant
// ============================================================

/** A single chat message in a conversation */
export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  /** True while streaming a response token-by-token */
  streaming?: boolean
  /** Error text if this message failed to generate */
  error?: string
  /** Which backend produced this response */
  backend?: 'webgpu' | 'wasm' | 'none'
}

/** A full conversation (multiple messages under a title) */
export interface Conversation {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  messages: Message[]
}

/** Model loading state machine */
export type ModelStatus =
  | 'idle'
  | 'downloading'
  | 'loading'
  | 'ready'
  | 'error'
  | 'unsupported'

/** Model loading progress details */
export interface ModelProgress {
  status: ModelStatus
  /** 0–100 percentage during download */
  progress: number
  /** Human-readable status label */
  label: string
  /** Which backend is active */
  backend: 'webgpu' | 'wasm' | 'unknown'
  /** Error message if status === 'error' */
  error?: string
  /** Loaded model identifier */
  modelId?: string
}

/** User preferences persisted in IndexedDB */
export interface AppSettings {
  theme: 'dark' | 'light' | 'system'
  maxNewTokens: number
  temperature: number
  /** Whether to show token-by-token streaming */
  streamingEnabled: boolean
  systemPrompt: string
}

/** Messages sent TO the inference Web Worker */
export type WorkerInMessage =
  | { type: 'load'; modelId: string }
  | { type: 'generate'; conversationHistory: Array<{ role: string; content: string }>; settings: Pick<AppSettings, 'maxNewTokens' | 'temperature'> }
  | { type: 'abort' }

/** Messages received FROM the inference Web Worker */
export type WorkerOutMessage =
  | { type: 'progress'; progress: ModelProgress }
  | { type: 'token'; token: string }
  | { type: 'complete'; fullText: string; backend: 'webgpu' | 'wasm' }
  | { type: 'error'; error: string }

/** Default app settings */
export const DEFAULT_SETTINGS: AppSettings = {
  theme: 'dark',
  maxNewTokens: 256,
  temperature: 0.6,
  streamingEnabled: true,
  systemPrompt:
    'You are a helpful, concise personal AI assistant running on the user\'s device. ' +
    'Answer the user\'s latest question directly. ' +
    'Use the conversation history when relevant. ' +
    'Never claim to know personal details about the user unless they told you. ' +
    'If you don\'t know something, say so briefly. ' +
    'Keep replies short and practical.',
}

/** The model we ship with — small, verified, Apache 2.0 */
export const DEFAULT_MODEL_ID = 'HuggingFaceTB/SmolLM2-360M-Instruct'

/**
 * Human-readable model information shown in the Settings panel.
 * SmolLM2-360M-Instruct:
 *   - 360M parameters
 *   - Apache 2.0 license (free for any use)
 *   - ONNX format — runs via @huggingface/transformers in-browser
 *   - Q4 quantized: ~270 MB download
 *   - WebGPU-first with automatic WASM (CPU) fallback
 */
export const MODEL_INFO = {
  id: DEFAULT_MODEL_ID,
  name: 'SmolLM2 360M Instruct',
  publisher: 'HuggingFace',
  license: 'Apache 2.0',
  parameters: '360M',
  quantization: 'q4 (4-bit)',
  downloadSizeMB: '~270 MB',
  description:
    'SmolLM2 is a compact language model optimised for on-device instruction following. ' +
    'It is NOT a large language model — expect short, simple responses. ' +
    'Best for Q&A, summaries, and simple code snippets.',
  limitations: [
    'May generate incorrect or repetitive text',
    'Context window is limited (~2048 tokens)',
    'Not suitable for complex multi-step reasoning',
    'Quality is significantly below cloud LLMs like GPT-4 or Gemini',
    'First use requires ~270 MB download (Wi-Fi recommended)',
  ],
} as const