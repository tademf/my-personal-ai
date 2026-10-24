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
  backend?: 'webgpu' | 'wasm' | 'python' | 'none'
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
  backend: 'webgpu' | 'wasm' | 'python' | 'unknown'
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
  | { type: 'complete'; fullText: string; backend: 'webgpu' | 'wasm' | 'python' }
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

/** The model identifier — our own from-scratch model */
export const DEFAULT_MODEL_ID = 'my-personal-ai-817K'

/**
 * Human-readable model information shown in the Settings panel.
 * Personal AI 817K:
 *   - 817K parameters — trained from scratch
 *   - Fine-tuned on ~20 conversation examples
 *   - Runs on your CPU via a local Python server
 *   - Character-level tokenizer (72 chars)
 *   - Base training: Shakespeare (loss 2.19)
 *   - Fine-tuning: conversations (loss 0.045)
 */
export const MODEL_INFO = {
  id: DEFAULT_MODEL_ID,
  name: 'Personal AI 817K',
  publisher: 'Trained from scratch (tademf)',
  license: 'MIT',
  parameters: '817K',
  quantization: 'fp32',
  downloadSizeMB: '<5 MB',
  description:
    'A tiny language model trained completely from scratch on a single laptop. ' +
    'It is NOT a large language model — expect short, simple responses. ' +
    'The entire pipeline (tokenizer, transformer, training, fine-tuning) was built from zero.',
  limitations: [
    'Not a large language model — expect short, simple responses',
    'No multi-turn memory (each message is independent)',
    'Cannot reason about complex questions',
    'May produce garbled text on unfamiliar prompts',
    'Quality is far below GPT-4 or Gemini',
    'Best suited for greetings and short factual questions',
  ],
} as const