// ============================================================
// Inference Web Worker
//
// This file runs in a SEPARATE BROWSER THREAD so that
// model loading and text generation NEVER freeze the UI.
//
// Architecture:
//   Main thread  <──(postMessage)──>  this Worker
//
// The worker owns the transformers.js pipeline singleton.
// It receives 'load' and 'generate' commands and replies with
// 'progress', 'token', 'complete', and 'error' messages.
//
// Model: HuggingFaceTB/SmolLM2-360M-Instruct
//   - 360M parameters, Apache 2.0 license
//   - q4 quantization (~270 MB)
//   - WebGPU first, WASM fallback
// ============================================================

import {
  pipeline,
  env,
  TextStreamer,
  type TextGenerationPipeline,
} from '@huggingface/transformers'
import type { WorkerInMessage, WorkerOutMessage, ModelProgress } from '../types'

// ── Configuration ─────────────────────────────────────────────
env.useBrowserCache = true
env.allowLocalModels = false

// Single WASM thread on mobile prevents crashes from thread exhaustion.
const wasmBackend = env.backends.onnx?.wasm

if (wasmBackend) {
  wasmBackend.numThreads = 1
}

// ── ChatML template (SmolLM2 Instruct format) ─────────────────
// Fallback template in case the tokenizer doesn't ship one.
// (The Instruct model usually has one, but this is a safety net.)
const CHATML_TEMPLATE =
  "{% for message in messages %}" +
  "{% if loop.first and messages[0]['role'] != 'system' %}" +
  "{{ '<|im_start|>system\\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\\n' }}" +
  "{% endif %}" +
  "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}" +
  "{% endfor %}" +
  "{% if add_generation_prompt %}" +
  "{{ '<|im_start|>assistant\\n' }}" +
  "{% endif %}"

// ── State ─────────────────────────────────────────────────────
let pipe: TextGenerationPipeline | null = null
let activeBackend: 'webgpu' | 'wasm' = 'wasm'
let abortController: AbortController | null = null
let isGenerating = false

// ── Helpers ───────────────────────────────────────────────────
function send(msg: WorkerOutMessage) {
  self.postMessage(msg)
}

function sendProgress(partial: Partial<ModelProgress> & { status: ModelProgress['status'] }) {
  const base: ModelProgress = {
    status: partial.status,
    progress: partial.progress ?? 0,
    label: partial.label ?? '',
    backend: partial.backend ?? 'unknown',
    error: partial.error,
    modelId: partial.modelId,
  }
  send({ type: 'progress', progress: base })
}

/** Ensure the tokenizer has a chat template before generation. */
function ensureChatTemplate() {
  if (!pipe) return
  const tok = pipe.tokenizer as unknown as { chat_template?: string }
  if (!tok.chat_template) {
    tok.chat_template = CHATML_TEMPLATE
    console.log('[Worker] Injected ChatML chat_template (model had none)')
  } else {
    console.log('[Worker] Tokenizer already has a chat_template — using it')
  }
}

/**
 * Build the prompt manually as ChatML.
 * This bypasses apply_chat_template() entirely and guarantees that
 * the system prompt + full conversation history reach the model in
 * the exact format SmolLM2-Instruct was trained on.
 */
function buildChatMLPrompt(
  history: Array<{ role: string; content: string }>
): string {
  const turns = history.map(
    (m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>`
  )
  // Append the assistant opening marker to tell the model "your turn"
  return turns.join('\n') + '\n<|im_start|>assistant\n'
}

// ── Load model ────────────────────────────────────────────────
async function loadModel(modelId: string) {
  pipe = null

  sendProgress({
    status: 'downloading',
    progress: 0,
    label: 'Initialising model download…',
    backend: 'unknown',
  })

  // Progress callback forwarded from transformers.js
  const onProgress = (p: { status: string; progress?: number; name?: string }) => {
    if (p.status === 'progress' && p.progress !== undefined) {
      sendProgress({
        status: 'downloading',
        progress: Math.round(p.progress),
        label: `Downloading model file: ${p.name ?? ''}…`,
        backend: 'unknown',
      })
    } else if (p.status === 'initiate') {
      sendProgress({
        status: 'downloading',
        progress: 0,
        label: `Starting download: ${p.name ?? ''}…`,
        backend: 'unknown',
      })
    } else if (p.status === 'done') {
      sendProgress({
        status: 'loading',
        progress: 100,
        label: 'Download complete. Loading into memory…',
        backend: 'unknown',
      })
    }
  }

  // ── createPipeline is declared above BOTH try blocks
  // so it's visible to the WebGPU attempt AND the WASM fallback.
  const createPipeline = pipeline as unknown as (
    task: string,
    model: string,
    options: Record<string, unknown>
  ) => Promise<TextGenerationPipeline>

  // ── Try WebGPU first ───────────────────────────────────────
  try {
    sendProgress({
      status: 'loading',
      progress: 0,
      label: 'Attempting WebGPU acceleration…',
      backend: 'webgpu',
    })

    pipe = await createPipeline('text-generation', modelId, {
      device: 'webgpu',
      dtype: 'q4',
      progress_callback: onProgress,
    })
    activeBackend = 'webgpu'
    ensureChatTemplate()
    sendProgress({
      status: 'ready',
      progress: 100,
      label: 'Model ready (WebGPU)',
      backend: 'webgpu',
      modelId,
    })
    return
  } catch (gpuErr) {
    console.warn('[Worker] WebGPU failed, falling back to WASM:', gpuErr)
    pipe = null
  }

  // ── WASM (CPU) fallback ────────────────────────────────────
  try {
    sendProgress({
      status: 'loading',
      progress: 0,
      label: 'Loading with WASM (CPU) — this may take a minute…',
      backend: 'wasm',
    })

    pipe = await createPipeline('text-generation', modelId, {
      device: 'wasm',
      dtype: 'q4',
      progress_callback: onProgress,
    })
    activeBackend = 'wasm'
    ensureChatTemplate()
    sendProgress({
      status: 'ready',
      progress: 100,
      label: 'Model ready (CPU/WASM)',
      backend: 'wasm',
      modelId,
    })
  } catch (wasmErr) {
    const msg = wasmErr instanceof Error ? wasmErr.message : String(wasmErr)
    sendProgress({
      status: 'error',
      progress: 0,
      label: 'Failed to load model',
      backend: 'wasm',
      error: msg,
    })
  }
}

// ── Generate text ─────────────────────────────────────────────
async function generate(
  conversationHistory: Array<{ role: string; content: string }>,
  settings: { maxNewTokens: number; temperature: number }
) {
  if (!pipe) {
    send({ type: 'error', error: 'Model is not loaded. Please load the model first.' })
    return
  }

  if (isGenerating) {
    send({ type: 'error', error: 'Already generating. Please wait or use the stop button.' })
    return
  }

  isGenerating = true
  abortController = new AbortController()

  try {
    let fullText = ''

    // Safety net: make sure the template is present even if loadModel
    // ran before this fix was deployed (e.g. cached worker state).
    ensureChatTemplate()

    // ── TextStreamer replaces the old callback_function
    const streamer = new TextStreamer(pipe.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (token: string) => {
        // If the user hit "stop", stop pushing tokens to the UI.
        // (Note: the model itself will finish running in the
        // background — real cancellation is a later improvement.)
        if (abortController?.signal.aborted) return
        if (token) {
          fullText += token
          send({ type: 'token', token })
        }
      },
    })

    // ── KEY FIX: build the prompt manually as ChatML string.
    // This bypasses apply_chat_template() entirely and guarantees
    // the system prompt + full history reach the model.
    const prompt = buildChatMLPrompt(conversationHistory)
    console.log('[Worker] Final prompt sent to model:\n', prompt)

    const result = await (pipe as TextGenerationPipeline)(prompt, {
max_new_tokens: Math.min(settings.maxNewTokens, 128),      temperature: Math.min(settings.temperature, 0.7),
      do_sample: settings.temperature > 0,
      repetition_penalty: 1.15,
      streamer,
    })

    // If streaming produced nothing (e.g. cached / non-streamed
    // result), extract text from `result` as a fallback.
    if (fullText === '' && Array.isArray(result)) {
      const generated = result[0]
      if (typeof generated === 'object' && generated !== null) {
        const genAny = generated as Record<string, unknown>
        if (typeof genAny['generated_text'] === 'string') {
          fullText = genAny['generated_text']
          // Strip the input prompt from the output if it's echoed back
          if (fullText.startsWith(prompt)) {
            fullText = fullText.slice(prompt.length).trimStart()
          }
        }
      }
    }

    if (!abortController.signal.aborted) {
      send({ type: 'complete', fullText, backend: activeBackend })
    }
  } catch (err) {
    if (abortController?.signal.aborted) {
      send({ type: 'complete', fullText: '[Generation stopped by user]', backend: activeBackend })
    } else {
      const msg = err instanceof Error ? err.message : String(err)
      send({ type: 'error', error: `Generation failed: ${msg}` })
    }
  } finally {
    isGenerating = false
    abortController = null
  }
}

// ── Message router ────────────────────────────────────────────
self.addEventListener('message', (event: MessageEvent<WorkerInMessage>) => {
  const msg = event.data
  switch (msg.type) {
    case 'load':
      loadModel(msg.modelId)
      break
    case 'generate':
      generate(msg.conversationHistory, msg.settings)
      break
    case 'abort':
      if (abortController && isGenerating) {
        abortController.abort()
      }
      break
  }
})