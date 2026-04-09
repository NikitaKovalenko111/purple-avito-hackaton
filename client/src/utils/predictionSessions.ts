import type { Draft, Prediction, PredictionRequest } from "../types"

const STORAGE_KEY = "prediction_sessions_v1"
const MAX_SESSIONS = 20
const UPDATE_EVENT = "prediction-sessions-updated"

export interface PredictionSession {
  id: string
  createdAt: string
  request: PredictionRequest
  response: {
    request_id: string
    detectedMcIds: number[]
    shouldSplit: boolean
    drafts: Draft[]
  }
}

function isBrowser(): boolean {
  return typeof window !== "undefined" && typeof localStorage !== "undefined"
}

export function loadPredictionSessions(): PredictionSession[] {
  if (!isBrowser()) {
    return []
  }

  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) {
      return []
    }
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) {
      return []
    }
    return parsed.filter((item) => item && typeof item === "object") as PredictionSession[]
  } catch {
    return []
  }
}

function savePredictionSessions(sessions: PredictionSession[]): void {
  if (!isBrowser()) {
    return
  }

  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions))
  window.dispatchEvent(new CustomEvent(UPDATE_EVENT))
}

export function upsertPredictionSession(request: PredictionRequest, prediction: Prediction): void {
  const sessions = loadPredictionSessions()
  const nowIso = new Date().toISOString()
  const id = prediction.request_id

  const next: PredictionSession = {
    id,
    createdAt: nowIso,
    request,
    response: {
      request_id: prediction.request_id,
      detectedMcIds: prediction.detectedMcIds,
      shouldSplit: prediction.shouldSplit,
      drafts: prediction.drafts,
    },
  }

  const existingIndex = sessions.findIndex((s) => s.id === id)
  if (existingIndex >= 0) {
    const existing = sessions[existingIndex]
    sessions[existingIndex] = {
      ...existing,
      request,
      response: next.response,
    }
  } else {
    sessions.unshift(next)
  }

  const sorted = sessions
    .sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1))
    .slice(0, MAX_SESSIONS)

  savePredictionSessions(sorted)
}

export function subscribePredictionSessionsUpdated(handler: () => void): () => void {
  if (!isBrowser()) {
    return () => undefined
  }

  const listener = () => handler()
  window.addEventListener(UPDATE_EVENT, listener)
  return () => window.removeEventListener(UPDATE_EVENT, listener)
}
