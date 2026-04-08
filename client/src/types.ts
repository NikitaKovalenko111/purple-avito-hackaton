export const socketDraftReady = "draft_ready"
export const socketPredictDone = "predict_done"

export interface PredictionRequest {
    sourceMcId: number
    sourceMcTitle: string
    description: string
}

export interface Prediction {
    request_id: string
    detectedMcIds: Array<number>
    shouldSplit: boolean
    drafts: Array<Draft>
}

export interface Draft {
    mcId: number
    mcTitle: string
    text: string
}