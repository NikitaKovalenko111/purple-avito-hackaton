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